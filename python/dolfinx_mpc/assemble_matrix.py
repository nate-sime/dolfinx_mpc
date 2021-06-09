# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
import dolfinx
import dolfinx.common
import dolfinx.log
import numba
import numpy

from dolfinx_mpc import cpp

from .multipointconstraint import MultiPointConstraint
from .numba_setup import PETSc, ffi, mode, set_values_local, sink

Timer = dolfinx.common.Timer


def pack_slave_facet_info(mpc: MultiPointConstraint, active_facets: numpy.ndarray) -> numpy.ndarray:
    """
    Given an MPC and a set of active facets, return a two dimensional array, where the ith row corresponds to the
    ith facet that belongs to a slave cell.
    The first column is the location in the slave_cells array (from the MPC).
    Second column is the facet index (local to cell)
    """
    t = Timer("~MPC: Pack facet info")
    # Set up data required for exterior facet assembly
    mesh = mpc.function_space().mesh
    tdim = mesh.topology.dim
    fdim = mesh.topology.dim - 1
    # This connectivities has been computed by normal assembly
    c_to_f = mesh.topology.connectivity(tdim, fdim)
    f_to_c = mesh.topology.connectivity(fdim, tdim)
    facet_info = pack_slave_facet_info_numba(active_facets,
                                             (c_to_f.array, c_to_f.offsets),
                                             (f_to_c.array, f_to_c.offsets), mpc.slave_cells())
    t.stop()
    return facet_info


@numba.njit(fastmath=True, cache=True)
def pack_slave_facet_info_numba(active_facets, c_to_f, f_to_c, slave_cells):
    """
    Numba helper for packing slave facet info, facet index (local to cell)
    and slave index (position in slave_cell)
    """
    facet_info = numpy.zeros((len(active_facets), 2), dtype=numpy.int32)
    c_to_f_pos, c_to_f_offs = c_to_f
    f_to_c_pos, f_to_c_offs = f_to_c
    i = 0
    for facet in active_facets:
        cells = f_to_c_pos[f_to_c_offs[facet]:f_to_c_offs[facet + 1]]
        assert(len(cells) == 1)
        local_slave_cell_index = numpy.flatnonzero(slave_cells == cells[0])
        if len(local_slave_cell_index) > 0:
            local_facets = c_to_f_pos[c_to_f_offs[cells[0]]: c_to_f_offs[cells[0] + 1]]
            local_index = numpy.flatnonzero(local_facets == facet)[0]
            facet_info[i, :] = [local_index, local_slave_cell_index[0]]
            i += 1

    return facet_info[:i, :]


def assemble_matrix_cpp(form, constraint, bcs=[]):
    assert(form.arguments()[0].ufl_function_space() == form.arguments()[1].ufl_function_space())

    # Generate matrix with MPC sparsity pattern
    cpp_form = dolfinx.Form(form)._cpp_object
    A = cpp.mpc.create_matrix(cpp_form, constraint._cpp_object)
    A.zeroEntries()
    cpp.mpc.assemble_matrix(A, cpp_form, constraint._cpp_object, bcs)

    # Add one on diagonal for Dirichlet boundary conditions
    if cpp_form.function_spaces[0].id == cpp_form.function_spaces[1].id:
        A.assemblyBegin(PETSc.Mat.AssemblyType.FLUSH)
        A.assemblyEnd(PETSc.Mat.AssemblyType.FLUSH)
        dolfinx.cpp.fem.insert_diagonal(A, cpp_form.function_spaces[0], bcs, 1)
    with dolfinx.common.Timer("~MPC: A.assemble()"):
        A.assemble()
    return A


def assemble_matrix(form, constraint, bcs=[], A=None,
                    form_compiler_parameters={}, jit_parameters={}):
    """
    Assembles a ufl form given a multi point constraint and possible
    Dirichlet boundary conditions.
    NOTE: Dirichlet conditions cant be on master dofs.
    """

    timer_matrix = Timer("~MPC: Assemble matrix")
    V = constraint.function_space()
    dofmap = V.dofmap
    dofs = dofmap.list.array

    # Unravel data from MPC
    slave_cells = constraint.slave_cells()
    coefficients = constraint.coefficients()
    masters = constraint.masters_local()
    slave_cell_to_dofs = constraint.cell_to_slaves()
    cell_to_slave = slave_cell_to_dofs.array
    c_to_s_off = slave_cell_to_dofs.offsets
    slaves_local = constraint.slaves()
    num_local_slaves = constraint.num_local_slaves()

    masters_local = masters.array
    offsets = masters.offsets
    mpc_data = (slaves_local, masters_local, coefficients, offsets, slave_cells, cell_to_slave, c_to_s_off)

    # Create 1D bc indicator for matrix assembly
    num_dofs_local = (dofmap.index_map.size_local + dofmap.index_map.num_ghosts) * dofmap.index_map_bs
    is_bc = numpy.zeros(num_dofs_local, dtype=bool)
    if len(bcs) > 0:
        for bc in bcs:
            is_bc[bc.dof_indices()[0]] = True

    # Get data from mesh
    pos = V.mesh.geometry.dofmap.offsets
    x_dofs = V.mesh.geometry.dofmap.array
    x = V.mesh.geometry.x

    # Generate ufc_form
    ufc_form, _, _ = dolfinx.jit.ffcx_jit(V.mesh.mpi_comm(), form,
                                          form_compiler_parameters=form_compiler_parameters,
                                          jit_parameters=jit_parameters)

    # Generate matrix with MPC sparsity pattern
    cpp_form = dolfinx.Form(form, form_compiler_parameters=form_compiler_parameters,
                            jit_parameters=jit_parameters)._cpp_object

    # Pack constants and coefficients
    form_coeffs = dolfinx.cpp.fem.pack_coefficients(cpp_form)
    form_consts = dolfinx.cpp.fem.pack_constants(cpp_form)

    # Create sparsity pattern and matrix if not supplied
    if A is None:
        pattern = constraint.create_sparsity_pattern(cpp_form)
        with Timer("~MPC: Assemble sparsity pattern"):
            pattern.assemble()
        with Timer("~MPC: Assemble matrix (Create matrix)"):
            A = dolfinx.cpp.la.create_matrix(V.mesh.mpi_comm(), pattern)
    A.zeroEntries()

    # Assemble the matrix with all entries
    with Timer("~MPC: Assemble unconstrained matrix"):
        # dolfinx.cpp.fem.assemble_matrix_petsc(A, cpp_form, form_consts, form_coeffs, bcs)
        # FIXME: Replace once https://github.com/FEniCS/dolfinx/pull/1564 is merged
        dolfinx.cpp.fem.assemble_matrix_petsc(A, cpp_form, bcs)

    # General assembly data
    block_size = dofmap.dof_layout.block_size()
    num_dofs_per_element = dofmap.dof_layout.num_dofs

    tdim = V.mesh.topology.dim

    # Assemble over cells
    subdomain_ids = cpp_form.integral_ids(dolfinx.fem.IntegralType.cell)
    num_cell_integrals = len(subdomain_ids)

    subdomain_ids = cpp_form.integral_ids(dolfinx.fem.IntegralType.cell)
    num_cell_integrals = len(subdomain_ids)
    if num_cell_integrals > 0:
        timer = Timer("~MPC: Assemble matrix (cells)")
        V.mesh.topology.create_entity_permutations()
        permutation_info = V.mesh.topology.get_cell_permutation_info()
        for i, id in enumerate(subdomain_ids):
            cell_kernel = ufc_form.integrals(dolfinx.fem.IntegralType.cell)[i].tabulate_tensor
            active_cells = cpp_form.domains(dolfinx.fem.IntegralType.cell, id)
            assemble_slave_cells(A.handle, cell_kernel, active_cells[numpy.isin(active_cells, slave_cells)], (pos, x_dofs, x), form_coeffs,
                                 form_consts, permutation_info, dofs, block_size, num_dofs_per_element, mpc_data, is_bc)
        timer.stop()

    # Assemble over exterior facets
    subdomain_ids = cpp_form.integral_ids(dolfinx.fem.IntegralType.exterior_facet)
    num_exterior_integrals = len(subdomain_ids)

    # Get cell orientation data
    if num_exterior_integrals > 0:
        timer = Timer("~MPC: Assemble matrix (ext. facet kernel)")
        V.mesh.topology.create_entities(tdim - 1)
        V.mesh.topology.create_connectivity(tdim - 1, tdim)
        permutation_info = V.mesh.topology.get_cell_permutation_info()
        facet_permutation_info = V.mesh.topology.get_facet_permutations()
        perm = (permutation_info, facet_permutation_info)

        for i, id in enumerate(subdomain_ids):
            facet_kernel = ufc_form.integrals(dolfinx.fem.IntegralType.exterior_facet)[i].tabulate_tensor
            active_facets = cpp_form.domains(dolfinx.fem.IntegralType.exterior_facet, id)
            facet_info = pack_slave_facet_info(constraint, active_facets)
            num_facets_per_cell = len(V.mesh.topology.connectivity(tdim, tdim - 1).links(0))
            assemble_exterior_slave_facets(A.handle, facet_kernel, (pos, x_dofs, x), form_coeffs, form_consts,
                                           perm, dofs, block_size, num_dofs_per_element, facet_info, mpc_data, is_bc,
                                           num_facets_per_cell)
        timer.stop()

    # Add mpc entries on diagonal
    add_diagonal(A.handle, slaves_local[:num_local_slaves])

    with Timer("~MPC: Assemble matrix (diagonal handling)"):
        # Add one on diagonal for diriclet bc and slave dofs
        # NOTE: In the future one could use a constant in the DirichletBC
        if cpp_form.function_spaces[0].id == cpp_form.function_spaces[1].id:
            A.assemblyBegin(PETSc.Mat.AssemblyType.FLUSH)
            A.assemblyEnd(PETSc.Mat.AssemblyType.FLUSH)
            dolfinx.cpp.fem.insert_diagonal(A, cpp_form.function_spaces[0],
                                            bcs, 1.0)

    with Timer("~MPC: Assemble matrix (Finalize matrix)"):
        A.assemble()
    timer_matrix.stop()
    return A


@numba.jit
def add_diagonal(A, dofs):
    """
    Insert one on diagonal of matrix for given dofs.
    """
    ffi_fb = ffi.from_buffer
    dof_list = numpy.zeros(1, dtype=numpy.int32)
    dof_value = numpy.ones(1, dtype=PETSc.ScalarType)
    for dof in dofs:
        dof_list[0] = dof
        ierr_loc = set_values_local(A, 1, ffi_fb(dof_list), 1, ffi_fb(dof_list), ffi_fb(dof_value), mode)
        assert(ierr_loc == 0)
    sink(dof_list, dof_value)


@numba.njit
def assemble_slave_cells(A, kernel, active_cells, mesh, coeffs, constants,
                         permutation_info, dofmap, block_size, num_dofs_per_element, mpc, is_bc):
    """
    Assemble MPC contributions for cell integrals
    """
    ffi_fb = ffi.from_buffer

    # Get mesh and geometry data
    pos, x_dofmap, x = mesh

    # Empty arrays mimicking Nullpointers
    facet_index = numpy.zeros(0, dtype=numpy.int32)
    facet_perm = numpy.zeros(0, dtype=numpy.uint8)

    # NOTE: All cells are assumed to be of the same type
    geometry = numpy.zeros((pos[1] - pos[0], 3))

    A_local = numpy.zeros((block_size * num_dofs_per_element, block_size
                           * num_dofs_per_element), dtype=PETSc.ScalarType)

    # Loop over all cells
    local_dofs = numpy.zeros(block_size * num_dofs_per_element, dtype=numpy.int32)
    for slave_cell_index, cell_index in enumerate(active_cells):
        num_vertices = pos[cell_index + 1] - pos[cell_index]

        # Compute vertices of cell from mesh data
        cell = pos[cell_index]
        geometry[:, :] = x[x_dofmap[cell:cell + num_vertices]]

        # Assemble local contributions
        A_local.fill(0.0)
        kernel(ffi_fb(A_local), ffi_fb(coeffs[cell_index, :]), ffi_fb(constants), ffi_fb(geometry),
               ffi_fb(facet_index), ffi_fb(facet_perm), permutation_info[cell_index])

        # Local dof position
        local_blocks = dofmap[num_dofs_per_element
                              * cell_index: num_dofs_per_element * cell_index + num_dofs_per_element]
        # Remove all contributions for dofs that are in the Dirichlet bcs
        for j in range(num_dofs_per_element):
            for k in range(block_size):
                if is_bc[local_blocks[j] * block_size + k]:
                    A_local[j * block_size + k, :] = 0
                    A_local[:, j * block_size + k] = 0

        A_local_copy = A_local.copy()

        # Find local position of slaves
        (slaves_local, masters_local, coefficients, offsets, slave_cells, cell_to_slave, c_to_s_off) = mpc
        slave_indices = cell_to_slave[c_to_s_off[slave_cell_index]: c_to_s_off[slave_cell_index + 1]]
        num_slaves = len(slave_indices)
        local_indices = numpy.full(num_slaves, -1, dtype=numpy.int32)
        is_slave = numpy.full(block_size * num_dofs_per_element, False, dtype=numpy.bool_)
        for i in range(num_slaves):
            for j in range(num_dofs_per_element):
                for k in range(block_size):
                    if local_blocks[j] * block_size + k == slaves_local[slave_indices[i]]:
                        local_indices[i] = j * block_size + k
                        is_slave[j * block_size + k] = True
        mpc_cell = (slave_indices, masters_local, coefficients, offsets, local_indices, is_slave)
        modify_mpc_cell(A, num_dofs_per_element, block_size, A_local, local_blocks, mpc_cell)
        # Remove already assembled contribution to matrix
        A_contribution = A_local - A_local_copy

        # Expand local blocks to dofs
        for i in range(num_dofs_per_element):
            for j in range(block_size):
                local_dofs[i * block_size + j] = local_blocks[i] * block_size + j
        # Insert local contribution
        ierr_loc = set_values_local(A, block_size * num_dofs_per_element, ffi_fb(local_dofs),
                                    block_size * num_dofs_per_element, ffi_fb(local_dofs), ffi_fb(A_contribution), mode)
        assert(ierr_loc == 0)

    sink(A_contribution, local_dofs)


@numba.njit
def modify_mpc_cell(A, num_dofs, block_size, Ae, local_blocks, mpc_cell):

    (slave_indices, masters, coeffs, offsets, local_indices, is_slave) = mpc_cell
    # Flatten slave->master structure to 1D arrays
    flattened_masters = []
    flattened_slaves = []
    slaves_loc = []
    flattened_coeffs = []
    for i in range(len(slave_indices)):
        local_masters = masters[offsets[slave_indices[i]]: offsets[slave_indices[i] + 1]]
        local_coeffs = coeffs[offsets[slave_indices[i]]: offsets[slave_indices[i] + 1]]
        slaves_loc.extend([i, ] * len(local_masters))
        flattened_slaves.extend([local_indices[i], ] * len(local_masters))
        flattened_masters.extend(local_masters)
        flattened_coeffs.extend(local_coeffs)
    # Create element matrix stripped of slave entries
    Ae_original = numpy.copy(Ae)
    Ae_stripped = numpy.zeros((block_size * num_dofs, block_size * num_dofs), dtype=PETSc.ScalarType)
    for i in range(block_size * num_dofs):
        for j in range(block_size * num_dofs):
            if is_slave[i] and is_slave[j]:
                Ae_stripped[i, j] = 0
            else:
                Ae_stripped[i, j] = Ae_original[i, j]

    m0 = numpy.zeros(1, dtype=numpy.int32)
    m1 = numpy.zeros(1, dtype=numpy.int32)
    Am0m1 = numpy.zeros((1, 1), dtype=PETSc.ScalarType)
    Arow = numpy.zeros((block_size * num_dofs, 1), dtype=PETSc.ScalarType)
    Acol = numpy.zeros((1, block_size * num_dofs), dtype=PETSc.ScalarType)
    mpc_dofs = numpy.zeros(block_size * num_dofs, dtype=numpy.int32)

    num_masters = len(flattened_masters)
    ffi_fb = ffi.from_buffer
    for i in range(num_masters):
        local_index = flattened_slaves[i]
        master = flattened_masters[i]
        coeff = flattened_coeffs[i]
        Ae[:, local_index] = 0
        Ae[local_index, :] = 0
        m0[0] = master
        Arow[:, 0] = coeff * Ae_stripped[:, local_index]
        Acol[0, :] = coeff * Ae_stripped[local_index, :]
        Am0m1[0, 0] = coeff**2 * Ae_original[local_index, local_index]
        for j in range(num_dofs):
            for k in range(block_size):
                mpc_dofs[j * block_size + k] = local_blocks[j] * block_size + k
        mpc_dofs[local_index] = master
        ierr_row = set_values_local(A, block_size * num_dofs, ffi_fb(mpc_dofs), 1, ffi_fb(m0), ffi_fb(Arow), mode)
        assert(ierr_row == 0)

        # Add slave row to master row
        ierr_col = set_values_local(A, 1, ffi_fb(m0), block_size * num_dofs, ffi_fb(mpc_dofs), ffi_fb(Acol), mode)
        assert(ierr_col == 0)

        ierr_master = set_values_local(A, 1, ffi_fb(m0), 1, ffi_fb(m0), ffi_fb(Am0m1), mode)
        assert(ierr_master == 0)

        # Create[0, ...., N]\[i]
        other_masters = numpy.arange(0, num_masters - 1, dtype=numpy.int32)
        if i < num_masters - 1:
            other_masters[i] = num_masters - 1
        for j in other_masters:
            other_local_index = flattened_slaves[j]
            other_master = flattened_masters[j]
            other_coeff = flattened_coeffs[j]
            m1[0] = other_master
            if slaves_loc[i] != slaves_loc[j]:
                Am0m1[0, 0] = coeff * other_coeff * Ae_original[local_index, other_local_index]
                ierr_other_slave = set_values_local(A, 1, ffi_fb(m0), 1, ffi_fb(m1), ffi_fb(Am0m1), mode)
                assert(ierr_other_slave == 0)
            else:
                Am0m1[0, 0] = coeff * other_coeff * Ae_original[local_index, local_index]
                ierr_same_slave = set_values_local(A, 1, ffi_fb(m0), 1, ffi_fb(m1), ffi_fb(Am0m1), mode)
                assert(ierr_same_slave == 0)
    sink(Arow, Acol, Am0m1, m0, m1, mpc_dofs)


@numba.njit
def assemble_exterior_slave_facets(A, kernel, mesh, coeffs, consts, perm,
                                   dofmap, block_size, num_dofs_per_element, facet_info, mpc, is_bc, num_facets_per_cell):
    """Assemble MPC contributions over exterior facet integrals"""

    slave_cells = mpc[4]

    # Mesh data
    pos, x_dofmap, x = mesh

    # Empty arrays for facet information
    facet_index = numpy.zeros(1, dtype=numpy.int32)
    facet_perm = numpy.zeros(1, dtype=numpy.uint8)

    # NOTE: All cells are assumed to be of the same type
    geometry = numpy.zeros((pos[1] - pos[0], 3))

    A_local = numpy.zeros((num_dofs_per_element * block_size,
                           num_dofs_per_element * block_size), dtype=PETSc.ScalarType)

    # Permutation info
    cell_perms, facet_perms = perm
    # Loop over all external facets that are active
    for i in range(facet_info.shape[0]):
        # Get facet index (local to cell) and cell index (local to process)
        local_facet, slave_cell_index = facet_info[i]
        cell_index = slave_cells[slave_cell_index]
        cell = pos[cell_index]
        facet_index[0] = local_facet
        num_vertices = pos[cell_index + 1] - pos[cell_index]
        c = x_dofmap[cell:cell + num_vertices]
        geometry[:, :] = x[c]
        A_local.fill(0.0)
        facet_perm[0] = facet_perms[cell_index * num_facets_per_cell + local_facet]
        kernel(ffi.from_buffer(A_local), ffi.from_buffer(coeffs[cell_index, :]), ffi.from_buffer(consts),
               ffi.from_buffer(geometry), ffi.from_buffer(facet_index), ffi.from_buffer(facet_perm),
               cell_perms[cell_index])

        local_blocks = dofmap[num_dofs_per_element
                              * cell_index: num_dofs_per_element * cell_index + num_dofs_per_element]

        # Remove all contributions for dofs that are in the Dirichlet bcs
        for j in range(num_dofs_per_element):
            for k in range(block_size):
                if is_bc[local_blocks[j] * block_size + k]:
                    A_local[j * block_size + k, :] = 0
                    A_local[:, j * block_size + k] = 0

        A_local_copy = A_local.copy()
        #  If this slave contains a slave dof, modify local contribution
        # Find local position of slaves
        (slaves_local, masters_local, coefficients, offsets, slave_cells, cell_to_slave, c_to_s_off) = mpc
        slave_indices = cell_to_slave[c_to_s_off[slave_cell_index]: c_to_s_off[slave_cell_index + 1]]
        num_slaves = len(slave_indices)
        local_indices = numpy.full(num_slaves, -1, dtype=numpy.int32)
        is_slave = numpy.full(block_size * num_dofs_per_element, False, dtype=numpy.bool_)
        for i in range(num_slaves):
            for j in range(num_dofs_per_element):
                for k in range(block_size):
                    if local_blocks[j] * block_size + k == slaves_local[slave_indices[i]]:
                        local_indices[i] = j * block_size + k
                        is_slave[j * block_size + k] = True
        mpc_cell = (slave_indices, masters_local, coefficients, offsets, local_indices, is_slave)
        modify_mpc_cell(A, num_dofs_per_element, block_size, A_local, local_blocks, mpc_cell)

        # Remove already assembled contribution to matrix
        A_contribution = A_local - A_local_copy
        # Expand local blocks to dofs
        local_dofs = numpy.zeros(block_size * num_dofs_per_element, dtype=numpy.int32)
        for i in range(num_dofs_per_element):
            for j in range(block_size):
                local_dofs[i * block_size + j] = local_blocks[i] * block_size + j
        # Insert local contribution
        ierr_loc = set_values_local(A, block_size * num_dofs_per_element, ffi.from_buffer(local_dofs),
                                    block_size * num_dofs_per_element, ffi.from_buffer(local_dofs),
                                    ffi.from_buffer(A_contribution), mode)
        assert(ierr_loc == 0)

    sink(A_contribution, local_dofs)
