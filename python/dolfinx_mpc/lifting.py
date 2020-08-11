import numba
import numpy
import dolfinx

from .numba_setup import PETSc, ffi

Timer = dolfinx.common.Timer


def apply_lifting(b, a, constraint, bcs, x0=None, scale=1):
    dolfinx.log.log(dolfinx.log.LogLevel.INFO, "Lift vector")
    timer_lifting = Timer("~MPC: Lift vector")

    V = constraint.function_space()
    imap = V.dofmap.index_map
    if len(bcs) > 1 or len(a) > 1:
        raise NotImplementedError("Blocked mpc assembly not implemented")
    if x0 is not None:
        raise NotImplementedError("x0 lifting not implemented")

    # Unravel data from MPC
    slave_cells = constraint.slave_cells()
    coefficients = constraint.coefficients()
    masters = constraint.masters_local()
    slave_cell_to_dofs = constraint.cell_to_slaves()
    cell_to_slave = slave_cell_to_dofs.array
    c_to_s_off = slave_cell_to_dofs.offsets
    slaves_local = constraint.slaves()
    masters_local = masters.array
    offsets = masters.offsets
    mpc_data = (slaves_local, masters_local, coefficients, offsets,
                slave_cells, cell_to_slave, c_to_s_off)

    # Convert BC into two flat arrays
    crange = imap.block_size*(imap.size_local + imap.num_ghosts)
    num_local_dofs = imap.block_size*imap.size_local
    bc_markers = numpy.full(crange, False)
    bc_values = numpy.zeros(crange, dtype=numpy.float64)
    bcs = bcs[0]
    for bc in bcs:
        bc_markers[bc.dof_indices[:, 0]] = True
        bc_values[bc.dof_indices[:, 0]
                  ] = bc.value.x.array()[bc.dof_indices[:, 0]]

    # Create C++ form information
    ufc_form = dolfinx.jit.ffcx_jit(a[0])
    cpp_form = dolfinx.Form(a[0])._cpp_object
    assert(cpp_form.rank == 2)
    assert(cpp_form.function_spaces[0] == cpp_form.function_spaces[1])
    assert(constraint.function_space().element ==
           cpp_form.function_spaces[0].element)

    # Create sparsity pattern
    pattern = constraint.create_sparsity_pattern(cpp_form)
    pattern.assemble()

    dofmap = constraint.function_space().dofmap
    dofs = dofmap.list.array
    num_dofs_per_element = (dofmap.dof_layout.num_dofs *
                            dofmap.dof_layout.block_size())
    gdim = V.mesh.geometry.dim
    pos = V.mesh.geometry.dofmap.offsets
    x_dofs = V.mesh.geometry.dofmap.array
    x = V.mesh.geometry.x

    form_coeffs = dolfinx.cpp.fem.pack_coefficients(cpp_form)
    form_consts = dolfinx.cpp.fem.pack_constants(cpp_form)

    subdomain_ids = cpp_form.integrals.integral_ids(
        dolfinx.fem.IntegralType.cell)
    num_cell_integrals = len(subdomain_ids)
    if num_cell_integrals > 0:
        cpp_form.mesh.topology.create_entity_permutations()
        permutation_info = V.mesh.topology.get_cell_permutation_info()
    for i in range(num_cell_integrals):
        subdomain_id = subdomain_ids[i]
        active_cells = numpy.array(cpp_form.integrals.integral_domains(
            dolfinx.fem.IntegralType.cell, i), dtype=numpy.int64)
        is_slave_cell = numpy.isin(active_cells, slave_cells)
        cell_kernel = ufc_form.create_cell_integral(
            subdomain_id).tabulate_tensor
        no_bcs = numpy.array([], dtype=numpy.int32)
        with b.localForm() as vec:
            apply_lifting_numba(
                numpy.asarray(
                    vec), gdim, cell_kernel, form_coeffs, form_consts,
                permutation_info, dofs, num_dofs_per_element,
                active_cells, is_slave_cell, (pos, x_dofs, x),
                mpc_data, (bc_markers, bc_values),
                num_local_dofs, no_bcs)

    # Assemble over exterior facets
    subdomain_ids = cpp_form.integrals.integral_ids(
        dolfinx.fem.IntegralType.exterior_facet)
    num_exterior_integrals = len(subdomain_ids)
    if num_exterior_integrals > 0:
        raise NotImplementedError("not implemented yet")

    timer_lifting.stop()


@numba.njit
def apply_lifting_numba(
    b, gdim, kernel, coefficients, constants, permutation_info,
    dofmap, num_dofs_per_element, active_cells, is_slave_cell,
        mesh, mpc, bcs, num_local_dofs, no_bcs):
    markers, values = bcs
    (pos, x_dofs, x) = mesh

    ffi_fb = ffi.from_buffer

    (slaves, masters, master_coeffs, offsets,
     slave_cells, cell_to_slave, c_to_s_off) = mpc
    # Get mesh and geometry data
    pos, x_dofmap, x = mesh

    # Empty arrays mimicking Nullpointers
    facet_index = numpy.zeros(0, dtype=numpy.int32)
    facet_perm = numpy.zeros(0, dtype=numpy.uint8)

    # NOTE: All cells are assumed to be of the same type
    geometry = numpy.zeros((pos[1]-pos[0], gdim))
    Ae = numpy.zeros((num_dofs_per_element, num_dofs_per_element),
                     dtype=PETSc.ScalarType)
    Ag = numpy.zeros(num_dofs_per_element,
                     dtype=PETSc.ScalarType)

    # Loop over all cells
    for is_slave, cell_index in zip(is_slave_cell, active_cells):
        # Local dof position
        local_pos = dofmap[num_dofs_per_element * cell_index:
                           num_dofs_per_element * cell_index
                           + num_dofs_per_element]
        # If cell has BC lift array
        has_bc = False
        for dof in local_pos:
            if markers[dof]:
                has_bc = True
                break
        if not has_bc:
            continue
        # Compute vertices of cell from mesh data
        # FIXME: This assumes a particular geometry dof layout
        num_vertices = pos[cell_index + 1] - pos[cell_index]
        cell = pos[cell_index]
        c = x_dofmap[cell:cell + num_vertices]
        for j in range(num_vertices):
            for k in range(gdim):
                geometry[j, k] = x[c[j], k]

        Ae.fill(0.0)
        # FIXME: Numba does not support edge reflections
        kernel(ffi_fb(Ae),
               ffi_fb(coefficients[cell_index, :]),
               ffi_fb(constants), ffi_fb(geometry),
               ffi_fb(facet_index), ffi_fb(facet_perm),
               permutation_info[cell_index])
        # Compute local contribution of -Ag
        Ag.fill(0.0)
        if not is_slave:
            for i, dof in enumerate(local_pos):
                if markers[dof]:
                    Ag -= Ae[:, i] * values[dof]
            for i, dof in enumerate(local_pos):
                if not markers[dof]:
                    b[dof] += Ag[i]

        else:
            for i, dof in enumerate(local_pos):
                if markers[dof]:
                    Ag -= Ae[:, i] * values[dof]
            slave_cell_index = numpy.flatnonzero(slave_cells == cell_index)[0]
            cell_slaves = cell_to_slave[c_to_s_off[slave_cell_index]:
                                        c_to_s_off[slave_cell_index+1]]
            for i, dof in enumerate(local_pos):
                is_slave_dof = numpy.flatnonzero(slaves[cell_slaves] == dof)
                if len(is_slave_dof) > 0:
                    local_index = is_slave_dof[0]
                    local_masters = masters[offsets[local_index]:
                                            offsets[local_index+1]]
                    local_coeffs = master_coeffs[offsets[local_index]:
                                                 offsets[local_index+1]]
                    for master, coeff in zip(local_masters, local_coeffs):
                        b[master] += coeff*Ag[i]

                else:
                    if not markers[dof]:
                        b[dof] += Ag[i]
