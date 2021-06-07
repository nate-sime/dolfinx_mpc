# Copyright (C) 2020-2021 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import dolfinx
import dolfinx.io
import dolfinx_mpc
import dolfinx_mpc.utils
import numpy as np
import scipy.sparse.linalg
import ufl
from mpi4py import MPI
from petsc4py import PETSc


def test_surface_integrals():
    N = 4
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, N, N)
    V = dolfinx.VectorFunctionSpace(mesh, ("Lagrange", 1))

    # Fixed Dirichlet BC on the left wall
    def left_wall(x):
        return np.isclose(x[0], np.finfo(float).eps)

    fdim = mesh.topology.dim - 1
    left_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, left_wall)
    bc_dofs = dolfinx.fem.locate_dofs_topological(V, 1, left_facets)
    u_bc = dolfinx.Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(0.0)
    bc = dolfinx.fem.DirichletBC(u_bc, bc_dofs)
    bcs = [bc]

    # Traction on top of domain
    def top(x):
        return np.isclose(x[1], 1)
    top_facets = dolfinx.mesh.locate_entities_boundary(mesh, 1, top)
    mt = dolfinx.mesh.MeshTags(mesh, fdim, top_facets, 3)

    ds = ufl.Measure("ds", domain=mesh, subdomain_data=mt, subdomain_id=3)
    g = dolfinx.Constant(mesh, (0, -9.81e2))

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Elasticity parameters
    E = 1.0e4
    nu = 0.0
    mu = dolfinx.Constant(mesh, E / (2.0 * (1.0 + nu)))
    lmbda = dolfinx.Constant(mesh, E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))

    # Stress computation
    def sigma(v):
        return (2.0 * mu * ufl.sym(ufl.grad(v))
                + lmbda * ufl.tr(ufl.sym(ufl.grad(v))) * ufl.Identity(len(v)))

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(sigma(u), ufl.grad(v)) * ufl.dx
    rhs = ufl.inner(dolfinx.Constant(mesh, (0, 0)), v) * ufl.dx\
        + ufl.inner(g, v) * ds

    # Setup LU solver
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)

    # Setup multipointconstraint
    def l2b(li):
        return np.array(li, dtype=np.float64).tobytes()
    s_m_c = {}
    for i in range(1, N):
        s_m_c[l2b([1, i / N])] = {l2b([1, 1]): 0.8}
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_general_constraint(s_m_c, 1, 1)
    mpc.finalize()
    with dolfinx.common.Timer("~TEST: Assemble matrix old"):
        A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs)
    with dolfinx.common.Timer("~TEST: Assemble matrix (cached)"):
        A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs)
    with dolfinx.common.Timer("~TEST: Assemble matrix (C++)"):
        Acpp = dolfinx_mpc.assemble_matrix_cpp(a, mpc, bcs=bcs)
    with dolfinx.common.Timer("~TEST: Assemble vector"):
        b = dolfinx_mpc.assemble_vector(rhs, mpc)

    dolfinx.fem.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                  mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b, bcs)
    solver.setOperators(A)
    uh = b.copy()
    uh.set(0)
    solver.solve(b, uh)
    uh.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                   mode=PETSc.ScatterMode.FORWARD)
    mpc.backsubstitution(uh)

    # Write solution to file
    # u_h = dolfinx.Function(mpc.function_space())
    # u_h.vector.setArray(uh.array)
    # u_h.name = "u_mpc"
    # outfile = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "output/uh.xdmf", "w")
    # outfile.write_mesh(mesh)
    # outfile.write_function(u_h)
    # outfile.close()

    # Solve the MPC problem using a global transformation matrix
    # and numpy solvers to get reference values
    # Generate reference matrices and unconstrained solution
    A_org = dolfinx.fem.assemble_matrix(a, bcs)
    A_org.assemble()
    L_org = dolfinx.fem.assemble_vector(rhs)
    dolfinx.fem.apply_lifting(L_org, [a], [bcs])
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(L_org, bcs)

    root = 0
    comm = mesh.mpi_comm()
    with dolfinx.common.Timer("~TEST: Compare"):
        dolfinx_mpc.utils.compare_MPC_LHS(A_org, A, mpc, root=root)

        # Create global transformation matrix
        A_csr = dolfinx_mpc.utils.gather_PETScMatrix(A_org, root=root)
        K = dolfinx_mpc.utils.gather_transformation_matrix(mpc, root=root)
        L_np = dolfinx_mpc.utils.gather_PETScVector(L_org, root=root)

        if MPI.COMM_WORLD.rank == root:
            KTAK = K.T * A_csr * K
            reduced_L = K.T @ L_np
            # Solve linear system
            d = scipy.sparse.linalg.spsolve(KTAK, reduced_L)
            # Back substitution to full solution vector
            uh_numpy = K @ d

    b_np = dolfinx_mpc.utils.gather_PETScVector(b, root=root)

    A_mpc_cpp = dolfinx_mpc.utils.gather_PETScMatrix(Acpp, root=root)
    A_mpc_python = dolfinx_mpc.utils.gather_PETScMatrix(A, root=root)
    if MPI.COMM_WORLD.rank == root:
        dolfinx_mpc.utils.compare_CSR(A_mpc_cpp, A_mpc_python)

        # Compare LHS, RHS and solution with reference values
        dolfinx_mpc.utils.compare_vectors(reduced_L, b_np, mpc)
        assert np.allclose(uh.array, uh_numpy[uh.owner_range[0]:uh.owner_range[1]])
    dolfinx.common.list_timings(comm, [dolfinx.common.TimingType.wall])


def test_surface_integral_dependency():
    N = 10
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, N, N)
    V = dolfinx.VectorFunctionSpace(mesh, ("Lagrange", 1))

    def top(x):
        return np.isclose(x[1], 1)
    fdim = mesh.topology.dim - 1
    top_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, fdim, top)

    indices = np.array([], dtype=np.intc)
    values = np.array([], dtype=np.intc)
    markers = {3: top_facets}
    for key in markers.keys():
        indices = np.append(indices, markers[key])
        values = np.append(values, np.full(len(markers[key]), key,
                                           dtype=np.intc))

    mt = dolfinx.mesh.MeshTags(mesh, mesh.topology.dim - 1,
                               np.array(indices, dtype=np.intc),
                               np.array(values, dtype=np.intc))
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=mt)
    g = dolfinx.Constant(mesh, [2, 1])
    h = dolfinx.Constant(mesh, [3, 2])
    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(u, v) * ds(3) + ufl.inner(ufl.grad(u), ufl.grad(v)) * ds

    rhs = ufl.inner(g, v) * ds + ufl.inner(h, v) * ds(3)

    # Create multipoint constraint and assemble system
    def l2b(li):
        return np.array(li, dtype=np.float64).tobytes()
    s_m_c = {}
    for i in range(1, N):
        s_m_c[l2b([1, i / N])] = {l2b([1, 1]): 0.3}
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_general_constraint(s_m_c, 1, 1)
    mpc.finalize()

    with dolfinx.common.Timer("~TEST: Assemble matrix old"):
        A = dolfinx_mpc.assemble_matrix(a, mpc)
    with dolfinx.common.Timer("~TEST: Assemble matrix (cached)"):
        A = dolfinx_mpc.assemble_matrix(a, mpc)
    with dolfinx.common.Timer("~TEST: Assemble matrix (C++)"):
        Acpp = dolfinx_mpc.assemble_matrix_cpp(a, mpc)

    with dolfinx.common.Timer("~TEST: Assemble vector"):
        b = dolfinx_mpc.assemble_vector(rhs, mpc)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    # Solve the MPC problem using a global transformation matrix
    # and numpy solvers to get reference values

    # Generate reference matrices and unconstrained solution
    A_org = dolfinx.fem.assemble_matrix(a)

    A_org.assemble()
    L_org = dolfinx.fem.assemble_vector(rhs)
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    root = 0
    comm = mesh.mpi_comm()
    with dolfinx.common.Timer("~TEST: Compare"):
        dolfinx_mpc.utils.compare_MPC_LHS(A_org, A, mpc, root=root)
        dolfinx_mpc.utils.compare_MPC_RHS(L_org, b, mpc, root=root)

        A_mpc_cpp = dolfinx_mpc.utils.gather_PETScMatrix(Acpp, root=root)
        A_mpc_python = dolfinx_mpc.utils.gather_PETScMatrix(A, root=root)

        if MPI.COMM_WORLD.rank == root:
            dolfinx_mpc.utils.compare_CSR(A_mpc_cpp, A_mpc_python)

    dolfinx.common.list_timings(comm, [dolfinx.common.TimingType.wall])
