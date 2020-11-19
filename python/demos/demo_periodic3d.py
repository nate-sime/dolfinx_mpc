
# This demo program solves Poisson's equation
#
#     - div grad u(x, y) = f(x, y)
#
# on the unit square with homogeneous Dirichlet boundary conditions
# at y = 0, 1 and periodic boundary conditions at x = 0, 1.
#
# Original implementation in DOLFIN by Kristian B. Oelgaard and Anders Logg
# This implementation can be found at:
# https://bitbucket.org/fenics-project/dolfin/src/master/python/demo/documented/periodic/demo_periodic.py
#
# Copyright (C) Jørgen S. Dokken 2020.
#
# This file is part of DOLFINX_MPCX.
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


import dolfinx
import dolfinx.io
import dolfinx_mpc
import dolfinx_mpc.utils
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import ufl
from mpi4py import MPI
from petsc4py import PETSc

# Get PETSc int and scalar types
if np.dtype(PETSc.ScalarType).kind == 'c':
    complex = True
else:
    complex = False


def demo_periodic3D(celltype, out_periodic):
    # Create mesh and finite element
    if celltype == dolfinx.cpp.mesh.CellType.tetrahedron:
        # Tet setup
        N = 4
        mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, N, N, N)
        V = dolfinx.FunctionSpace(mesh, ("CG", 2))
    else:
        # Hex setup
        N = 10
        mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, N, N, N,
                                    dolfinx.cpp.mesh.CellType.hexahedron)
        V = dolfinx.FunctionSpace(mesh, ("CG", 1))

    # Create Dirichlet boundary condition
    u_bc = dolfinx.function.Function(V)
    with u_bc.vector.localForm() as u_local:
        u_local.set(0.0)

    def DirichletBoundary(x):
        return np.logical_or(np.logical_or(np.isclose(x[1], 0),
                                           np.isclose(x[1], 1)),
                             np.logical_or(np.isclose(x[2], 0),
                                           np.isclose(x[2], 1)))

    mesh.topology.create_connectivity(2, 1)
    geometrical_dofs = dolfinx.fem.locate_dofs_geometrical(
        V, DirichletBoundary)
    bc = dolfinx.fem.DirichletBC(u_bc, geometrical_dofs)
    bcs = [bc]

    def PeriodicBoundary(x):
        return np.isclose(x[0], 1)

    facets = dolfinx.mesh.locate_entities_boundary(
        mesh, mesh.topology.dim - 1, PeriodicBoundary)
    mt = dolfinx.MeshTags(mesh, mesh.topology.dim - 1,
                          facets, np.full(len(facets), 2, dtype=np.int32))

    def periodic_relation(x):
        out_x = np.zeros(x.shape)
        out_x[0] = 1 - x[0]
        out_x[1] = x[1]
        out_x[2] = x[2]
        return out_x

    with dolfinx.common.Timer("~Periodic: New init"):
        mpc = dolfinx_mpc.MultiPointConstraint(V)
        mpc.create_periodic_constraint(mt, 2, periodic_relation, bcs)
        mpc.finalize()

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

    x = ufl.SpatialCoordinate(mesh)
    dx = x[0] - 0.9
    dy = x[1] - 0.5
    dz = x[2] - 0.1
    f = x[0] * ufl.sin(5.0 * ufl.pi * x[1]) \
        + 1.0 * ufl.exp(-(dx * dx + dy * dy + dz * dz) / 0.02)

    rhs = ufl.inner(f, v) * ufl.dx

    with dolfinx.common.Timer("~Periodic: Assemble LHS and RHS"):
        A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs)
        b = dolfinx_mpc.assemble_vector(rhs, mpc)

    ai, aj, av = A.getValuesCSR()
    A_mpc_scipy = scipy.sparse.csr_matrix((av, aj, ai))
    # Apply boundary conditions
    dolfinx.fem.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                  mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b, bcs)

    # Solve Linear problem
    solver = PETSc.KSP().create(MPI.COMM_WORLD)

    if complex:
        solver.setType(PETSc.KSP.Type.PREONLY)
        solver.getPC().setType(PETSc.PC.Type.LU)
    else:
        opts = PETSc.Options()
        opts["ksp_type"] = "cg"
        opts["ksp_rtol"] = 1.0e-6
        opts["pc_type"] = "hypre"
        opts['pc_hypre_type'] = 'boomeramg'
        opts["pc_hypre_boomeramg_max_iter"] = 1
        opts["pc_hypre_boomeramg_cycle_type"] = "v"
        # opts["pc_hypre_boomeramg_print_statistics"] = 1
        solver.setFromOptions()

    with dolfinx.common.Timer("~Periodic: Solve MPC problem"):
        solver.setOperators(A)
        uh = b.copy()
        uh.set(0)
        solver.solve(b, uh)
        uh.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                       mode=PETSc.ScatterMode.FORWARD)
        mpc.backsubstitution(uh)
        # solver.view()
        it = solver.getIterationNumber()
        print("Constrained solver iterations {0:d}".format(it))

    # Write solution to file
    u_h = dolfinx.Function(mpc.function_space())
    u_h.vector.setArray(uh.array)
    if celltype == dolfinx.cpp.mesh.CellType.tetrahedron:
        ext = "tet"
    else:
        ext = "hex"

    mesh.name = "mesh_" + ext
    u_h.name = "u_" + ext

    out_periodic.write_mesh(mesh)
    out_periodic.write_function(u_h, 0.0,
                                "Xdmf/Domain/"
                                + "Grid[@Name='{0:s}'][1]"
                                .format(mesh.name))

    print("----Verification----")
    # --------------------VERIFICATION-------------------------
    A_org = dolfinx.fem.assemble_matrix(a, bcs)
    A_org.assemble()

    # Plot sparisty patterns (before creating dense matrix)
    ai, aj, av = A_org.getValuesCSR()
    A_scipy = scipy.sparse.csr_matrix((av, aj, ai))
    fig, axs = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)
    axs[0].grid("on", zorder=-1)
    axs[1].grid("on", zorder=-1)
    axs[0].tick_params(axis='both', which='major', labelsize=22)
    axs[0].spy(A_scipy, color="r", markersize=1.7, markeredgewidth=0.0, label="Neumann", zorder=3)
    axs[0].spy(A_mpc_scipy, color="b", marker="x", markersize=1.7, markeredgewidth=0.8, label="Periodic", zorder=2)
    axs[0].add_patch(matplotlib.patches.Rectangle((800, 800), 150, 150,
                                                  linewidth=1, edgecolor='k', facecolor='none', zorder=4))
    axs[0].legend(markerscale=8, fontsize=20)
    axs[1].spy(A_scipy, color="r", markersize=3, markeredgewidth=0.0, label="Neumann", zorder=3)
    axs[1].tick_params(axis='both', which='major', labelsize=22)
    axs[1].spy(A_mpc_scipy, color="b", marker="x", markersize=3, markeredgewidth=0.8, label="Periodic", zorder=4)
    axs[1].legend(markerscale=2, fontsize=20)
    axs[1].set_xlim([1000, 850])
    axs[1].set_ylim([850, 1000])

    if MPI.COMM_WORLD.size > 1:
        plt.savefig("sp_per_3D_rank{0:d}_{1:s}.png".format(MPI.COMM_WORLD.rank, ext), dpi=200)
    else:
        plt.savefig("sp_per_3D_{0:s}.png".format(ext), dpi=300)

    L_org = dolfinx.fem.assemble_vector(rhs)
    dolfinx.fem.apply_lifting(L_org, [a], [bcs])
    L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                      mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(L_org, bcs)
    solver.setOperators(A_org)
    u_ = dolfinx.Function(V)
    with dolfinx.common.Timer("~Periodic: Unconstrained solve"):
        solver.solve(L_org, u_.vector)
        it = solver.getIterationNumber()
    print("Unconstrained solver iterations: {0:d}".format(it))
    u_.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                          mode=PETSc.ScatterMode.FORWARD)
    u_.name = "u_" + ext + "_unconstrained"
    out_periodic.write_function(u_, 0.0,
                                "Xdmf/Domain/"
                                + "Grid[@Name='{0:s}'][1]"
                                .format(mesh.name))

    # Create global transformation matrix
    K = dolfinx_mpc.utils.create_transformation_matrix(V, mpc)
    # Create reduced A
    A_global = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A_org)
    reduced_A = np.matmul(np.matmul(K.T, A_global), K)

    # Created reduced L
    vec = dolfinx_mpc.utils.PETScVector_to_global_numpy(L_org)
    reduced_L = np.dot(K.T, vec)
    # Solve linear system
    d = np.linalg.solve(reduced_A, reduced_L)
    # Back substitution to full solution vector
    uh_numpy = np.dot(K, d)

    # Compare LHS, RHS and solution with reference values
    A_np = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A)
    b_np = dolfinx_mpc.utils.PETScVector_to_global_numpy(b)
    dolfinx_mpc.utils.compare_vectors(reduced_L, b_np, mpc)
    dolfinx_mpc.utils.compare_matrices(reduced_A, A_np, mpc)
    assert np.allclose(uh.array, uh_numpy[uh.owner_range[0]:uh.owner_range[1]])


if __name__ == "__main__":
    fname = "results/demo_periodic3d.xdmf"
    out_periodic = dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                                       fname, "w")
    for celltype in [dolfinx.cpp.mesh.CellType.tetrahedron,
                     dolfinx.cpp.mesh.CellType.hexahedron]:
        demo_periodic3D(celltype, out_periodic)
    out_periodic.close()
    dolfinx.common.list_timings(
        MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])
