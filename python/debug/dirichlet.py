# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from IPython import embed
import dolfinx_mpc
import dolfinx_mpc.utils
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
import dolfinx.log

# Create mesh and function space
mesh = dolfinx.UnitSquareMesh(
    MPI.COMM_WORLD, 2, 2, dolfinx.cpp.mesh.CellType.quadrilateral)
V = dolfinx.FunctionSpace(mesh, ("Lagrange", 1))

# Test against generated code and general assembler
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(mesh)
f = x[0]+2*x[1]  # ufl.sin(2*ufl.pi*x[0])*ufl.sin(ufl.pi*x[1])
a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
rhs = ufl.inner(f, v)*ufl.dx


def bottom_corner(x):
    return np.isclose(x, [[0], [0], [0]]).all(axis=0)


def corner(x):
    return np.isclose(x, [[1], [0], [0]]).all(axis=0)


# Fix bottom corner
u_bc = dolfinx.function.Function(V)
with u_bc.vector.localForm() as u_local:
    u_local.set(3)
bottom_dofs = dolfinx.fem.locate_dofs_geometrical(V, bottom_corner)
bc_bottom = dolfinx.fem.DirichletBC(u_bc, bottom_dofs)
bc_corner = dolfinx.fem.DirichletBC(
    u_bc, dolfinx.fem.locate_dofs_geometrical(V, corner))
bcs = [bc_bottom]  # , bc_corner]


def l2b(li):
    return np.array(li, dtype=np.float64).tobytes()


s_m_c = {l2b([1, 1]): {
    l2b([0, 0]): 5},
    l2b([0, 1]): {l2b([1, 0]): 2}}
mpc = dolfinx_mpc.MultiPointConstraint(V)
mpc.create_general_constraint(s_m_c)
mpc.finalize()
print(V.dim, mpc.slaves(), mpc.masters_local().array)
A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs)
b = dolfinx_mpc.assemble_vector(rhs, mpc, bcs=bcs)
# dolfinx.fem.apply_lifting(b, [a], [bcs])
# b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
#               mode=PETSc.ScatterMode.REVERSE)
# dolfinx.fem.set_bc(b, bcs)
dolfinx_mpc.apply_lifting(b, [a], mpc, [bcs])
b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
              mode=PETSc.ScatterMode.REVERSE)

solver = PETSc.KSP().create(MPI.COMM_WORLD)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)
solver.setOperators(A)

# Solve
uh = b.copy()
uh.set(0)
solver.solve(b, uh)
uh.ghostUpdate(addv=PETSc.InsertMode.INSERT,
               mode=PETSc.ScatterMode.FORWARD)
mpc.backsubstitution(uh)


# LHS
K = dolfinx_mpc.utils.create_transformation_matrix(V, mpc)
A_org = dolfinx.fem.assemble_matrix(a, bcs=bcs)
A_org.assemble()
A_org_np = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A_org)
reduced_A = np.matmul(np.matmul(K.T, A_org_np), K)

# RHS
L_org = dolfinx.fem.assemble_vector(rhs)
dolfinx.fem.apply_lifting(L_org, [a], [bcs])
L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                  mode=PETSc.ScatterMode.REVERSE)
dolfinx.fem.set_bc(L_org, bcs)
vec = dolfinx_mpc.utils.PETScVector_to_global_numpy(L_org)
reduced_L = np.dot(K.T, vec)

d = np.linalg.solve(reduced_A, reduced_L)
uh_numpy = np.dot(K, d)

# Transfer data from the MPC problem to numpy arrays for comparison
A_mpc_np = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A)
b_np = dolfinx_mpc.utils.PETScVector_to_global_numpy(b)
print(reduced_L, "\n", b_np)
dolfinx_mpc.utils.compare_matrices(reduced_A, A_mpc_np, mpc)
dolfinx_mpc.utils.compare_vectors(reduced_L, b_np, mpc)
assert np.allclose(uh.array, uh_numpy[uh.owner_range[0]:
                                      uh.owner_range[1]])
