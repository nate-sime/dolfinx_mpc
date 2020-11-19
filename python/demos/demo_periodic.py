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
import gmsh
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

# -------------------------- Create mesh --------------------------
res_left = 0.075
res_right = 0.05
# If res_left< res_right we have DirichletBC on facets in the same cell as the slave dof
# which is not supported
assert(res_left > res_right)
L = 1
H = 1
vol_marker = 1
left_marker, right_marker, tb_marker = 3, 2, 1
gmsh.initialize()
if MPI.COMM_WORLD.rank == 0:
    gmsh.model.add("Square duct")
    rect = gmsh.model.occ.addRectangle(0, 0, 0, L, H)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [rect], vol_marker)
    gmsh.model.setPhysicalName(2, vol_marker, "Volume")

    surfaces = gmsh.model.occ.getEntities(dim=1)
    tb = []
    left = []
    right = []
    # Find and tag facets
    for surface in surfaces:
        com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
        if np.allclose(com, [0, H / 2, 0]):
            left.append(surface[1])
        elif np.allclose(com, [L, H / 2, 0]):
            right.append(surface[1])
        elif np.isclose(com[1], 0) or np.isclose(com[1], H):
            tb.append(surface[1])

    # Add physical markers
    gmsh.model.addPhysicalGroup(1, tb, tb_marker)
    gmsh.model.setPhysicalName(1, tb_marker, "Walls")
    gmsh.model.addPhysicalGroup(1, left, left_marker)
    gmsh.model.setPhysicalName(1, left_marker, "Fluid inlet")
    gmsh.model.addPhysicalGroup(1, right, right_marker)
    gmsh.model.setPhysicalName(1, right_marker, "Fluid outlet")
    # Specify mesh resolution fields
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "EdgesList", right)
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "IField", 1)
    gmsh.model.mesh.field.setNumber(2, "LcMin", res_right)
    gmsh.model.mesh.field.setNumber(2, "LcMax", res_left)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.1 * L)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 0.3 * L)
    gmsh.model.mesh.field.add("Min", 7)
    gmsh.model.mesh.field.setNumbers(7, "FieldsList", [2])
    gmsh.model.mesh.field.setAsBackgroundMesh(7)
    # Generate mesh
    gmsh.option.setNumber("Mesh.MaxNumThreads1D", MPI.COMM_WORLD.size)
    gmsh.option.setNumber("Mesh.MaxNumThreads2D", MPI.COMM_WORLD.size)
    gmsh.option.setNumber("Mesh.MaxNumThreads3D", MPI.COMM_WORLD.size)
    gmsh.model.mesh.generate(2)

mesh, ft = dolfinx_mpc.utils.gmsh_model_to_mesh(gmsh.model, facet_data=True, gdim=2)
gmsh.finalize()


# ---------------- Define function space and BCs -------------------

def periodic_relation(x):
    """ Mapping the left boundary to the right boundary """
    out_x = np.zeros(x.shape)
    out_x[0] = 1 - x[0]
    out_x[1] = x[1]
    out_x[2] = x[2]
    return out_x


V = dolfinx.FunctionSpace(mesh, ("CG", 1))

# DiricletBC for top and bottom facets
u_bc = dolfinx.function.Function(V)
with u_bc.vector.localForm() as u_local:
    u_local.set(0.0)
tb_facets = ft.indices[ft.values == tb_marker]
topological_dofs = dolfinx.fem.locate_dofs_topological(V, 1, tb_facets)
bc = dolfinx.fem.DirichletBC(u_bc, topological_dofs)
bcs = [bc]


with dolfinx.common.Timer("~PERIODIC: Initialize MPC"):
    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_periodic_constraint(ft, left_marker, periodic_relation, bcs)
    mpc.finalize()

# Define variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

x = ufl.SpatialCoordinate(mesh)
dx = x[0] - 0.9
dy = x[1] - 0.5
f = x[0] * ufl.sin(5.0 * ufl.pi * x[1]) \
    + 1.0 * ufl.exp(-(dx * dx + dy * dy) / 0.02)

rhs = ufl.inner(f, v) * ufl.dx


# Setup MPC system
with dolfinx.common.Timer("~PERIODIC: Assemble LHS and RHS"):
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


with dolfinx.common.Timer("~PERIODIC: Solve old"):
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
u_h.name = "u_mpc"
outfile = dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                              "results/demo_periodic.xdmf", "w")
outfile.write_mesh(mesh)
outfile.write_function(u_h)


# --------------------VERIFICATION-------------------------
A_org = dolfinx.fem.assemble_matrix(a, bcs)
A_org.assemble()

# Plot sparisty patterns (before making the matrix dense)
ai, aj, av = A_org.getValuesCSR()
A_scipy = scipy.sparse.csr_matrix((av, aj, ai))
fig, axs = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)
axs[0].grid("on", zorder=-1)
axs[1].grid("on", zorder=-1)
axs[0].tick_params(axis='both', which='major', labelsize=22)
axs[0].spy(A_scipy, color="r", markersize=1.7, markeredgewidth=0.0, label="Neumann", zorder=3)
axs[0].spy(A_mpc_scipy, color="b", markersize=1.7, markeredgewidth=0.0, label="Periodic", zorder=2)
axs[0].legend(markerscale=8)
axs[1].spy(A_scipy, color="r", markersize=1.7, markeredgewidth=0.0, label="Neumann")
axs[1].tick_params(axis='both', which='major', labelsize=22)
axs[1].spy(A_mpc_scipy, color="b", markersize=1.7, markeredgewidth=0.0, label="Periodic")
axs[1].legend(markerscale=8)
if MPI.COMM_WORLD.size > 1:
    plt.savefig("sp_periodic_rank{0:d}.png".format(MPI.COMM_WORLD.rank), dpi=200)
else:
    plt.savefig("sp_periodic.png", dpi=600)

L_org = dolfinx.fem.assemble_vector(rhs)
dolfinx.fem.apply_lifting(L_org, [a], [bcs])
L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                  mode=PETSc.ScatterMode.REVERSE)
dolfinx.fem.set_bc(L_org, bcs)
solver.setOperators(A_org)
u_ = dolfinx.Function(V)
solver.solve(L_org, u_.vector)

it = solver.getIterationNumber()
print("Unconstrained solver iterations {0:d}".format(it))
u_.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                      mode=PETSc.ScatterMode.FORWARD)
u_.name = "u_unconstrained"
outfile.write_function(u_)


# Create global transformation matrix
K = dolfinx_mpc.utils.create_transformation_matrix(V, mpc)

# Create reduced RHS and LHS
A_global = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A_org)
reduced_A = np.matmul(np.matmul(K.T, A_global), K)
vec = dolfinx_mpc.utils.PETScVector_to_global_numpy(L_org)
reduced_L = np.dot(K.T, vec)
# Solve linear system
d = np.linalg.solve(reduced_A, reduced_L)
uh_numpy = np.dot(K, d)

# Transfer data from the MPC problem to numpy arrays for comparison
A_np = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A)
b_np = dolfinx_mpc.utils.PETScVector_to_global_numpy(b)

dolfinx_mpc.utils.compare_vectors(reduced_L, b_np, mpc)
dolfinx_mpc.utils.compare_matrices(reduced_A, A_np, mpc)
assert np.allclose(uh.array, uh_numpy[uh.owner_range[0]:uh.owner_range[1]])

dolfinx.common.list_timings(MPI.COMM_WORLD, [dolfinx.common.TimingType.wall])
