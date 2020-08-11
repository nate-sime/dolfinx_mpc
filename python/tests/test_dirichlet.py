# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import dolfinx_mpc
import dolfinx_mpc.utils
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.log
import pytest


@pytest.mark.parametrize("alpha", [0.5, 1, 2])
def test_dirichlet(alpha):
    # Create mesh and function space
    mesh = dolfinx.UnitSquareMesh(
        MPI.COMM_WORLD, 1, 1, dolfinx.cpp.mesh.CellType.quadrilateral)

    V = dolfinx.FunctionSpace(mesh, ("Lagrange", 1))

    def l2b(li):
        return np.array(li, dtype=np.float64).tobytes()

    s_m_c = {l2b([1, 0]): {
        l2b([0, 0]): alpha}}

    mpc = dolfinx_mpc.MultiPointConstraint(V)
    mpc.create_general_constraint(s_m_c)
    mpc.finalize()
    V = mpc.function_space()

    # Test against generated code and general assembler
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    f = x[0]+2*x[1]
    a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
    rhs = ufl.inner(f, v)*ufl.dx

    # Fix bottom corner
    u_bc = dolfinx.function.Function(V)
    u_b_val = 2
    with u_bc.vector.localForm() as u_local:
        u_local.set(u_b_val)

    def bottom_corner(x):
        return np.isclose(x, [[0], [0], [0]]).all(axis=0)

    bottom_dofs = dolfinx.fem.locate_dofs_geometrical(V, bottom_corner)
    bc_bottom = dolfinx.fem.DirichletBC(u_bc, bottom_dofs)

    bcs = [bc_bottom]

    A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs)
    b = dolfinx_mpc.assemble_vector(rhs, mpc)
    dolfinx_mpc.apply_lifting(b, [a], mpc, [bcs])
    # dolfinx_mpc.apply_lifting(b, [a], mpc, [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                  mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b, bcs)

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

    # Reference problem
    def slave_corner(x):
        return np.isclose(x, [[1], [0], [0]]).all(axis=0)

    slave_dofs = dolfinx.fem.locate_dofs_geometrical(V, slave_corner)
    u_slave = dolfinx.function.Function(V)
    with u_slave.vector.localForm() as u_local:
        u_local.set(u_b_val*alpha)
    bc_slave = dolfinx.fem.DirichletBC(u_slave, slave_dofs)

    bcs = [bc_bottom, bc_slave]
    A_ref = dolfinx.fem.assemble_matrix(a, bcs=bcs)
    A_ref.assemble()
    b_ref = dolfinx.fem.assemble_vector(rhs)
    dolfinx.fem.apply_lifting(b_ref, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                  mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b_ref, bcs)
    solver.setOperators(A_ref)
    u_ref = b_ref.copy()
    u_ref.set(0)
    solver.solve(b_ref, u_ref)
    u_ref.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                      mode=PETSc.ScatterMode.FORWARD)

    assert(np.allclose(u_ref.array, uh.array))
