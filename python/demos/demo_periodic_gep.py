# Copyright (C) 2021 fmonteghetti and Jørgen S. DOkken
#
# This file is part of DOLFINX_MPC
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
# This demo, adapted from 'demo_periodic.py', solves
#
#     - div grad u(x, y) = lambda*u(x, y)
#
# on the unit square with homogeneous Dirichlet boundary conditions
# at y = 0, 1 and periodic boundary conditions at x = 0, 1.
# The weak form reads
#
#       (grad(u),grad(v)) = lambda * (u,v),
#
# which leads to the generalized eigenvalue problem
#
#       A * U = lambda * B * U,
#
# where A and B are real symmetric positive definite matrices. The generalized
# eigenvalue problem is solved using SLEPc and the computed eigenvalues are
# compared to the exact ones.

import dolfinx
import dolfinx.io
import dolfinx_mpc
import dolfinx_mpc.utils
import ufl
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc
from typing import Tuple, List


def print0(string: str):
    """Print on rank 0 only"""
    if MPI.COMM_WORLD.rank == 0:
        print(string)


def monitor_EPS_short(EPS: SLEPc.EPS, it: int, nconv: int, eig: list, err: list, it_skip: int):
    """
    Concise monitor for EPS.solve().

    Parameters
    ----------
    eps
        Eigenvalue Problem Solver class.
    it
       Current iteration number.
    nconv
       Number of converged eigenvalue.
    eig
       Eigenvalues
    err
       Computed errors.
    it_skip
        Iteration skip.

    """
    if (it == 1):
        print0('******************************')
        print0('***  SLEPc Iterations...   ***')
        print0('******************************')
        print0("Iter. | Conv. | Max. error")
        print0(f"{it:5d} | {nconv:5d} | {max(err):1.1e}")
    elif not it % it_skip:
        print0(f"{it:5d} | {nconv:5d} | {max(err):1.1e}")


def EPS_print_results(EPS: SLEPc.EPS):
    """ Print summary of solution results. """
    print0("\n******************************")
    print0("*** SLEPc Solution Results ***")
    print0("******************************")
    its = EPS.getIterationNumber()
    print0(f"Iteration number: {its}")
    nconv = EPS.getConverged()
    print0(f"Converged eigenpairs: {nconv}")

    if nconv > 0:
        # Create the results vectors
        vr, vi = EPS.getOperators()[0].createVecs()
        print0("\nConverged eigval.  Error ")
        print0("----------------- -------")
        for i in range(nconv):
            k = EPS.getEigenpair(i, vr, vi)
            error = EPS.computeError(i)
            if k.imag != 0.0:
                print0(f" {k.real:2.2e} + {k.imag:2.2e}j {error:1.1e}")
            else:
                pad = " " * 11
                print0(f" {k.real:2.2e} {pad} {error:1.1e}")


def EPS_get_spectrum(EPS: SLEPc.EPS, mpc: dolfinx_mpc.MultiPointConstraint) -> Tuple[List[complex], List[PETSc.Vec], List[PETSc.Vec]]:
    """ Retrieve eigenvalues and eigenfunctions from SLEPc EPS object.
    Parameters
    ----------
    EPS
       The SLEPc solver
    mpc
       The multipoint constraint

    Returns
    -------
        Tuple consisting of: List of complex converted eigenvalues,
         lists of converted eigenvectors (real part) and (imaginary part)
    """
    # Get results in lists
    eigval = [EPS.getEigenvalue(i) for i in range(EPS.getConverged())]
    eigvec_r = list()
    eigvec_i = list()
    V = mpc.function_space()
    vr = dolfinx.Function(V).vector
    vi = vr.copy()
    for i in range(EPS.getConverged()):
        EPS.getEigenvector(i, vr, vi)
        eigvec_r.append(vr.copy())
        eigvec_i.append(vi.copy())
    # Sort by increasing real parts
    idx = np.argsort(np.real(np.array(eigval)), axis=0)
    eigval = [eigval[i] for i in idx]
    eigvec_r = [eigvec_r[i] for i in idx]
    eigvec_i = [eigvec_i[i] for i in idx]
    return (eigval, eigvec_r, eigvec_i)


def solve_GEP_shiftinvert(A: PETSc.Mat, B: PETSc.Mat,
                          problem_type: SLEPc.EPS.ProblemType = SLEPc.EPS.ProblemType.GNHEP,
                          solver: SLEPc.EPS.Type = SLEPc.EPS.Type.KRYLOVSCHUR,
                          nev: int = 10, tol: float = 1e-6, max_it: int = 10,
                          target: float = 0.0, shift: float = 0.0) -> SLEPc.EPS:
    """
    Solve generalized eigenvalue problem A*x=lambda*B*x using shift-and-invert
    as spectral transform method.

    Parameters
    ----------
    A
       The matrix A
    B
       The matrix B
    problem_type
       The problem type, for options see
       https://slepc.upv.es/slepc4py-current/docs/apiref/slepc4py.SLEPc.EPS.ProblemType-class.html 
    solver:
       Solver type, for options see
       https://slepc.upv.es/slepc4py-current/docs/apiref/slepc4py.SLEPc.EPS.Type-class.html
    nev
        Number of requested eigenvalues.
    tol
       Tolerance for slepc solver
    max_it
       Maximum number of iterations.
    target
       Target eigenvalue. Also used for sorting.
    shift
       Shift 'sigma' used in shift-and-invert.

    Returns
    -------
    EPS
       The SLEPc solver
    """

    # Build an Eigenvalue Problem Solver object
    EPS = SLEPc.EPS()
    EPS.create(comm=MPI.COMM_WORLD)
    EPS.setOperators(A, B)
    EPS.setProblemType(problem_type)
    # set the number of eigenvalues requested
    EPS.setDimensions(nev=nev)
    # Set solver
    EPS.setType(solver)
    # set eigenvalues of interest
    EPS.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    EPS.setTarget(target)  # sorting
    # set tolerance and max iterations
    EPS.setTolerances(tol=tol, max_it=max_it)
    # Set up shift-and-invert
    # Only work if 'whichEigenpairs' is 'TARGET_XX'
    ST = EPS.getST()
    ST.setType(SLEPc.ST.Type.SINVERT)
    ST.setShift(shift)
    EPS.setST(ST)
    # set monitor
    it_skip = 1
    EPS.setMonitor(lambda eps, it, nconv, eig, err:
                   monitor_EPS_short(eps, it, nconv, eig, err, it_skip))
    # parse command line options
    EPS.setFromOptions()
    # Display all options (including those of ST object)
    # EPS.view()
    EPS.solve()
    EPS_print_results(EPS)
    return EPS


# Create mesh and finite element
N = 50
mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, N, N)
V = dolfinx.FunctionSpace(mesh, ("CG", 1))

# Create Dirichlet boundary condition
u_bc = dolfinx.Function(V)
with u_bc.vector.localForm() as u_local:
    u_local.set(0.0)


def DirichletBoundary(x):
    return np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1))


facets = dolfinx.mesh.locate_entities_boundary(mesh, 1,
                                               DirichletBoundary)
topological_dofs = dolfinx.fem.locate_dofs_topological(V, 1, facets)
bc = dolfinx.fem.DirichletBC(u_bc, topological_dofs)
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


mpc = dolfinx_mpc.MultiPointConstraint(V)
mpc.create_periodic_constraint(mt, 2, periodic_relation, bcs)
mpc.finalize()

# Define variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
b = ufl.inner(u, v) * ufl.dx

# Diagonal values for slave and Dirichlet DoF
# The generalized eigenvalue problem will have spurious eigenvalues at
# lambda_spurious = diagval_A/diagval_B. Here we choose lambda_spurious=1e4,
# which is far from the region of interest.
diagval_A = 1e2
diagval_B = 1e-2

A = dolfinx_mpc.assemble_matrix(a, mpc, bcs=bcs, diagval=diagval_A)
B = dolfinx_mpc.assemble_matrix(b, mpc, bcs=bcs, diagval=diagval_B)

Nev = 10  # number of requested eigenvalues
EPS = solve_GEP_shiftinvert(A, B,
                            problem_type=SLEPc.EPS.ProblemType.GHEP,
                            solver=SLEPc.EPS.Type.KRYLOVSCHUR,
                            nev=Nev, tol=1e-6, max_it=10,
                            target=1.5, shift=1.5)
(eigval, eigvec_r, eigvec_i) = EPS_get_spectrum(EPS, mpc)

# update slave DoF
for i in range(len(eigval)):
    mpc.backsubstitution(eigvec_r[i])
    mpc.backsubstitution(eigvec_i[i])
print0(f"Computed eigenvalues:\n {np.around(eigval,decimals=2)}")

# Exact eigenvalues
l_exact = list(set([(i * np.pi)**2 + (2 * j * np.pi)**2 for i in range(Nev) for j in range(Nev)]))
l_exact.remove(0)
l_exact.sort()
print0(f"Exact eigenvalues:\n {np.around(l_exact[0:Nev-1],decimals=2)}")
