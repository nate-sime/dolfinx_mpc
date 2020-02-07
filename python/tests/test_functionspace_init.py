import dolfinx_mpc
import dolfinx
import numpy as np

def master_dofs(x):
    logical = np.logical_and(np.logical_or(np.isclose(x[:,0],0),np.isclose(x[:,0],1)),np.isclose(x[:,1],1))
    try:
        return np.argwhere(logical).T[0], np.ones(np.argwhere(logical).T[0].shape)
    except IndexError:
        return [],[]

def slave_dofs(x):
    logical = np.logical_and(np.isclose(x[:,0],1),np.isclose(x[:,1],0))
    try:
        return np.argwhere(logical)[0]
    except IndexError:
        return []

def test_mpc_assembly():
    # Create mesh and function space
    mesh = dolfinx.UnitSquareMesh(dolfinx.MPI.comm_world, 5,3)
    V = dolfinx_mpc.MPCFunctionSpace(mesh, ("Lagrange", 1),markers={slave_dofs:master_dofs})


if __name__=="__main__":
    test_mpc_assembly()
