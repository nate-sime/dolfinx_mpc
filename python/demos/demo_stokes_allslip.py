import dolfinx.cpp
import dolfinx.io
import dolfinx_mpc
import dolfinx_mpc.utils
import meshio
import numpy as np
import pygmsh
import time
import ufl
from petsc4py import PETSc
from mpi4py import MPI

# Length, width and rotation of channel
L = 1
H = 1
theta = np.pi/3
r_matrix = pygmsh.helpers.rotation_matrix([0, 0, 1], theta)


def create_mesh_gmsh(res):
    """
    Create a channel of length 2, height one, rotated pi/6 degrees
    around origin, and corresponding facet markers:
    Walls: 1
    Outlet: 2
    Inlet: 3
    """
    geom = pygmsh.built_in.Geometry()
    rect = geom.add_rectangle(-L, L, -H, H, 0.0, lcar=res)
    geom.rotate(rect, [0, 0, 0], theta, [0, 0, 1])

    geom.add_physical([rect.line_loop.lines[0], rect.line_loop.lines[2]], 1)
    geom.add_physical([rect.line_loop.lines[1]], 2)
    geom.add_physical([rect.line_loop.lines[3]], 3)
    geom.add_physical([rect.surface], 4)

    # Generate mesh
    mesh = pygmsh.generate_mesh(geom, dim=2, prune_z_0=True)
    cells = np.vstack(np.array([cells.data for cells in mesh.cells
                                if cells.type == "triangle"]))

    facet_cells = np.vstack(np.array([cells.data for cells in mesh.cells
                                      if cells.type == "line"]))

    facet_data = mesh.cell_data_dict["gmsh:physical"]["line"]
    cell_data = mesh.cell_data_dict["gmsh:physical"]["triangle"]
    triangle_mesh = meshio.Mesh(points=mesh.points,
                                cells=[("triangle", cells)],
                                cell_data={"name_to_read": [cell_data]})

    facet_mesh = meshio.Mesh(points=mesh.points,
                             cells=[("line", facet_cells)],
                             cell_data={"name_to_read": [facet_data]})
    # Write mesh
    meshio.xdmf.write("meshes/mesh.xdmf", triangle_mesh)
    meshio.xdmf.write("meshes/facet_mesh.xdmf", facet_mesh)


def stokes(res):
    # Create mesh
    if MPI.COMM_WORLD.size == 1:
        create_mesh_gmsh(res)
    # Load mesh and corresponding facet markers
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                             "meshes/mesh.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh("Grid")

    mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                             "meshes/facet_mesh.xdmf", "r") as xdmf:
        mt = xdmf.read_meshtags(mesh, "Grid")

    outfile = dolfinx.io.XDMFFile(
        MPI.COMM_WORLD, "results/demo_stokes_allslip.xdmf", "w")
    outfile.write_mesh(mesh)
    # Create the function space
    P2 = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    TH = P2 * P1
    W = dolfinx.FunctionSpace(mesh, TH)
    V = dolfinx.FunctionSpace(mesh, P2)

    # Corners as you in theory would have slip in both direction,
    # the velocity has to be zero there
    zero = dolfinx.Function(V)
    with zero.vector.localForm() as zero_local:
        zero_local.set(0.0)
    bcs = []
    corners = np.array([[-1, -1, 0], [-1, 1, 0], [1, -1, 0], [1, 1, 0]])
    for corner in corners:
        rot_point = np.dot(r_matrix, corner)

        def is_corner(x):
            return np.isclose(x.T, rot_point).all(axis=1)

        dofs = dolfinx.fem.locate_dofs_geometrical((W.sub(0), V),
                                                   is_corner)

        bc_vel = dolfinx.DirichletBC(zero, dofs, W.sub(0))
        bcs.append(bc_vel)

    # Since for this problem the pressure is only determined up to a constant,
    # we pin the pressure at a corner
    Q = dolfinx.FunctionSpace(mesh, P1)
    bottom_corner = np.dot(r_matrix, [-1, -1, 0])
    zero = dolfinx.Function(Q)
    with zero.vector.localForm() as zero_local:
        zero_local.set(0.0)
    dofs = dolfinx.fem.locate_dofs_geometrical((W.sub(1), Q),
                                               lambda x: np.isclose(
        x.T, bottom_corner)
        .all(axis=1))
    bc_p = dolfinx.DirichletBC(zero, dofs, W.sub(1))
    bcs.append(bc_p)

    def set_master_slave_slip_relationship(W, V, mt, values, bcs=None):
        """
        Set a slip condition for all dofs in W (where V is the collapsed
        0th subspace of W) that corresponds to the facets in mt
        marked with value
        """
        x = W.tabulate_dof_coordinates()
        global_indices = W.dofmap.index_map.global_indices(False)
        slaves = []
        masters = []
        coeffs = []

        for value in values:
            wall_facets = mt.indices[np.flatnonzero(mt.values == value)]
            if bcs is not None:
                bc_dofs = []
                for bc in bcs:
                    bc_g = [global_indices[bdof]
                            for bdof in bc.dof_indices[:, 0]]
                    bc_dofs.append(np.hstack(MPI.COMM_WORLD.allgather(bc_g)))
                bc_dofs = np.hstack(bc_dofs)
            else:
                bc_dofs = np.array([])
            Vx = V.sub(0).collapse()
            Vy = V.sub(1).collapse()
            dofx = dolfinx.fem.locate_dofs_topological((W.sub(0).sub(0),
                                                        Vx),
                                                       1, wall_facets)
            dofy = dolfinx.fem.locate_dofs_topological((W.sub(0).sub(1),
                                                        Vy),
                                                       1, wall_facets)

            nh = dolfinx_mpc.facet_normal_approximation(V, mt, value)
            nhx, nhy = nh.sub(0).collapse(), nh.sub(1).collapse()
            nh.name = "n"
            # outfile.write_function(nh)

            nx = nhx.vector.getArray()
            ny = nhy.vector.getArray()

            # Find index of each pair of x and y components.
            for d_x in dofx:
                # Skip if dof is a ghost
                if d_x[1] > Vx.dofmap.index_map.size_local:
                    continue
                for d_y in dofy:
                    # Skip if dof is a ghost
                    if d_y[1] > Vy.dofmap.index_map.size_local:
                        continue
                    # Skip if not at same physical coordinate
                    if not np.allclose(x[d_x[0]], x[d_y[0]]):
                        continue

                    slave_dof = global_indices[d_x[0]]
                    master_dof = global_indices[d_y[0]]
                    master_not_bc = master_dof not in bc_dofs
                    new_slave = slave_dof not in slaves
                    new_master = master_dof not in masters
                    if master_not_bc and new_slave and new_master:
                        slaves.append(slave_dof)
                        masters.append(master_dof)
                        local_coeff = - ny[d_y[1]]/nx[d_x[1]]
                        coeffs.append(local_coeff)
        # As all dofs is in the same block, we do not need to communicate
        # all master and slave nodes have been found
        global_slaves = np.hstack(MPI.COMM_WORLD.allgather(slaves))
        global_masters = np.hstack(MPI.COMM_WORLD.allgather(masters))
        global_coeffs = np.hstack(MPI.COMM_WORLD.allgather(coeffs))
        offsets = np.arange(len(global_slaves)+1)

        return (np.array(global_masters), np.array(global_slaves),
                np.array(global_coeffs), offsets)

    start = time.time()
    (masters, slaves,
     coeffs, offsets) = set_master_slave_slip_relationship(W, V, mt, [1, 2, 3],
                                                           bcs)

    end = time.time()
    print("Setup master slave relationship: {0:.2e}".format(end-start))

    mpc = dolfinx_mpc.cpp.mpc.MultiPointConstraint(W._cpp_object, slaves,
                                                   masters, coeffs, offsets)
    # Define variational problem
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    mu = dolfinx.Constant(mesh, 1)

    def tangential_proj(u, n):
        """
        See for instance:
        https://link.springer.com/content/pdf/10.1023/A:1022235512626.pdf
        """
        return (ufl.Identity(u.ufl_shape[0]) - ufl.outer(n, n)) * u

    def sym_grad(u):
        return ufl.sym(ufl.grad(u))

    def T(u, p):
        return 2 * mu * sym_grad(u) - p * ufl.Identity(u.ufl_shape[0])

    x, y = ufl.SpatialCoordinate(mesh)
    r_inverse = pygmsh.helpers.rotation_matrix([0, 0, 1], -theta)
    x_unrot = (r_inverse[0, 0]*x + r_inverse[0, 1]*y)
    y_unrot = (r_inverse[1, 0]*x + r_inverse[1, 1]*y)
    u_ex = ufl.as_matrix(r_matrix[:-1, :-1])*ufl.as_vector(
        (2*y_unrot*(1-x_unrot**2), -2*x_unrot*(1-y_unrot**2)))
    f = -ufl.div(2*mu*ufl.sym(ufl.grad(u_ex)))
    p0 = dolfinx.Function(W.sub(1).collapse())
    n = dolfinx.FacetNormal(mesh)
    g_tau = tangential_proj(T(u_ex, p0)*n, n)

    a = (2*mu*ufl.inner(sym_grad(u), sym_grad(v))
         - ufl.inner(p, ufl.div(v))
         - ufl.inner(ufl.div(u), q)) * ufl.dx

    L = ufl.inner(f, v) * ufl.dx + ufl.inner(g_tau, v)*ufl.ds

    # Debug for rotating the velocity field
    # u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    # a = ufl.inner(u, v)*ufl.dx
    # L = ufl.inner(u_ex, v)*ufl.dx
    # uh = dolfinx.Function(V)
    # dolfinx.solve(a == L, uh)
    # outfile.write_function(uh)
    # exit(1)

    # Assemble LHS matrix and RHS vector
    start = time.time()
    A = dolfinx_mpc.assemble_matrix(a, mpc, bcs)

    end = time.time()
    print("Matrix assembly time: {0:.2e} ".format(end-start))
    A.assemble()
    start = time.time()
    b = dolfinx_mpc.assemble_vector(L, mpc)
    end = time.time()
    print("Vector assembly time: {0:.2e} ".format(end-start))

    dolfinx.fem.assemble.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Set Dirichlet boundary condition values in the RHS
    dolfinx.fem.assemble.set_bc(b, bcs)

    # Create and configure solver
    ksp = PETSc.KSP().create(mesh.mpi_comm())
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")

    # Compute the solution
    uh = b.copy()
    start = time.time()
    ksp.solve(b, uh)
    end = time.time()
    print("Solve time {0:.2e}".format(end-start))

    # Back substitute to slave dofs
    dolfinx_mpc.backsubstitution(mpc, uh, W.dofmap)

    Wmpc_cpp = dolfinx.cpp.function.FunctionSpace(mesh, W.element,
                                                  mpc.mpc_dofmap())
    Wmpc = dolfinx.FunctionSpace(None, W.ufl_element(), Wmpc_cpp)

    # Write solution to file
    U = dolfinx.Function(Wmpc)
    U.vector.setArray(uh.array)

    # Split the mixed solution and collapse
    u = U.sub(0).collapse()
    p = U.sub(1).collapse()
    u.name = "u"
    p.name = "p"
    outfile.write_function(u)
    outfile.write_function(p)
    outfile.close()

    # Transfer data from the MPC problem to numpy arrays for comparison
    # A_mpc_np = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A)
    # mpc_vec_np = dolfinx_mpc.utils.PETScVector_to_global_numpy(b)

    # Solve the MPC problem using a global transformation matrix
    # and numpy solvers to get reference values

    # Generate reference matrices and unconstrained solution
    # print("---Reference timings---")
    # start = time.time()
    # A_org = dolfinx.fem.assemble_matrix(a, bcs)
    # end = time.time()
    # print("Normal matrix assembly {0:.2e}".format(end-start))
    # A_org.assemble()

    # start = time.time()
    # L_org = dolfinx.fem.assemble_vector(L)
    # end = time.time()
    # print("Normal vector assembly {0:.2e}".format(end-start))

    # dolfinx.fem.apply_lifting(L_org, [a], [bcs])
    # L_org.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
    #                   mode=PETSc.ScatterMode.REVERSE)
    # dolfinx.fem.set_bc(L_org, bcs)
    # solver = PETSc.KSP().create(MPI.COMM_WORLD)
    # ksp.setType("preonly")
    # ksp.getPC().setType("lu")
    # ksp.getPC().setFactorSolverType("mumps")
    # solver.setOperators(A_org)

    # # Create global transformation matrix
    # K = dolfinx_mpc.utils.create_transformation_matrix(W.dim(), slaves,
    #                                                    masters, coeffs,
    #                                                    offsets)
    # # Create reduced A
    # A_global = dolfinx_mpc.utils.PETScMatrix_to_global_numpy(A_org)
    # start = time.time()
    # reduced_A = np.matmul(np.matmul(K.T, A_global), K)
    # end = time.time()
    # print("Numpy matrix reduction: {0:.2e}".format(end-start))
    # # Created reduced L
    # vec = dolfinx_mpc.utils.PETScVector_to_global_numpy(L_org)
    # reduced_L = np.dot(K.T, vec)
    # # Solve linear system
    # start = time.time()
    # d = np.linalg.solve(reduced_A, reduced_L)
    # end = time.time()
    # print("Numpy solve {0:.2e}".format(end-start))
    # # Back substitution to full solution vector
    # uh_numpy = np.dot(K, d)

    # Compare LHS, RHS and solution with reference values
    # dolfinx_mpc.utils.compare_matrices(reduced_A, A_mpc_np, slaves)
    # dolfinx_mpc.utils.compare_vectors(reduced_L, mpc_vec_np, slaves)
    # assert np.allclose(uh.array, uh_numpy[uh.owner_range[0]:
    # uh.owner_range[1]])

    def u_ex_lambda(x):
        r_inverse = pygmsh.helpers.rotation_matrix([0, 0, 1], -theta)
        x_unrot = (r_inverse[0, 0]*x[0] + r_inverse[0, 1]*x[1])
        y_unrot = (r_inverse[1, 0]*x[0] + r_inverse[1, 1]*x[1])
        return np.dot(r_matrix[:-1, :-1],
                      np.array([2*y_unrot*(1-x_unrot**2),
                                -2*x_unrot*(1-y_unrot**2)]))

    u_L2 = np.sqrt(dolfinx.fem.assemble_scalar(
        ufl.inner(u-u_ex, u-u_ex)*ufl.dx))
    p_L2 = np.sqrt(dolfinx.fem.assemble_scalar(
        ufl.inner(p-p0, p-p0)*ufl.dx))
    print("Velocity L2 error:", u_L2)
    print("Pressure L2 error:", p_L2)
    return u_L2, p_L2


error_u = []
error_p = []
h = [0.1*0.5**i for i in range(4)]
for res in h:
    u_L2, p_L2 = stokes(res)
    error_u.append(u_L2)
    error_p.append(p_L2)
error_u = np.array(error_u)
error_p = np.array(error_p)
h = np.array(h)
print("U rate", np.log(error_u[1:]/error_u[:-1])/np.log(h[1:]/h[:-1]))
print("P rate", np.log(error_p[1:]/error_p
                       [:-1])/np.log(h[1:]/h[:-1]))
