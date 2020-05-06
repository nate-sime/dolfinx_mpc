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


def donut(res=0.5, order=2):
    geom = pygmsh.built_in.Geometry()
    inner_circle = geom.add_circle([0, 0, 0], 1, lcar=res)
    outer_circle = geom.add_circle(
        [0, 0, 0], 2, lcar=res, holes=[inner_circle])
    if order > 1:
        geom.add_raw_code("Mesh.ElementOrder={0:d};".format(order))
    geom.add_physical([outer_circle.plane_surface], 1)
    geom.add_physical(outer_circle.line_loop.lines, 2)
    geom.add_physical(inner_circle.line_loop.lines, 3)

    mesh = pygmsh.generate_mesh(geom, dim=2, prune_z_0=True, verbose=False)

    ct = "triangle" if order == 1 else "triangle6"
    cells = np.vstack(np.array([cells.data for cells in mesh.cells
                                if cells.type == ct]))
    ft = "line" if order == 1 else "line3"
    facet_cells = np.vstack(np.array([cells.data for cells in mesh.cells
                                      if cells.type == ft]))

    facet_data = mesh.cell_data_dict["gmsh:physical"][ft]
    cell_data = mesh.cell_data_dict["gmsh:physical"][ct]
    triangle_mesh = meshio.Mesh(points=mesh.points,
                                cells=[(ct, cells)],
                                cell_data={"name_to_read": [cell_data]})

    facet_mesh = meshio.Mesh(points=mesh.points,
                             cells=[(ft, facet_cells)],
                             cell_data={"name_to_read": [facet_data]})
    # Write mesh
    meshio.xdmf.write("meshes/donut.xdmf", triangle_mesh)
    meshio.xdmf.write("meshes/facet_donut.xdmf", facet_mesh)


def stokes(res, order=2):
    # Create mesh
    if MPI.COMM_WORLD.size == 1:
        donut(res, order)
    # Load mesh and corresponding facet markers
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                             "meshes/donut.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh("Grid")
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)
    if order == 1:
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD,
                                 "meshes/facet_donut.xdmf", "r") as xdmf:
            mt = xdmf.read_meshtags(mesh, "Grid")
    else:
        def inner_circle(x):
            return np.sqrt(x[0]**2+x[1]**2) < 1.1

        def outer_circle(x):
            return np.sqrt(x[0]**2+x[1]**2) > 1.9
        inner_facets = dolfinx.mesh.locate_entities_geometrical(
            mesh, fdim, inner_circle, boundary_only=True)
        values = 3*np.ones(len(inner_facets), dtype=np.int32)
        outer_facets = dolfinx.mesh.locate_entities_geometrical(
            mesh, fdim, outer_circle, boundary_only=True)
        indices = np.concatenate((inner_facets, outer_facets))
        values = np.concatenate(
            (values, 2*np.ones(len(outer_facets), dtype=np.int32)))
        mt = dolfinx.MeshTags(mesh, fdim, indices, values)
    print(order)
    if order == 1:
        outfile = dolfinx.io.XDMFFile(
            MPI.COMM_WORLD, "results/demo_stokes_donut.xdmf", "w")
        outfile.write_mesh(mesh)
    # Create the function space
    P2 = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    TH = P2 * P1
    W = dolfinx.FunctionSpace(mesh, TH)
    V = dolfinx.FunctionSpace(mesh, P2)

    def u_ex_expr(x):
        values = np.zeros((2, x.shape[1]))
        values[0] = -x[1]*np.sqrt(x[0]**2+x[1]**2)
        values[1] = x[0]*np.sqrt(x[0]**2+x[1]**2)
        return values
    bcs = []
    # Corners as you in theory would have slip in both direction,
    # the velocity has to be zero there
    u_bc = dolfinx.Function(V)
    u_bc.interpolate(u_ex_expr)
    facets = mt.indices[mt.values == 3]
    dofs = dolfinx.fem.locate_dofs_topological((W.sub(0), V), fdim,
                                               facets)

    bc_vel = dolfinx.DirichletBC(u_bc, dofs, W.sub(0))
    bcs.append(bc_vel)

    # Since for this problem the pressure is only determined up to a constant,
    # we pin the pressure at one of the internal nodes
    Q = dolfinx.FunctionSpace(mesh, P1)

    zero = dolfinx.Function(Q)
    with zero.vector.localForm() as zero_local:
        zero_local.set(0.0)
    dofs = dolfinx.fem.locate_dofs_topological((W.sub(1), Q), fdim,
                                               facets)

    bc_p = dolfinx.DirichletBC(zero, dofs[0], W.sub(1))
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
     coeffs, offsets) = set_master_slave_slip_relationship(W, V, mt, [2],
                                                           bcs)
    # masters = np.array([])
    # slaves = np.array([])
    # coeffs = np.array([])
    # offsets = np.array([0])
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
    u_ex = ufl.as_vector((-y*ufl.sqrt(x**2+y**2), x*ufl.sqrt(x**2+y**2)))
    f = -ufl.div(2*mu*ufl.sym(ufl.grad(u_ex)))
    p0 = dolfinx.Function(W.sub(1).collapse())
    n = dolfinx.FacetNormal(mesh)
    g_tau = tangential_proj(T(u_ex, p0)*n, n)

    a = (2*mu*ufl.inner(sym_grad(u), sym_grad(v))
         - ufl.inner(p, ufl.div(v))
         - ufl.inner(ufl.div(u), q)) * ufl.dx

    L = ufl.inner(f, v) * ufl.dx + ufl.inner(g_tau, v) * \
        ufl.ds(domain=mesh, subdomain_data=mt, subdomain_id=2)

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
    if order == 1:
        outfile.write_function(u)
        outfile.write_function(p)
        outfile.close()
    else:
        outfile = dolfinx.io.VTKFile("results/demo_stokes_donut_u.pvd")
        outfile.write(u)
        outfile = dolfinx.io.VTKFile("results/demo_stokes_donut_p.pvd")
        outfile.write(p)
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

    u_L2 = np.sqrt(dolfinx.fem.assemble_scalar(
        ufl.inner(u-u_ex, u-u_ex)*ufl.dx))
    u_H1 = np.sqrt(dolfinx.fem.assemble_scalar(
        (ufl.inner(u-u_ex, u-u_ex) +
         ufl.inner(ufl.grad(u-u_ex), ufl.grad(u-u_ex)))*ufl.dx))
    p_L2 = np.sqrt(dolfinx.fem.assemble_scalar(
        ufl.inner(p-p0, p-p0)*ufl.dx))
    print("Velocity L2 error:", u_L2)
    print("Pressure L2 error:", p_L2)
    print("Velocity H1 error:", u_H1)
    print("Flux", dolfinx.fem.assemble_scalar(
        ufl.inner(u, n) *
        ufl.ds(domain=mesh, subdomain_data=mt, subdomain_id=2)))

    return u_L2, p_L2, u_H1


error_u = []
error_p = []
error_h1 = []
h = [1*0.5**i for i in range(5)]
for res in h:
    u_L2, p_L2, u_h1 = stokes(res, 2)
    error_u.append(u_L2)
    error_p.append(p_L2)
    error_h1.append(u_h1)
error_u = np.array(error_u)
error_p = np.array(error_p)
error_h1 = np.array(error_h1)
h = np.array(h)
print("U rate", np.log(error_u[1:]/error_u[:-1])/np.log(h[1:]/h[:-1]))
print("P rate", np.log(error_p[1:]/error_p
                       [:-1])/np.log(h[1:]/h[:-1]))
print("U H1 rate", np.log(error_h1[1:]/error_h1[:-1])/np.log(h[1:]/h[:-1]))
