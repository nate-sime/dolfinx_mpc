import numba
import dolfinx
import ufl
import types
import typing
import cffi
import numpy
from dolfinx_mpc import cpp

@numba.njit
def backsubstitution(b, mpc):
    """
    Insert mpc values into vector bc
    """
    (slaves, masters, coefficients, offsets) = mpc
    for i, slave in enumerate(slaves):
        masters_i = masters[offsets[i]:offsets[i+1]]
        coeffs_i = coefficients[offsets[i]:offsets[i+1]]
        for j, (master, coeff) in enumerate(zip(masters_i, coeffs_i)):
            if slave == master:
                print("No slaves (since slave is same as master dof)")
                continue

            b[slave] += coeff*b[master]


class MPCFunctionSpace(dolfinx.FunctionSpace):

    def __init__(self,
                 mesh: dolfinx.cpp.mesh.Mesh,
                 element: typing.Union[ufl.FiniteElementBase, dolfinx.function.ElementMetaData],
                 markers: typing.Dict[types.FunctionType, types.FunctionType]):


        # Initialise the ufl.FunctionSpace
        if isinstance(element, ufl.FiniteElementBase):
            super().__init__(mesh.ufl_domain(), element)
        else:
            e = dolfinx.function.ElementMetaData(*element)
            ufl_element = ufl.FiniteElement(e.family, mesh.ufl_cell(), e.degree, form_degree=e.form_degree)
            ufl.FunctionSpace.__init__(self,mesh.ufl_domain(), ufl_element)

        # Compile dofmap and element and create DOLFIN objects
        ufc_element, ufc_dofmap_ptr = dolfinx.jit.ffcx_jit(
            self.ufl_element(), form_compiler_parameters=None, mpi_comm=mesh.mpi_comm())

        ffi = cffi.FFI()
        ufc_element = dolfinx.fem.dofmap.make_ufc_finite_element(ffi.cast("uintptr_t", ufc_element))
        cpp_element = dolfinx.cpp.fem.FiniteElement(ufc_element)

        ufc_dofmap = dolfinx.fem.dofmap.make_ufc_dofmap(ffi.cast("uintptr_t", ufc_dofmap_ptr))
        cpp_dofmap = dolfinx.cpp.fem.create_dofmap(ufc_dofmap, mesh)
        self._cpp_object = dolfinx.cpp.function.FunctionSpace(mesh, cpp_element, cpp_dofmap)

        # Get coordinate of dofs to determine whereabouts of masters and slaves through the markers functions
        slaves = numpy.array([],dtype = numpy.int32)
        masters = numpy.array([],dtype=numpy.int32)
        coefficients = numpy.array([])
        offsets = numpy.array([],dtype=numpy.int32)

        # Determine master and slave dofs as a 1D array structure
        X = self.tabulate_dof_coordinates()
        for slave_function in markers.keys():
            offsets = numpy.append(offsets, len(masters))
            slaves = numpy.append(slaves, slave_function(X))
            master_dofs, values = markers[slave_function](X)
            masters = numpy.append(masters, master_dofs)
            coefficients = numpy.append(coefficients, values)
        offsets = numpy.append(offsets, len(masters))

        # Generate MultiPointConstraint function that is saved internally
        self.original_function_space = dolfinx.cpp.function.FunctionSpace(mesh, cpp_element, cpp_dofmap)
        self.mpc = cpp.mpc.MultiPointConstraint(self.original_function_space, list(slaves), list(masters), list(coefficients), list(offsets))
        # Need to extract all data from previous dofmap here, and append ghost data from dolfinx_mpc
        new_index_map = self.mpc.generate_index_map()

        modified_dofmap = dolfinx.cpp.fem.DofMap(cpp_dofmap.dof_layout, new_index_map, cpp_dofmap.dof_array())
        self._cpp_object = dolfinx.cpp.function.FunctionSpace(mesh, cpp_element, modified_dofmap)
