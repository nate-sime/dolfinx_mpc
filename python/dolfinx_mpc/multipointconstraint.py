import warnings
from typing import Callable, List

import dolfinx
import numpy
from petsc4py import PETSc

import dolfinx_mpc.cpp

from .dictcondition import create_dictionary_constraint
from .periodic_condition import create_periodic_condition_topological, create_periodic_condition_geometrical


class MultiPointConstraint():
    """
    Multi-point constraint class.
    This class will hold data for multi point constraint relation ships,
    including new index maps for local assembly of matrices and vectors.
    """

    def __init__(self, V: dolfinx.fem.FunctionSpace):
        """
        Initial multi point constraint for a given function space.
        """
        self.local_slaves = []
        self.ghost_slaves = []
        self.local_masters = []
        self.ghost_masters = []
        self.local_coeffs = []
        self.ghost_coeffs = []
        self.local_owners = []
        self.ghost_owners = []
        self.offsets = [0]
        self.ghost_offsets = [0]
        self.V = V
        self.finalized = False

    def add_constraint(self, V: dolfinx.FunctionSpace, slaves: tuple([list, list]),
                       masters: tuple([list, list]), coeffs: tuple([list, list]),
                       owners: tuple([list, list]), offsets: tuple([list, list])):
        """
        Add new constraint given by numpy arrays.
        Input:
            V: The function space for the constraint
            slaves: Tuple of arrays. First tuple contains the local index of slaves on this process.
                    The second tuple contains the local index of ghosted slaves on this process.
            masters: Tuple of arrays. First tuple contains the local index of masters on this process.
                    The second tuple contains the local index of ghosted masters on this process.
                    As a single slave can have multiple master indices, they should be sorted such
                    that they follow the offset tuples.
            coeffs: Tuple of arrays. The coefficients for each master.
            owners: Tuple of arrays containing the index for the process each master is owned by.
            offsets: Tuple of arrays indicating the location in the masters array for the i-th slave
                    in the slaves arrays. I.e.
                    local_masters_of_owned_slave[i] = masters[0][offsets[0][i]:offsets[0][i+1]]
                    ghost_masters_of_ghost_slave[i] = masters[1][offsets[1][i]:offsets[1][i+1]]
        """
        assert(V == self.V)
        if self.finalized:
            raise RuntimeError("MultiPointConstraint has already been finalized")
        local_slaves, ghost_slaves = slaves
        local_masters, ghost_masters = masters
        local_coeffs, ghost_coeffs = coeffs
        local_owners, ghost_owners = owners
        local_offsets, ghost_offsets = offsets
        if len(local_slaves) > 0:
            self.offsets.extend(offset + len(self.local_masters)
                                for offset in local_offsets[1:])
            self.local_slaves.extend(local_slaves)
            self.local_masters.extend(local_masters)
            self.local_coeffs.extend(local_coeffs)
            self.local_owners.extend(local_owners)
        if len(ghost_slaves) > 0:
            self.ghost_offsets.extend(offset + len(self.ghost_masters)
                                      for offset in ghost_offsets[1:])
            self.ghost_slaves.extend(ghost_slaves)
            self.ghost_masters.extend(ghost_masters)
            self.ghost_coeffs.extend(ghost_coeffs)
            self.ghost_owners.extend(ghost_owners)

    def add_constraint_from_mpc_data(self, V: dolfinx.FunctionSpace, mpc_data: dolfinx_mpc.cpp.mpc.mpc_data):
        if self.finalized:
            raise RuntimeError("MultiPointConstraint has already been finalized")
        self.add_constraint(V, mpc_data.get_slaves(), mpc_data.get_masters(),
                            mpc_data.get_coeffs(), mpc_data.get_owners(),
                            mpc_data.get_offsets())

    def finalize(self):
        """
        Finalizes a multi point constraint by adding all constraints together,
        locating slave cells and building a new index map with
        additional ghosts
        """
        if self.finalized:
            raise RuntimeError("MultiPointConstraint has already been finalized")
        num_local_slaves = len(self.local_slaves)
        num_local_masters = len(self.local_masters)
        slaves = self.local_slaves
        masters = self.local_masters
        coeffs = self.local_coeffs
        owners = self.local_owners
        offsets = self.offsets
        # Merge local and ghost arrays
        if len(self.ghost_slaves) > 0:
            ghost_offsets = [g_offset + num_local_masters for g_offset in self.ghost_offsets[1:]]
            slaves.extend(self.ghost_slaves)
            masters.extend(self.ghost_masters)
            coeffs.extend(self.ghost_coeffs)
            owners.extend(self.ghost_owners)
            offsets.extend(ghost_offsets)
        # Initialize C++ object and create slave->cell maps
        self._cpp_object = dolfinx_mpc.cpp.mpc.MultiPointConstraint(
            self.V._cpp_object, slaves, num_local_slaves, masters, coeffs, owners, offsets)
        # Add masters and compute new index maps
        # Replace function space
        V_cpp = dolfinx.cpp.fem.FunctionSpace(self.V.mesh, self.V.element, self._cpp_object.dofmap())
        self.V_mpc = dolfinx.FunctionSpace(None, self.V.ufl_element(), V_cpp)
        self.finalized = True
        # Delete variables that are no longer required
        del (self.local_slaves, self.local_masters,
             self.local_coeffs, self.local_owners, self.offsets,
             self.ghost_coeffs, self.ghost_masters, self.ghost_offsets,
             self.ghost_owners, self.ghost_slaves)

    def create_periodic_constraint_topological(self, meshtag: dolfinx.MeshTags, tag: int,
                                               relation: Callable[[numpy.ndarray], numpy.ndarray],
                                               bcs: list([dolfinx.DirichletBC]), scale: PETSc.ScalarType = 1):
        """
        Create periodic condition for all dofs in MeshTag with given marker:
        u(x_i) = scale * u(relation(x_i))
        for all x_i on marked entities.

        Parameters
        ==========
        meshtag
            MeshTag for entity to apply the periodic condition on
        tag
            Tag indicating which entities should be slaves
        relation
            Lambda-function describing the geometrical relation
        bcs
            Dirichlet boundary conditions for the problem
            (Periodic constraints will be ignored for these dofs)
        scale
            Float for scaling bc
        """
        slaves, masters, coeffs, owners, offsets = create_periodic_condition_topological(
            self.V, meshtag, tag, relation, bcs, scale)
        self.add_constraint(self.V, slaves, masters, coeffs, owners, offsets)

    def create_periodic_constraint_geometrical(self, V: dolfinx.FunctionSpace,
                                               indicator: Callable[[numpy.ndarray], numpy.ndarray],
                                               relation: Callable[[numpy.ndarray], numpy.ndarray],
                                               bcs: List[dolfinx.DirichletBC], scale: PETSc.ScalarType = 1):
        """
        Create a periodic condition for all degrees of freedom's satisfying indicator(x):
        u(x_i) = scale * u(relation(x_i)) for all x_i where indicator(x_i) == True

        Parameters
        ==========
        indicator
            Lambda-function to locate degrees of freedom that should be slaves
        relation
            Lambda-function describing the geometrical relation to master dofs
        bcs
            Dirichlet boundary conditions for the problem
            (Periodic constraints will be ignored for these dofs)
        scale
            Float for scaling bc
        """
        slaves, masters, coeffs, owners, offsets = create_periodic_condition_geometrical(
            self.V, indicator, relation, bcs, scale)
        self.add_constraint(self.V, slaves, masters, coeffs, owners, offsets)

    def create_slip_constraint(self, facet_marker: tuple([dolfinx.MeshTags, int]), v: dolfinx.Function,
                               sub_space: dolfinx.FunctionSpace = None, sub_map: numpy.ndarray = numpy.array([]),
                               bcs: list([dolfinx.DirichletBC]) = []):
        """
        Create a slip constraint dot(u, v)=0 over the entities defined in a dolfinx.Meshtags
        marked with index i. normal is the normal vector defined as a vector function.

        Parameters
        ==========
        facet_marker
            Tuple containg the mesh tag and marker used to locate degrees of freedom that should be constrained
        v
            Dolfin function containing the directional vector to dot your slip condition (most commonly a normal vector)
        sub_space
           If the vector v is in a sub space of the multi point function space, supply the sub space
        sub_map
           Map from sub-space to parent space
        bcs
           List of Dirichlet BCs (slip conditions will be ignored on these dofs)

        Example
        =======
        Create constaint dot(u, n)=0 of all indices in mt marked with i
             create_slip_constaint((mt,i), n)

        Create slip constaint for u when in a sub space:
             me = MixedElement(VectorElement("CG", triangle, 2), FiniteElement("CG", triangle 1))
             W = FunctionSpace(mesh, me)
             V, V_to_W = W.sub(0).collapse(True)
             n = Function(V)
             create_slip_constraint((mt, i), normal, V, V_to_W, bcs=[])

        A slip condition cannot be applied on the same degrees of freedom as a Dirichlet BC, and therefore
        any Dirichlet bc for the space of the multi point constraint should be supplied.
             me = MixedElement(VectorElement("CG", triangle, 2), FiniteElement("CG", triangle 1))
             W = FunctionSpace(mesh, me)
             W0 = W.sub(0)
             V, V_to_W = W0.collapse(True)
             n = Function(V)
             bc = dolfinx.DirichletBC(inlet_velocity, dofs, W0)
             create_slip_constraint((mt, i), normal, V, V_to_W, bcs=[bc])
        """
        if sub_space is None:
            W = [self.V._cpp_object]
        else:
            W = [self.V._cpp_object, sub_space._cpp_object]
        mesh_tag, marker = facet_marker
        mpc_data = dolfinx_mpc.cpp.mpc.create_slip_condition(W, mesh_tag, marker, v._cpp_object,
                                                             numpy.asarray(sub_map, dtype=numpy.int32), bcs)
        self.add_constraint_from_mpc_data(self.V, mpc_data=mpc_data)

    def create_general_constraint(self, slave_master_dict, subspace_slave=None, subspace_master=None):
        """
        Parameters
        ==========
        V
            The function space
        slave_master_dict
            Nested dictionary, where the first key is the bit representing the slave dof's coordinate in the mesh. 
            The item of this key is a dictionary, where each key of this dictionary is the bit representation 
            of the master dof's coordinate, and the item the coefficient for the MPC equation.
        subspace_slave
            If using mixed or vector space, and only want to use dofs from a sub space as slave add index here
        subspace_master
            Subspace index for mixed or vector spaces

        Example
        =======
        If the dof D located at [d0,d1] should be constrained to the dofs
        E and F at [e0,e1] and [f0,f1] as
        D = alpha E + beta F
        the dictionary should be:
            {numpy.array([d0, d1], dtype=numpy.float64).tobytes():
                {numpy.array([e0, e1], dtype=numpy.float64).tobytes(): alpha,
                numpy.array([f0, f1], dtype=numpy.float64).tobytes(): beta}}
        """
        slaves, masters, coeffs, owners, offsets = create_dictionary_constraint(
            self.V, slave_master_dict, subspace_slave, subspace_master)
        self.add_constraint(self.V, slaves, masters, coeffs, owners, offsets)

    def slaves(self):
        if not self.finalized:
            raise RuntimeError("MultiPointConstraint has not been finalized")
        return self._cpp_object.slaves()

    def masters_local(self):
        if not self.finalized:
            raise RuntimeError("MultiPointConstraint has not been finalized")
        return self._cpp_object.masters_local()

    def coefficients(self):
        if not self.finalized:
            raise RuntimeError("MultiPointConstraint has not been finalized")
        return self._cpp_object.coefficients()

    def num_local_slaves(self):
        if self.finalized:
            return self._cpp_object.num_local_slaves
        else:
            return len(self.local_slaves)

    def index_map(self):
        if not self.finalized:
            raise RuntimeError("MultiPointConstraint has not been finalized")
        return self._cpp_object.index_map()

    def slave_cells(self):
        if not self.finalized:
            raise RuntimeError("MultiPointConstraint has not been finalized")
        return self._cpp_object.slave_cells()

    def cell_to_slaves(self):
        if not self.finalized:
            raise RuntimeError("MultiPointConstraint has not been finalized")
        return self._cpp_object.cell_to_slaves()

    def create_sparsity_pattern(self, cpp_form: dolfinx.cpp.fem.Form):
        """
        Create sparsity-pattern for MPC given a compiled DOLFINx form
        """
        if not self.finalized:
            raise RuntimeError("MultiPointConstraint has not been finalized")
        return self._cpp_object.create_sparsity_pattern(cpp_form)

    def function_space(self):
        if not self.finalized:
            warnings.warn(
                "Returning original function space for MultiPointConstraint")
            return self.V
        else:
            return self.V_mpc

    def backsubstitution(self, vector: PETSc.Vec):
        """
        For a given vector, empose the multi-point constraint by backsubstiution.
        I.e.
        u[slave] += sum(coeff*u[master] for (coeff, master) in zip(slave.coeffs, slave.masters)
        """
        # Unravel data from constraint
        with vector.localForm() as vector_local:
            self._cpp_object.backsubstitution(vector_local.array_w)
        vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
