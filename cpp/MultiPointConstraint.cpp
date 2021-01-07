// Copyright (C) 2019 Jorgen S. Dokken
//
// This file is part of DOLFINX_MPC
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MultiPointConstraint.h"
#include "utils.h"
#include <Eigen/Dense>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/common/span.hpp>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/dofmapbuilder.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Mesh.h>
#include <iostream>
using namespace dolfinx_mpc;

MultiPointConstraint::MultiPointConstraint(
    std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> slaves,
    std::int32_t num_local_slaves)
    : _V(V), _slaves(), _slave_cells(), _cell_to_slaves_map(),
      _slave_to_cells_map(), _num_local_slaves(num_local_slaves), _master_map(),
      _coeff_map(), _owner_map(), _master_local_map(), _master_block_map(),
      _dofmap()
{
  _slaves
      = std::make_shared<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>(slaves);
  auto [slave_to_cells, cell_to_slaves_map] = create_cell_maps(slaves);
  auto [unique_cells, cell_to_slaves] = cell_to_slaves_map;
  _slave_cells
      = std::make_shared<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>>(
          unique_cells);
  _cell_to_slaves_map = cell_to_slaves;
  _slave_to_cells_map = slave_to_cells;
}

std::pair<
    std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>,
    std::pair<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>,
              std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>>>
MultiPointConstraint::create_cell_maps(
    Eigen::Array<std::int32_t, Eigen::Dynamic, 1> dofs)
{
  dolfinx::common::Timer timer("~MPC: Create slave->cell  and cell->slave map");
  const dolfinx::mesh::Mesh& mesh = *(_V->mesh());
  const dolfinx::fem::DofMap& dofmap = *(_V->dofmap());
  const int tdim = mesh.topology().dim();
  const int num_cells = mesh.topology().index_map(tdim)->size_local();
  const int bs = dofmap.index_map_bs();
  const int num_local_dofs
      = (dofmap.index_map->size_local() + dofmap.index_map->num_ghosts()) * bs;

  // Create an array for the whole function space, where
  // dof index contains the index of the dof if it is in the list,
  // otherwise -1
  std::vector<std::int32_t> dof_index(num_local_dofs, -1);
  std::map<std::int32_t, std::vector<std::int32_t>> idx_to_cell;
  for (std::int32_t i = 0; i < dofs.rows(); ++i)
  {
    std::vector<std::int32_t> cell_vec;
    idx_to_cell.insert({i, cell_vec});
    dof_index[dofs[i]] = i;
  }
  // Loop through all cells and make cell to dof index and index to cell map
  std::vector<std::int32_t> unique_cells;
  std::map<std::int32_t, std::vector<std::int32_t>> cell_to_idx;
  for (std::int32_t i = 0; i < num_cells; ++i)
  {
    auto cell_dofs = dofmap.cell_dofs(i);
    bool any_dof_in_cell = false;
    std::vector<std::int32_t> dofs_in_cell;
    for (std::int32_t j = 0; j < cell_dofs.size(); ++j)
    {
      // Check if cell dof in dofs list
      for (Eigen::Index k = 0; k < bs; ++k)
      {
        const std::int32_t dof_idx = dof_index[bs * cell_dofs[j] + k];
        if (dof_idx != -1)
        {
          any_dof_in_cell = true;
          dofs_in_cell.push_back(dof_idx);
          idx_to_cell[dof_idx].push_back(i);
        }
      }
    }
    if (any_dof_in_cell)
    {
      unique_cells.push_back(i);
      cell_to_idx.insert({i, dofs_in_cell});
    }
  }
  // Flatten unstructured maps
  std::vector<std::int32_t> indices_to_cells;
  std::vector<std::int32_t> cells_to_indices;
  std::vector<std::int32_t> offsets_cells{0};
  std::vector<std::int32_t> offsets_idx{0};
  for (auto it = cell_to_idx.begin(); it != cell_to_idx.end(); ++it)
  {
    auto idx = it->second;
    cells_to_indices.insert(cells_to_indices.end(), idx.begin(), idx.end());
    offsets_cells.push_back(cells_to_indices.size());
  }
  for (auto it = idx_to_cell.begin(); it != idx_to_cell.end(); ++it)
  {
    auto idx = it->second;
    indices_to_cells.insert(indices_to_cells.end(), idx.begin(), idx.end());
    offsets_idx.push_back(cells_to_indices.size());
  }

  // Create dof to cells adjacency list
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> adj_ptr
      = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
          indices_to_cells, offsets_idx);

  // Create cell to dofs adjacency list
  Eigen::Map<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> dof_cells(
      unique_cells.data(), unique_cells.size());
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> adj2_ptr;
  adj2_ptr = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
      cells_to_indices, offsets_cells);

  return std::make_pair(adj_ptr, std::make_pair(dof_cells, adj2_ptr));
}
//-----------------------------------------------------------------------------
void MultiPointConstraint::add_masters(std::vector<std::int64_t> masters,
                                       std::vector<PetscScalar> coeffs,
                                       std::vector<std::int32_t> owners,
                                       std::vector<std::int32_t> offsets)
{
  _master_map = std::make_shared<dolfinx::graph::AdjacencyList<std::int64_t>>(
      masters, offsets);
  _coeff_map = std::make_shared<dolfinx::graph::AdjacencyList<PetscScalar>>(
      coeffs, offsets);
  _owner_map = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
      owners, offsets);

  // Create new index map with all masters
  create_new_index_map();
}
//-----------------------------------------------------------------------------
void MultiPointConstraint::create_new_index_map()
{
  dolfinx::common::Timer timer("~MPC: Create new index map");
  MPI_Comm comm = _V->mesh()->mpi_comm();
  const dolfinx::fem::DofMap& dofmap = *(_V->dofmap());

  std::shared_ptr<const dolfinx::common::IndexMap> index_map = dofmap.index_map;

  // Compute local master index before creating new index-map
  const std::int32_t& block_size = _V->dofmap()->index_map_bs();

  std::vector<std::int64_t> master_array = _master_map->array();
  for (std::size_t i = 0; i < master_array.size(); ++i)
    master_array[i] /= block_size;
  const std::vector<std::int32_t> master_as_local
      = _V->dofmap()->index_map->global_to_local(master_array);
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>
      old_master_block_map
      = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
          master_as_local, _master_map->offsets());
  int mpi_size = -1;
  MPI_Comm_size(comm, &mpi_size);
  if (mpi_size == 1)
  {
    _index_map = index_map;
    _master_block_map = old_master_block_map;
  }
  else
  {
    std::vector<std::int64_t> new_ghosts;
    std::vector<std::int32_t> new_ranks;

    for (Eigen::Index i = 0; i < old_master_block_map->num_nodes(); ++i)
    {
      for (Eigen::Index j = 0; j < old_master_block_map->links(i).size(); ++j)
      {
        // Check if master is already ghosted
        if (old_master_block_map->links(i)[j] == -1)
        {
          const int& master_as_int = _master_map->links(i)[j];
          const std::div_t div = std::div(master_as_int, block_size);
          const int& block = div.quot;
          auto already_ghosted
              = std::find(new_ghosts.begin(), new_ghosts.end(), block);

          if (already_ghosted == new_ghosts.end())
          {
            new_ghosts.push_back(block);
            new_ranks.push_back(_owner_map->links(i)[j]);
          }
        }
      }
    }
    const std::vector<std::int32_t>& old_ranks = index_map->ghost_owner_rank();
    std::vector<std::int64_t> old_ghosts = index_map->ghosts();
    const std::int32_t num_ghosts = old_ghosts.size();
    old_ghosts.resize(num_ghosts + new_ghosts.size());
    for (std::int64_t i = 0; i < new_ghosts.size(); i++)
      old_ghosts[num_ghosts + i] = new_ghosts[i];

    // Append rank of new ghost owners
    std::vector<int> ghost_ranks(old_ghosts.size());
    for (std::int64_t i = 0; i < old_ranks.size(); i++)
      ghost_ranks[i] = old_ranks[i];

    for (int i = 0; i < new_ghosts.size(); ++i)
    {
      ghost_ranks[num_ghosts + i] = new_ranks[i];
    }
    dolfinx::common::Timer timer_indexmap("MPC: Init indexmap");

    _index_map = std::make_shared<dolfinx::common::IndexMap>(
        comm, index_map->size_local(),
        dolfinx::MPI::compute_graph_edges(
            MPI_COMM_WORLD,
            std::set<int>(ghost_ranks.begin(), ghost_ranks.end())),
        old_ghosts, ghost_ranks);

    timer_indexmap.stop();
  }
  // Create new dofmap
  const dolfinx::fem::DofMap* old_dofmap = _V->dofmap().get();
  dolfinx::mesh::Topology topology = _V->mesh()->topology();
  dolfinx::fem::ElementDofLayout layout = *old_dofmap->element_dof_layout;
  auto [unused_indexmap, bs, o_dofmap] = dolfinx::fem::build_dofmap_data(
      _V->mesh()->mpi_comm(), topology, layout);
  _dofmap = std::make_shared<dolfinx::fem::DofMap>(
      old_dofmap->element_dof_layout, _index_map, bs, o_dofmap, bs);

  // Compute local master index before creating new index-map
  std::vector<std::int64_t> master_blocks = _master_map->array();
  for (std::size_t i = 0; i < master_blocks.size(); ++i)
    master_blocks[i] /= block_size;
  const std::vector<std::int32_t>& master_block_local
      = _index_map->global_to_local(master_blocks);
  _master_block_map
      = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
          master_block_local, _master_map->offsets());

  // Go from blocks to actual local index
  std::vector<std::int32_t> masters_local(master_block_local.size());
  std::int32_t c = 0;
  for (Eigen::Index i = 0; i < _master_map->num_nodes(); ++i)
  {
    auto masters = _master_map->links(i);
    for (Eigen::Index j = 0; j < masters.size(); ++j)
    {
      const int master_as_int = masters[j];
      const std::div_t div = std::div(master_as_int, block_size);
      const int rem = div.rem;
      masters_local[c] = master_block_local[c] * block_size + rem;
      c++;
    }
  }
  _master_local_map
      = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
          masters_local, _master_map->offsets());
}
//-----------------------------------------------------------------------------
/// Create MPC specific sparsity pattern
dolfinx::la::SparsityPattern MultiPointConstraint::create_sparsity_pattern(
    const dolfinx::fem::Form<PetscScalar>& a)
{
  LOG(INFO) << "Generating MPC sparsity pattern";
  dolfinx::common::Timer timer("~MPC: Create sparsity pattern");
  if (a.rank() != 2)
  {
    throw std::runtime_error(
        "Cannot create sparsity pattern. Form is not a bilinear form");
  }

  /// Check that we are using the correct function-space in the bilinear
  /// form otherwise the index map will be wrong

  assert(a.function_spaces().at(0) == _V);
  assert(a.function_spaces().at(1) == _V);
  auto bs0 = a.function_spaces().at(0)->dofmap()->index_map_bs();
  auto bs1 = a.function_spaces().at(1)->dofmap()->index_map_bs();
  assert(bs0 == bs1);
  const dolfinx::mesh::Mesh& mesh = *(a.mesh());

  std::array<std::shared_ptr<const dolfinx::common::IndexMap>, 2> new_maps;
  new_maps[0] = _index_map;
  new_maps[1] = _index_map;
  std::array<int, 2> bs = {bs0, bs1};
  dolfinx::la::SparsityPattern pattern(mesh.mpi_comm(), new_maps, bs);

  ///  Create and build sparsity pattern for original form. Should be
  ///  equivalent to calling create_sparsity_pattern(Form a)
  dolfinx_mpc::build_standard_pattern(pattern, a);
  // Arrays replacing slave dof with master dof in sparsity pattern
  const int& block_size = bs0;
  Eigen::Array<PetscInt, Eigen::Dynamic, 1> master_for_slave(block_size);
  Eigen::Array<PetscInt, Eigen::Dynamic, 1> master_for_other_slave(block_size);

  std::vector<std::int32_t> master_block(1);
  tcb::span<int32_t> master_span = tcb::make_span(master_block);
  std::vector<std::int32_t> other_master_block(1);
  tcb::span<int32_t> other_master_span = tcb::make_span(other_master_block);
  for (Eigen::Index i = 0; i < _cell_to_slaves_map->num_nodes(); ++i)
  {

    auto cell_dofs = _dofmap->cell_dofs(_slave_cells->array()[i]);
    auto slaves = _cell_to_slaves_map->links(i);

    // Arrays for flattened master slave data
    std::vector<std::int32_t> flattened_masters;
    for (std::int32_t j = 0; j < slaves.size(); ++j)
    {
      auto local_masters = _master_block_map->links(slaves[j]);
      for (std::int32_t k = 0; k < local_masters.size(); ++k)
        flattened_masters.push_back(local_masters[k]);
    }
    // Remove duplicate master blocks
    std::sort(flattened_masters.begin(), flattened_masters.end());
    flattened_masters.erase(
        std::unique(flattened_masters.begin(), flattened_masters.end()),
        flattened_masters.end());

    for (std::int32_t j = 0; j < flattened_masters.size(); ++j)
    {
      master_block[0] = flattened_masters[j];
      pattern.insert(master_span, cell_dofs);
      pattern.insert(cell_dofs, master_span);
      // Add sparsity pattern for all master dofs of any slave on this cell
      for (std::int32_t k = j + 1; k < flattened_masters.size(); ++k)
      {
        other_master_block[0] = flattened_masters[k];
        pattern.insert(master_span, other_master_span);
        pattern.insert(other_master_span, master_span);
      }
    }
  }
  return pattern;
};
//-----------------------------------------------------------------------------
void MultiPointConstraint::backsubstitution(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> vector)
{
  auto slaves = *_slaves;
  for (std::int32_t i = 0; i < slaves.size(); ++i)
  {
    auto masters = _master_local_map->links(i);
    auto coeffs = _coeff_map->links(i);
    assert(masters.size() == coeffs.size());
    for (std::int32_t k = 0; k < masters.size(); ++k)
      vector[slaves[i]] += coeffs[k] * vector[masters[k]];
  }
};
