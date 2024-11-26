//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2016
//
// Model coordinates a time step in a scenario: for each
// time step all agents need to be moved by one position if
// possible.
//
#ifndef _ped_model_h_
#define _ped_model_h_
#include <iostream>
#include <thread>
#include <vector>

#include "heatmap.h"
#include "ped_agent_soa.h"

namespace Ped {
// class Tagent;

// The implementation modes for Assignment 1 + 2 + 3:
// chooses which implementation to use for tick()
enum IMPLEMENTATION { SEQ, OMP, PTHREAD, CUDA, VECTOR, COL_PREVENT_SEQ, COL_PREVENT_PAR };

class Model {
 public:
  Model(std::vector<Ped::Tagent*> agentsInScenario, IMPLEMENTATION impl, int n_threads,
        Heatmap::HEATMAP_IMPL heatmap_impl);
  ~Model();

  void tick();

  TagentSoA* getAgentSoA() const { return agents_soa; };

  IMPLEMENTATION getImplementation() const { return impl; }

  bool isCheckingCollisions() { return impl == COL_PREVENT_SEQ || impl == COL_PREVENT_PAR; }
  bool printCollisions() { return agents_soa->printCollisions(); }

  int const* const* getHeatmap() const {
    if (heatmap == nullptr) {
      return nullptr;
    } else {
      return heatmap->getHeatmap();
    }
  };

  int getHeatmapSize() const { return heatmap->getHeatmapSize(); }
  void print_gpu_heatmap_avg_timings(int n_steps) { heatmap->printHeatmapCUDATimings(n_steps); };
  void print_seq_heatmap_timings(int n_steps) { heatmap->printHeatmapSeqTimings(n_steps); };
  void print_diff_timings(int n_steps);

 private:
  IMPLEMENTATION impl;
  int n_threads;
  Heatmap::HEATMAP_IMPL heatmap_impl;

  TagentSoA* agents_soa = nullptr;
  uint64_t total_cpu_time = 0;
  uint64_t total_gpu_time = 0;

  Heatmap* heatmap = nullptr;
};
}  // namespace Ped
#endif
