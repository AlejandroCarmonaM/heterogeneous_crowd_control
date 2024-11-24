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

  // TODO: Some heat on upper left corner for first frame when calling this on uninitiallized
  // desired_pos

  void tick() {
    std::thread heatmap_thread;

    // TODO: Make this cleaner
    static bool first = true;
    if (!first) {
      switch (heatmap_impl) {
        case Heatmap::SEQ_HM:
          heatmap->updateHeatmapSeq();
          break;
        case Heatmap::PAR_HM:
          heatmap->copyDesiredPosToGPU();
          heatmap_thread = std::thread(&Heatmap::updateHeatmapCUDA, heatmap);
          break;
        case Heatmap::HET_HM:
          heatmap->copyDesiredPosToGPU();
          heatmap_thread = std::thread(&Heatmap::updateHeatmapCUDA, heatmap);
          heatmap->updateHeatmapSeq();
          break;
        case Heatmap::NONE:
          break;
      }
    }

    switch (impl) {
      case SEQ: {
        agents_soa->seqTick();
        break;
      }
      case OMP: {
        agents_soa->ompTick();
        break;
      }
      case PTHREAD: {
        agents_soa->pthreadTick();
        break;
      }
      case VECTOR: {
        agents_soa->vectorTick();
        break;
      }
      case CUDA: {
        agents_soa->callTickCUDA();
        break;
      }
      case COL_PREVENT_SEQ: {
        agents_soa->colPreventSeqTick();
        break;
      }
      case COL_PREVENT_PAR: {
        agents_soa->colPreventParTick();
        break;
      }
    }

    if (!first) {
      if (heatmap_impl == Heatmap::PAR_HM || heatmap_impl == Heatmap::HET_HM) {
        auto cpu_end = std::chrono::high_resolution_clock::now();

        heatmap_thread.join();

        auto gpu_end = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - cpu_end);
        total_diff += diff.count();
      }
    }

    if (first) {
      first = false;
    }
  }

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
  Heatmap::HEATMAP_IMPL heatmap_impl;
  int n_threads;

  TagentSoA* agents_soa = nullptr;
  float total_diff = 0.0;

  Heatmap* heatmap = nullptr;
};
}  // namespace Ped
#endif
