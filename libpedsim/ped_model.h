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
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

// #include "ped_agent.h"
#include "ped_agent_soa.h"

namespace Ped {
// class Tagent;

// The implementation modes for Assignment 1 + 2 + 3:
// chooses which implementation to use for tick()
enum IMPLEMENTATION { SEQ, OMP, PTHREAD, CUDA, VECTOR, COL_PREVENT_SEQ, COL_PREVENT_PAR };
enum HEATMAP_IMPL { SEQ_HM, PAR_HM, NONE };

class Model {
 public:
  Model(std::vector<Ped::Tagent*> agentsInScenario, IMPLEMENTATION impl, int n_threads,
        HEATMAP_IMPL heatmapImpl);
  ~Model();

  // Coordinates a time step in the scenario: move all agents by one step (if
  // applicable).
  // TODO: Some heat on upper left corner for first frame when calling this on uninitiallized
  // desired_pos
  void tick() {
    std::thread heatmapThread;
    switch (heatmapImpl) {
      case SEQ_HM:
        updateHeatmapSeq();
        break;
      case PAR_HM:
        copyDesiredPosToGPU();
        heatmapThread = std::thread(&Ped::Model::updateHeatmapCUDA, this);
        break;

      default:
        break;
    }

#include <chrono>
#include <iostream>

    auto cpu_start = std::chrono::high_resolution_clock::now();

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

    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = cpu_end - cpu_start;
    total_tick_time += elapsed.count();

    if (heatmapImpl == PAR_HM) {
      // sync with the updateHeatmapCUDA calling thread
      heatmapThread.join();

      auto gpu_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = gpu_end - cpu_end;
      total_diff += diff.count();
    }
  }

  // Returns the agents of this scenario
  // const std::vector<Tagent*> getAgents() const { return agents; };

  TagentSoA* getAgentSoA() const { return agents_soa; };

  IMPLEMENTATION getImplementation() const { return impl; }

  bool isCheckingCollisions() { return impl == COL_PREVENT_SEQ || impl == COL_PREVENT_PAR; }
  bool printCollisions() { return agents_soa->printCollisions(); }

  // Returnst the heatmap visualizing the density of agents
  int const* const* getHeatmap() const { return blurred_heatmap; };

  int getHeatmapSize() const;
  void print_gpu_heatmap_avg_timings(int n_steps);
  void print_seq_heatmap_timings(int n_steps);

 private:
  IMPLEMENTATION impl;
  int n_threads;
  HEATMAP_IMPL heatmapImpl;

  TagentSoA* agents_soa = nullptr;
  float total_tick_time = 0.0;
  float total_diff = 0.0;
  float total_heatmap_seq_time = 0.0;

  // The agents in this scenario
  // std::vector<Tagent*> agents;

#define SIZE 1024
#define CELLSIZE 5
#define SCALED_SIZE SIZE* CELLSIZE
  // The heatmap representing the density of agents
  int** heatmap = nullptr;

  // The scaled heatmap that fits to the view
  int** scaled_heatmap = nullptr;

  // The final heatmap: blurred and scaled to fit the view
  int** blurred_heatmap = nullptr;

  void setupHeatmapSeq();
  // TODO:
  void updateHeatmapSeq();
  void freeHeatmapSeq();

  void setupHeatmapCUDA();
  void copyDesiredPosToGPU();
  void updateHeatmapCUDA();
  void freeHeatmapCUDA();
};
}  // namespace Ped
#endif
