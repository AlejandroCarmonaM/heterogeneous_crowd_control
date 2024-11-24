//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2015
//
#include "ped_model.h"

#include <omp.h>

#include <iostream>

#include "tick_cuda.h"

Ped::Model::Model(std::vector<Ped::Tagent*> agentsInScenario, IMPLEMENTATION impl, int n_threads,
                  Heatmap::HEATMAP_IMPL heatmap_impl)
    : impl(impl), n_threads(n_threads), heatmap_impl(heatmap_impl) {
  agents_soa = new TagentSoA(agentsInScenario,
                             (heatmap_impl == Heatmap::PAR_HM || heatmap_impl == Heatmap::HET_HM));

  if (isCheckingCollisions() && agents_soa->printCollisions()) {
    cerr << "ERROR: Collisions found in initial positions" << endl;
    exit(EXIT_FAILURE);
  }

  switch (impl) {
    case IMPLEMENTATION::SEQ:
      break;
    case IMPLEMENTATION::OMP:
      omp_set_num_threads(n_threads);
      break;
    case IMPLEMENTATION::VECTOR:
      omp_set_num_threads(n_threads);
      break;
    case IMPLEMENTATION::PTHREAD:
      agents_soa->setupPthreads(n_threads);
      break;
    case IMPLEMENTATION::CUDA:
      agents_soa->setupCUDA();
      break;
    case IMPLEMENTATION::COL_PREVENT_SEQ:
      agents_soa->setupColCheckSeq();
      break;
    case IMPLEMENTATION::COL_PREVENT_PAR:
      omp_set_num_threads(n_threads);
      agents_soa->setupColCheckPar();
      break;
  }

  if (heatmap_impl != Heatmap::NONE) {
    heatmap = new Heatmap(heatmap_impl, agents_soa);
  }
}

Ped::Model::~Model() {
  if (agents_soa != nullptr) {
    if (impl == IMPLEMENTATION::CUDA) {
      agents_soa->freeCUDA();
    }
    delete agents_soa;
    agents_soa = nullptr;
  }

  delete heatmap;
  heatmap = nullptr;
}

void Ped::Model::tick() {
  std::thread heatmap_thread;
  auto start = std::chrono::high_resolution_clock::now();

  std::chrono::_V2::system_clock::time_point gpu_end;

  // TODO: Make this cleaner
  static bool first = true;
  if (!first) {
    switch (heatmap_impl) {
      case Heatmap::SEQ_HM:
        heatmap->updateHeatmapSeq();
        break;
      case Heatmap::PAR_HM:
        heatmap->copyDesiredPosToGPU();
        heatmap_thread = std::thread(&Heatmap::updateHeatmapCUDA, heatmap, &gpu_end);
        break;
      case Heatmap::HET_HM:
        heatmap->copyDesiredPosToGPU();
        heatmap_thread = std::thread(&Heatmap::updateHeatmapCUDA, heatmap, &gpu_end);
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

      auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - start);
      total_cpu_time += cpu_time.count();

      auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - start);
      total_gpu_time += gpu_time.count();
    }
  }

  if (first) {
    first = false;
  }
}

void Ped::Model::print_diff_timings(int n_steps) {
  float avg_cpu = total_cpu_time / (float)n_steps / 1000;
  float avg_gpu = total_gpu_time / (float)n_steps / 1000;
  float avg_diff = abs(avg_cpu - avg_gpu);

  cout << "Average CPU time: " << avg_cpu << " ms" << endl;
  cout << "Average GPU time: " << avg_gpu << " ms" << endl;
  cout << "Average time diff: " << avg_diff << " ms" << endl;

  cout << "Average imbalance: " << avg_diff / max(avg_cpu, avg_gpu) * 100 << "%" << endl;
}