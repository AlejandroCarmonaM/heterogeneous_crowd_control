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
                  HEATMAP_IMPL heatmapImpl)
    : impl(impl), n_threads(n_threads), heatmapImpl(heatmapImpl) {
  // agents = std::vector<Ped::Tagent*>(agentsInScenario.begin(),
  //                                    agentsInScenario.end());

  agents_soa = new TagentSoA(agentsInScenario);

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

  switch (heatmapImpl) {
    case SEQ_HM:
      setupHeatmapSeq();
      break;
    case PAR_HM:
      setupHeatmapCUDA();
      break;
    default:
      break;
  }
}

////////////
/// Everything below here relevant for Assignment 3.
/// Don't use this for Assignment 1!
///////////////////////////////////////////////

Ped::Model::~Model() {
  if (agents_soa != nullptr) {
    if (impl == IMPLEMENTATION::CUDA) {
      agents_soa->freeCUDA();
    }
    delete agents_soa;
    agents_soa = nullptr;
  }

  switch (heatmapImpl) {
    case SEQ_HM:
      freeHeatmapSeq();
      break;
    case PAR_HM:
      freeHeatmapCUDA();
      break;

    default:
      break;
  }
}
