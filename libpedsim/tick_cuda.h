#ifndef _tick_cuda_h_
#define _tick_cuda_h_

#include <iostream>

#include "ped_agent_soa.h"
/**
 * These functions are placeholders for your functions
 * implementing CUDA solutions.
 * They are weakly linked. If there is any other function with the same
 * signature at link-time the conflict is resolved by scrapping the weak one.
 *
 * You should only care if you want to compile on a non-CUDA machine.
 */

void __attribute__((weak)) Ped::TagentSoA::callTickCUDA() {
  std::cerr << "Notice: calling a dummy function" << __FUNCTION__ << std::endl;
}

void __attribute__((weak)) setupCUDA() {
  std::cerr << "Notice: calling a dummy function" << __FUNCTION__ << std::endl;
}

void __attribute__((weak)) freeCUDA() {
  std::cerr << "Notice: calling a dummy function" << __FUNCTION__ << std::endl;
}

void __attribute__((weak)) Ped::Model::setupHeatmapCUDA() {
  std::cerr << "Notice: calling a dummy function" << __FUNCTION__ << std::endl;
}

void __attribute__((weak)) Ped::Model::copyDesiredPosToGPU() {
  std::cerr << "Notice: calling a dummy function" << __FUNCTION__ << std::endl;
}

void __attribute__((weak)) Ped::Model::updateHeatmapCUDA() {
  std::cerr << "Notice: calling a dummy function" << __FUNCTION__ << std::endl;
}

void __attribute__((weak)) Ped::Model::freeHeatmapCUDA() {
  std::cerr << "Notice: calling a dummy function" << __FUNCTION__ << std::endl;
}

#endif
