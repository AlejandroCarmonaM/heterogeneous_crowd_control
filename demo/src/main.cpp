///////////////////////////////////////////////////
// Low Level Parallel Programming 2016.
//
//     ==== Don't change this file! ====
//
// The main starting point for the crowd simulation.
//

#include <unistd.h>

#include <QApplication>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QTimer>
#include <chrono>
#include <cstring>
#include <ctime>
#include <iostream>
#include <string>
#include <thread>

// #include "MainWindow.h"
#include "ParseScenario.h"
#include "PedSimulation.h"
#include "ped_agent_soa.h"
#include "ped_model.h"

void printHelp(char* program_name) {
  cout << "Usage: " << program_name
       << " [-h|--help] [--timing-mode]  [--heatmap] "
          "[--implementation=(CUDA,vector,OMP,pthreads,sequential,col_prevent_seq,col_prevent_par)]"
          " [-n(NUM_THREADS)] SCENARIO_FILE"
       << endl;
  cout << "e.g.: " << program_name
       << " --timing-mode --heatmap --implementation=OMP -n12 demo/commute_200000.xml" << endl;
  cout << "(NUM_THREADS must be > 0 and < " << Ped::TagentSoA::MAX_THREADS << ")" << endl;
}

int main(int argc, char* argv[]) {
  QString scenefile = "";

  bool timing_mode = false;
  int n_threads = 1;
  Heatmap::HEATMAP_IMPL heatmap = Heatmap::NONE;

  const char* impl_arg = "implementation=";
  string impl_str = "sequential";
  map<string, Ped::IMPLEMENTATION> impl_map = {
      {"sequential", Ped::IMPLEMENTATION::SEQ},
      {"OMP", Ped::IMPLEMENTATION::OMP},
      {"pthreads", Ped::IMPLEMENTATION::PTHREAD},
      {"vector", Ped::IMPLEMENTATION::VECTOR},
      {"CUDA", Ped::IMPLEMENTATION::CUDA},
      {"col_prevent_seq", Ped::IMPLEMENTATION::COL_PREVENT_SEQ},
      {"col_prevent_par", Ped::IMPLEMENTATION::COL_PREVENT_PAR},
  };
  Ped::IMPLEMENTATION impl = impl_map.at(impl_str);

  // Argument handling
  for (int i = 1; i < argc; i++) {
    if (argv[i][0] == '-') {
      if (argv[i][1] == '-') {
        if (strcmp(&argv[i][2], "timing-mode") == 0) {
          timing_mode = true;

        } else if (strcmp(&argv[i][2], "heatmap_seq") == 0) {
          heatmap = Heatmap::SEQ_HM;

        } else if (strcmp(&argv[i][2], "heatmap_par") == 0) {
          heatmap = Heatmap::PAR_HM;

        } else if (strcmp(&argv[i][2], "heatmap_het") == 0) {
          heatmap = Heatmap::HET_HM;

        } else if (strcmp(&argv[i][2], "help") == 0) {
          printHelp(argv[0]);
          return 0;

        } else if (strncmp(&argv[i][2], impl_arg, strlen(impl_arg)) == 0) {
          impl_str = string(&argv[i][2] + strlen(impl_arg));
          if (impl_map.find(impl_str) == impl_map.end()) {
            cerr << "Unrecognized implementation: \"" << impl_str << "\"" << endl;
            return -1;
          }
          impl = impl_map.at(impl_str);

        } else {
          cerr << "Unrecognized command: \"" << argv[i] << endl;
          return -1;
        }

      } else if (argv[i][1] == 'n') {
        n_threads = atoi(&argv[i][2]);
        if (n_threads <= 0 || n_threads > Ped::TagentSoA::MAX_THREADS) {
          cerr << "Invalid number of threads: " << n_threads << endl;
          cerr << "(must be > 0 and < " << Ped::TagentSoA::MAX_THREADS << ")" << endl;
          return -1;
        }

      } else if (argv[i][1] == 'h' && strlen(argv[i]) == 2) {
        printHelp(argv[0]);
        return 0;

      } else {
        cerr << "Unrecognized command: \"" << argv[i] << endl;
        return -1;
      }

    } else {
      // Assume it is a path to scenefile
      scenefile = argv[i];
    }
  }

  // If impl is sequential, then n_threads should be 1
  if (impl == Ped::IMPLEMENTATION::SEQ || impl == Ped::IMPLEMENTATION::COL_PREVENT_SEQ ||
      impl == Ped::IMPLEMENTATION::CUDA) {
    if (n_threads > 1) {
      n_threads = 1;

      if (!timing_mode) {
        cout << "Setting number of threads to " << n_threads << endl;
      }
    }
  }

  // Reading the scenario file and setting up the crowd simulation model
  if (scenefile.isEmpty()) {
    cerr << "Please specify a scenario" << endl;
    printHelp(argv[0]);
    return -1;
  }
  ParseScenario parser(scenefile);

  auto parser_agents = parser.getAgents();

  if (!timing_mode) {
    cout << "Creating model for " << parser_agents.size() << " agents" << endl;
  }

  Ped::Model model(parser_agents, impl, n_threads, heatmap);

  // GUI related set ups
  QApplication app(argc, argv);
  MainWindow mainwindow(model);

  // Default number of steps to simulate
  const int maxNumberOfStepsToSimulate = 100;
  PedSimulation* simulation = new PedSimulation(model, mainwindow);

  if (!timing_mode) {
    cout << "Demo setup complete, running " << impl_str << " implementation" << " with "
         << n_threads << " threads" << endl;
  }
  int retval = 0;
  // Timing of simulation
  std::chrono::time_point<std::chrono::system_clock> start, stop;
  start = std::chrono::system_clock::now();

  if (timing_mode) {
    // Simulation mode to use when profiling (without any GUI)
    simulation->runSimulationWithoutQt(maxNumberOfStepsToSimulate);
  } else {
    // Simulation mode to use when visualizing
    mainwindow.show();
    simulation->runSimulationWithQt(maxNumberOfStepsToSimulate);
    retval = app.exec();
  }

  // End timing
  stop = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = stop - start;

  if (timing_mode) {
    // Header for CSV Format -> Fields={IMPLEMENTATION,NUM_THREADS,TIME(s)}
    // cout << "IMPLEMENTATION,NUM_THREADS,TIME(s)" << endl;
    cout << impl_str << "," << n_threads << "," << elapsed_seconds.count() << endl;

    switch (heatmap) {
      case Heatmap::SEQ_HM:
        model.print_seq_heatmap_timings(maxNumberOfStepsToSimulate);
      case Heatmap::PAR_HM:
        model.print_gpu_heatmap_avg_timings(maxNumberOfStepsToSimulate);
        break;
      case Heatmap::HET_HM:
        model.print_seq_heatmap_timings(maxNumberOfStepsToSimulate);
        model.print_gpu_heatmap_avg_timings(maxNumberOfStepsToSimulate);
        model.print_diff_timings(maxNumberOfStepsToSimulate);
        break;
      case Heatmap::NONE:
        break;
    }
  } else {
    cout << "Time: " << elapsed_seconds.count() << " seconds." << endl;
  }

  // print timings

  delete (simulation);

  return retval;
}
