import subprocess
import csv
from collections import defaultdict
from collections import namedtuple
import os
import sys

TASK = "task3"

N_ITERS = 5

ImplementationDetails = namedtuple(
    "ImplementationDetails", ["name", "color", "marker", "multithread"]
)

IMPLEMENTATIONS = {
    "sequential": ImplementationDetails("sequential", "red", None, False),
    "OMP": ImplementationDetails("OMP", "blue", "o", True),
    "pthreads": ImplementationDetails("pthreads", "orange", "o", True),
    "vector": ImplementationDetails("vector + OMP", "purple", "o", True),
    "CUDA": ImplementationDetails("CUDA", "green", None, False),
    "col_prevent_seq": ImplementationDetails("col_prevent_seq", "red", None, False),
    "col_prevent_par": ImplementationDetails("col_prevent_par", "blue", "o", True),
}

THREADS = [1, 2, 4]

SCENARIOS = [
    # "commute_200000.xml",
    "lab3-scenario.xml",
    # "lab3-scenario-uneven.xml",
    # "scenario_small.xml",
    "worst_scenario.xml",
    "big_scenario.xml",
]

COMMAND_BASE = "../demo/demo --timing-mode --implementation={impl_name} -n{num_threads} ../demo/{scenario}"
OUTPUT_FOLDER = "results"
OUTPUT_FILE = f"{OUTPUT_FOLDER}/{TASK}_all_results.csv"
AVERAGE_FILE = f"{OUTPUT_FOLDER}/{TASK}_avg_results.csv"


def run_benchmark(selected_impls=None):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["IMPLEMENTATION", "NUM_THREADS", "TIME(s)"])

        if selected_impls:
            for selected_impl in selected_impls:
                if selected_impl in IMPLEMENTATIONS:
                    if IMPLEMENTATIONS[selected_impl].multithread:
                        for num_threads in THREADS:
                            run_implementation(selected_impl, num_threads, writer)
                    else:
                        run_implementation(selected_impl, 1, writer)
                else:
                    print(f"Unknown implementation: {selected_impl}")
        else:
            for name, details in IMPLEMENTATIONS.items():
                if details.multithread:
                    for num_threads in THREADS:
                        run_implementation(name, num_threads, writer)
                else:
                    run_implementation(name, 1, writer)


def run_implementation(impl_name, num_threads, writer):
    for scenario in SCENARIOS:
        for _ in range(N_ITERS):
            command = COMMAND_BASE.format(
                impl_name=impl_name, num_threads=num_threads, scenario=scenario
            )

            try:
                print(f'Running "{command}"')
                result = subprocess.run(
                    command.split(), capture_output=True, text=True, check=True
                )

                # Parse the output
                output = result.stdout.strip()
                _, _, time_taken = output.split(",")
                time_taken = float(time_taken)

                # Write to CSV
                writer.writerow([impl_name, num_threads, time_taken, scenario])
            except subprocess.CalledProcessError as e:
                print(f"Error occurred while running {command}: {e}")


def calculate_averages(scenario):
    # Data structure to store times for each (implementation, num_threads)
    # Default value if key not present is empty list
    results = defaultdict(list)

    # Read the raw results
    if not os.path.exists(OUTPUT_FILE):
        print(f"{OUTPUT_FILE} not found. Please run the benchmarks first.")
        return

    with open(OUTPUT_FILE, "r") as outfile:
        reader = csv.reader(outfile)
        next(reader)  # Skip the header
        for row in reader:
            implementation, num_threads, time_taken, row_scenario = row
            if row_scenario != scenario:
                continue
            num_threads = int(num_threads)
            time_taken = float(time_taken)
            results[(implementation, num_threads)].append(time_taken)

    # Calculate averages and write to a new CSV file
    with open(AVERAGE_FILE, "a", newline="") as outfile:
        writer = csv.writer(outfile)

        for (implementation, num_threads), times in results.items():
            avg_time = sum(times) / len(times)
            writer.writerow([implementation, num_threads, avg_time, scenario])


if __name__ == "__main__":
    # There's errors if make is called from this script, so make should be manually run first
    if not os.path.isfile("../demo/demo"):
        print(
            "Please run 'make' on the root directory of this project", file=sys.stderr
        )
        sys.exit(-1)

    selected_impls = sys.argv[1:] if len(sys.argv) > 1 else None
    run_benchmark(selected_impls)
    with open(AVERAGE_FILE, "w", newline="") as outfile:
        outfile.write("IMPLEMENTATION, NUM_THREADS, TIME(s), SCENARIO\n")
    for scenario in SCENARIOS:
        calculate_averages(scenario)
