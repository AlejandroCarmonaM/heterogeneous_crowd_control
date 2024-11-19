import csv
import os
import matplotlib.pyplot as plt

import benchmark_demo

# Constants
TASK = benchmark_demo.TASK
AVG_RESULTS_FILE = benchmark_demo.AVERAGE_FILE
PLOTS_FOLDER = "plots"


def read_avg_results(scenario):
    results = {}
    with open(AVG_RESULTS_FILE, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header
        for row in reader:
            implementation, num_threads, time_taken, row_scenario = row
            if row_scenario != scenario:
                continue
            num_threads = int(num_threads)
            time_taken = float(time_taken)
            results[(implementation, num_threads)] = time_taken
    return results


def calculate_speedup(results, sequential_time):
    speedups = {}
    for (implementation, num_threads), time in results.items():
        speedup = sequential_time / time
        speedups[(implementation, num_threads)] = speedup
    return speedups


def get_impl_speedups(speedups, target_impl):
    threads = []
    impl_speedups = []

    for (impl, thread_value), speed in speedups.items():
        if impl == target_impl:
            threads.append(thread_value)
            impl_speedups.append(speed)

    return (threads, impl_speedups)


def plot_speedup(speedups, impl):
    threads, impl_speedups = get_impl_speedups(speedups, impl)

    plt.figure(figsize=(10, 6))
    plt.plot(
        threads,
        impl_speedups,
        label=benchmark_demo.IMPLEMENTATIONS[impl].name,
        marker=benchmark_demo.IMPLEMENTATIONS[impl].marker,
        color=benchmark_demo.IMPLEMENTATIONS[impl].color,
    )

    # Titles and labels
    plt.title(f"Speedup vs Number of Threads ({impl} vs Sequential)")
    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup")
    plt.grid(True)
    plt.xticks(benchmark_demo.THREADS)  # Ensure thread counts are marked
    plt.legend()

    # Save the plot
    filename = f"{PLOTS_FOLDER}/{TASK}_{impl}_speedup_plot.png"
    print(f"Creating {filename}")
    plt.savefig(filename)
    plt.close()


def plot_all_speedups(speedups, scenario):
    plt.figure(figsize=(10, 6))

    for name, details in benchmark_demo.IMPLEMENTATIONS.items():
        threads, impl_speedups = get_impl_speedups(speedups, name)
        if len(impl_speedups) == 0:
            continue
        if len(threads) <= 1:
            threads = benchmark_demo.THREADS
            impl_speedups = impl_speedups * len(benchmark_demo.THREADS)
        plt.plot(
            threads,
            impl_speedups,
            label=details.name,
            marker=details.marker,
            color=details.color,
        )

    # Titles and labels
    plt.title(f"Speedup vs Number of Threads {scenario}")
    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup")
    plt.grid(True)
    plt.xticks(benchmark_demo.THREADS)  # Ensure thread counts are marked
    plt.legend()

    # Save the plot
    filename = f"{PLOTS_FOLDER}/{TASK}_{scenario}_all_speedup_plot.png"
    print(f"Creating {filename}")
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    os.makedirs(PLOTS_FOLDER, exist_ok=True)

    for scenario in benchmark_demo.SCENARIOS:
        results = read_avg_results(scenario)

        if ("sequential", 1) not in results:
            if ("col_prevent_seq", 1) not in results:
                continue
            sequential_time = results[("col_prevent_seq", 1)]
        else:
            sequential_time = results[("sequential", 1)]

        speedups = calculate_speedup(results, sequential_time)

        # for impl, details in benchmark_demo.IMPLEMENTATIONS.items():
        #     if details.multithread:
        #         plot_speedup(speedups, impl)

        plot_all_speedups(speedups, scenario)
