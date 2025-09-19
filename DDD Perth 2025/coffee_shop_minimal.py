"""
coffee_shop_minimal.py
Single-barista coffee shop simulation (discrete-event) using SimPy.

Two demo stages toggled by comments at the bottom:
- Stage 1: verbose logs (no visuals). Prints when customers arrive, start, and finish.
- Stage 2: concise KPIs + visualisation (queue length and wait times).

Guiding star: SimPy is free, easy, and you can do it today.

LIVE TODO markers indicate lines you can type or uncomment during the demo.
"""

from __future__ import annotations
import random
import statistics
import simpy

# LIVE TODO: Uncomment when you flip to Stage 2 visuals
# from visual import plot_queue_and_waits

RANDOM_SEED = 1  # How good is Douglas Adams
SIM_MINUTES = 60  # simulate 60 minutes from "open"
ARRIVAL_RATE = 1 / 2.0  # mean interarrival 2.0 minutes (exp arrivals)
SERVICE_MEAN = 2.5      # mean service time 2.5 minutes (exp service)

# LIVE TODO: You can show these as the "two numbers to tweak"
# ARRIVAL_RATE = 1 / 1.2
# SERVICE_MEAN = 2.8


def expovariate_time(mean: float) -> float:
    return random.expovariate(1 / mean)


def p95(xs):
    xs_sorted = sorted(xs)
    if not xs_sorted:
        return 0.0
    idx = int(0.95 * (len(xs_sorted) - 1))
    return xs_sorted[idx]


def customer(env: simpy.Environment, name: str, barista: simpy.Resource, results: dict, verbose: bool):
    arrival = env.now
    if verbose:
        print(f"[{arrival:6.2f}] {name} arrives (queue={len(barista.queue)})")

    with barista.request() as req:
        yield req
        start = env.now
        wait = start - arrival
        if verbose:
            print(f"[{start:6.2f}] {name} starts service (wait={wait:.2f} min)")

        # LIVE TODO: Type the service_time line live in Stage 1
        service_time = expovariate_time(SERVICE_MEAN)
        yield env.timeout(service_time)

        finish = env.now
        if verbose:
            print(f"[{finish:6.2f}] {name} finishes (service={service_time:.2f} min)")

        # Metrics
        results["waits"].append(wait)
        results["system_times"].append(finish - arrival)
        results["waits_over_time"].append((finish, wait))
        results["arrivals"].append(arrival)
        results["start_times"].append(start)
        results["finish_times"].append(finish)


def arrival_process(env: simpy.Environment, barista: simpy.Resource, results: dict, verbose: bool):
    i = 0
    while env.now < SIM_MINUTES:
        i += 1
        # LIVE TODO: Type the interarrival draw live in Stage 1
        inter = expovariate_time(1 / ARRIVAL_RATE)  # mean interarrival = 1/ARRIVAL_RATE
        yield env.timeout(inter)

        env.process(customer(env, f"Customer {i}", barista, results, verbose))

        # Snapshot queue after arrival
        results["queue_len"].append((env.now, len(barista.queue)))


def run(plot: bool = False, verbose: bool = True):
    """
    Run the simulation once and print KPIs.
    - plot=False, verbose=True -> Stage 1 (event log, no visuals)
    - plot=True, verbose=False -> Stage 2 (KPIs + visuals)
    """
    random.seed(RANDOM_SEED)
    env = simpy.Environment()
    # LIVE TODO: Type the Resource line live in Stage 1
    barista = simpy.Resource(env, capacity=1)

    results = {
        "waits": [],
        "system_times": [],
        "queue_len": [],
        "waits_over_time": [],
        "arrivals": [],
        "start_times": [],
        "finish_times": [],
    }

    env.process(arrival_process(env, barista, results, verbose))
    env.run(until=SIM_MINUTES)

    # KPIs
    waits = results["waits"]
    if waits:
        print(f"Served: {len(waits)}")
        print(f"Avg wait: {statistics.mean(waits):.2f} min")
        print(f"Median wait: {statistics.median(waits):.2f} min")
        print(f"95th wait: {p95(waits):.2f} min")
        print(f"Max queue: {max((q for _, q in results['queue_len']), default=0)}")
    else:
        print("No customers served.")

    if plot:
        # LIVE TODO: Uncomment to enable the plots in Stage 2
        from visual import plot_queue_and_waits
        plot_queue_and_waits(results["queue_len"], results["waits_over_time"],
                             title="Minimal: 1 barista, exp arrivals/service")


if __name__ == "__main__":
    # Stage 1: verbose event log (no visualisation)
    #run(plot=False, verbose=True)

    # Stage 2: concise KPIs + visualisation (comment Stage 1 above, then uncomment below)
    run(plot=True, verbose=False)