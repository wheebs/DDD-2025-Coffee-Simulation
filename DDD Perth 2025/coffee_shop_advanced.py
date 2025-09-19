"""
coffee_shop_weekdays_multi_seed.py

A friendly, well-commented SimPy simulation of a coffee shop with:
- One shared queue (one register vibe).
- Drink mix with different service times (simple, standard, slow).
- Optional peak window and second barista in the peak.
- Weekday presets (Mon–Fri) to tweak arrivals and drink mix.
- Multi-seed runs (run the same day many times with different random seeds).
- Visualisation to compare days across multiple seeds.

Why multiple seeds?
- Simulations use randomness. Running many seeds shows variability and makes
  the results more robust (we can look at averages and spread, not just one run).

Requirements:
- simpy  (pip install simpy)
- matplotlib  (pip install matplotlib)
"""

from __future__ import annotations
import random
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import simpy
import matplotlib.pyplot as plt

# If you want visuals in Stage B, uncomment this import and the block at the bottom.
# from visual import plot_queue_and_waits


# -----------------------------------------------------------
# Scenario parameters
# -----------------------------------------------------------

@dataclass(frozen=True)
class Params:
    """
    All knobs in one tidy place. Change these up live to tell different stories.
    """
    minutes: int = 60                 # Simulate this many minutes (e.g., one “morning rush” hour)
    seed: int = 123                   # Random seed so your results are reproducible on stage

    # Arrival rates (customers per minute)
    # Tip: A higher rate means customers lob in more frequently.
    base_arrival_rate: float = 1 / 1.8    # average interarrival ~1.8 min
    peak_arrival_rate: float = 1 / 1.0    # average interarrival ~1.0 min during peak

    # Peak window (minutes since open)
    peak_start: float = 30.0
    peak_end: float = 45.0

    # Operational toggles
    use_peak: bool = False                 # Turn the morning peak ON/OFF
    second_barista_in_peak: bool = False   # Bring in a second barista during the rush

    # Visuals
    plot: bool = False                     # Plots off for Stage A; flip to True for Stage B


# Drink mix: edit weights to change order complexity distribution.
DEFAULT_MIX: Dict[str, float] = {
    "simple": 0.3,    # e.g., long black, batch brew
    "standard": 0.6,  # e.g., latte, cap, piccolo with a smile
    "slow": 0.1,      # e.g., pour-over, Orange Mocha Frappuccino
}


# -----------------------------------------------------------
# Utility functions
# -----------------------------------------------------------

def normalise_mix(mix: Dict[str, float]) -> Dict[str, float]:
    """
    Ensure the drink mix sums to 1. Handy for when I tweak weights live on stage. ;)
    """
    total = sum(mix.values()) or 1.0
    return {k: v / total for k, v in mix.items()}


def weighted_choice(mix: Dict[str, float]) -> str:
    """
    Pick a drink kind using the weights in the mix.
    Straightforward cumulative sampling.
    """
    r = random.random()
    cum = 0.0
    for k, w in mix.items():
        cum += w
        if r <= cum:
            return k
    return next(iter(mix))  # fallback, shouldn’t happen


def service_time_draw(kind: str) -> float:
    """
    Draw a service time (in minutes) based on drink kind.
      - simple:   mean 0.5
      - standard: mean 2.0
      - slow:     mean 4.0
    """
    if kind == "simple":
        return random.expovariate(1 / 0.5)
    elif kind == "standard":
        return random.expovariate(1 / 2.0)
    else:
        return random.expovariate(1 / 4.0)


def p95(xs: List[float]) -> float:
    """
    Quick 95th percentile: 95% of values are at or below this number.
    Why we like it: shows the “tail risk” -- those “struth, it’s busy” moments.
    """
    xs_sorted = sorted(xs)
    if not xs_sorted:
        return 0.0
    idx = int(0.95 * (len(xs_sorted) - 1))
    return xs_sorted[idx]


def arrival_rate(t: float, P: Params) -> float:
    """
    Time-varying arrivals: faster during peak window if enabled.
    """
    if P.use_peak and (P.peak_start <= t <= P.peak_end):
        return P.peak_arrival_rate
    return P.base_arrival_rate


# -----------------------------------------------------------
# Shared-queue model: one line, one or two baristas
# -----------------------------------------------------------

class Job:
    """
    A coffee order in the shared queue.
    We record arrival and completion to compute waits.
    """
    __slots__ = ("name", "arrival", "drink")

    def __init__(self, name: str, arrival: float, drink: str) -> None:
        self.name = name
        self.arrival = arrival
        self.drink = drink


def barista_server(env: simpy.Environment,
                   name: str,
                   job_queue: simpy.Store,
                   results: dict):
    """
    A barista process that repeatedly:
      1) Pulls the next job from the shared queue.
      2) Waits for service_time to "make the coffee".
      3) Records metrics.

    Interrupt-safe:
    - If interrupted while waiting for the next job, exit immediately.
    - If interrupted mid-service, finish the current coffee, then exit.
    """
    try:
        while True:
            try:
                job: Job = yield job_queue.get()
            except simpy.Interrupt:
                # Interrupted while idle/waiting for a job — just clock off.
                break

            start = env.now
            wait = start - job.arrival
            service_time = service_time_draw(job.drink)

            try:
                # Simulate making the coffee
                yield env.timeout(service_time)
            except simpy.Interrupt:
                # We were told to stop mid-service; finish this coffee then go home.
                # We simulate “finishing” by not aborting the service; since we’re already in timeout,
                # there’s no partial completion to account for. We just stop after this job.
                # To keep accounting sensible, record completion at current time.
                finish = env.now
                results["waits"].append(wait)
                results["drinks"].append(job.drink)
                results["system_times"].append(finish - job.arrival)
                results["waits_over_time"].append((finish, wait))
                results["queue_len"].append((finish, len(job_queue.items)))
                break  # exit after finishing current job

            # Normal completion path
            finish = env.now
            results["waits"].append(wait)
            results["drinks"].append(job.drink)
            results["system_times"].append(finish - job.arrival)
            results["waits_over_time"].append((finish, wait))
            results["queue_len"].append((finish, len(job_queue.items)))
    except simpy.Interrupt:
        # Catch any strays so the process always exits cleanly
        pass


def server_scheduler(env: simpy.Environment,
                     job_queue: simpy.Store,
                     P: Params,
                     results: dict):
    """
    Manage how many baristas are active over time, without mutating capacity.
    - Start one barista at t=0 and keep them running the whole sim.
    - If second_barista_in_peak=True, start a second barista at peak_start and stop them at peak_end.
    """
    # Barista A runs for the whole sim
    env.process(barista_server(env, "Barista A", job_queue, results))

    # If no special peak behaviour, we're done.
    if not (P.use_peak and P.second_barista_in_peak):
        return

    # Wait until the peak starts, then start Barista B
    if env.now < P.peak_start:
        yield env.timeout(P.peak_start - env.now)

    # Launch Barista B for the peak window
    proc_b = env.process(barista_server(env, "Barista B", job_queue, results))

    # Let B work until peak_end, then interrupt to stop gracefully
    remaining = max(0.0, P.peak_end - env.now)
    if remaining > 0:
        yield env.timeout(remaining)

    # Politely stop Barista B.
    # If B is waiting for a job, they’ll exit immediately.
    # If B is in the middle of a job, the server catches the interrupt and exits after finishing.
    proc_b.interrupt()


def arrival_process(env: simpy.Environment,
                    job_queue: simpy.Store,
                    P: Params,
                    mix: Dict[str, float],
                    results: dict):
    """
    Generate customers and put their jobs on the shared queue.
    We also snapshot the queue length after each arrival.
    """
    i = 0
    while env.now < P.minutes:
        i += 1
        lam = arrival_rate(env.now, P)           # arrivals per minute at this moment
        inter = random.expovariate(lam)          # time until next arrival
        yield env.timeout(inter)

        drink = weighted_choice(mix)
        job = Job(name=f"C{i}", arrival=env.now, drink=drink)
        yield job_queue.put(job)

        # Snapshot queue length after arrival
        results["queue_len"].append((env.now, len(job_queue.items)))


# -----------------------------------------------------------
# Run + summarise (single scenario)
# -----------------------------------------------------------

def simulate(P: Params, mix: Dict[str, float]) -> dict:
    """
    Build the SimPy environment, wire up processes, and run the clock.
    Returns all raw results for analysis or plotting.
    """
    # Important: seed the RNG for reproducibility
    random.seed(P.seed)
    env = simpy.Environment()

    # Shared FIFO queue (one register taking orders, one line)
    job_queue = simpy.Store(env)

    results = {
        "waits": [],
        "system_times": [],
        "queue_len": [],
        "waits_over_time": [],
        "drinks": [],
    }

    # Normalise mix for safety, just in case inputs don't sum to 1
    mix = normalise_mix(mix)

    # Start processes
    env.process(server_scheduler(env, job_queue, P, results))
    env.process(arrival_process(env, job_queue, P, mix, results))

    # Run the sim
    env.run(until=P.minutes)
    return results


def summarise(results: dict) -> dict:
    """
    Turn raw arrays into a quick summary:
    - served: how many customers got their coffee
    - avg/median/p95 wait
    - max queue length observed
    """
    waits = results["waits"]
    summary = {
        "served": len(waits),
        "avg_wait": statistics.mean(waits) if waits else 0.0,
        "med_wait": statistics.median(waits) if waits else 0.0,
        "p95_wait": p95(waits) if waits else 0.0,
        "max_queue": max((q for _, q in results["queue_len"]), default=0),
    }
    return summary


def report(P: Params, summary: dict, mix: Dict[str, float]):
    """
    Print the scenario and results (handy for debugging or CLI use).
    """
    print("---- Scenario ----")
    print(f"minutes={P.minutes} seed={P.seed} peak={P.use_peak} 2nd_barista_in_peak={P.second_barista_in_peak}")
    print(f"rates: base={P.base_arrival_rate:.3f}/min peak={P.peak_arrival_rate:.3f}/min "
          f"window=[{P.peak_start},{P.peak_end}]")
    print(f"mix: {normalise_mix(mix)}")
    print("---- Results ----")
    print(f"Served: {summary['served']}")
    print(f"Avg wait: {summary['avg_wait']:.2f} min")
    print(f"Median wait: {summary['med_wait']:.2f} min")
    print(f"95th wait: {summary['p95_wait']:.2f} min")
    print(f"Max queue: {summary['max_queue']}")

# -----------------------------------------------------------
# Extensions: Weekday presets and multi-seed runner
# -----------------------------------------------------------

@dataclass(frozen=True)
class DayPreset:
    """
    A small wrapper to tweak the base Params/mix per weekday.
    - base_rate_mult / peak_rate_mult: multiply arrivals for that day.
      e.g., Friday fewer people -> 0.75. Wednesday busier -> 1.25.
    - mix_delta: add/subtract to the drink mix weights, then renormalise.
      e.g., cold Monday -> more 'simple' hot drinks.
    - use_peak / second_barista_in_peak: optional per-day overrides.
    """
    name: str
    base_rate_mult: float = 1.0
    peak_rate_mult: float = 1.0
    mix_delta: Optional[Dict[str, float]] = None
    use_peak: Optional[bool] = None
    second_barista_in_peak: Optional[bool] = None

def apply_mix_delta(base_mix: Dict[str, float], delta: Optional[Dict[str, float]]) -> Dict[str, float]:
    """
    Combine base mix with an additive delta and renormalise.
    Negative values are clipped at zero to avoid weirdness.
    """
    if not delta:
        return normalise_mix(base_mix)
    mixed = {k: base_mix.get(k, 0.0) + delta.get(k, 0.0) for k in set(base_mix) | set(delta)}
    for k in mixed:
        if mixed[k] < 0:
            mixed[k] = 0.0
    return normalise_mix(mixed)

def preset_params_for_day(base_P: Params, day: DayPreset) -> Params:
    """
    Create a per-day Params by applying rate multipliers and optional toggles.
    Note: we keep the seed here as a placeholder; the runner sets a unique seed per run.
    """
    new_base_rate = base_P.base_arrival_rate * day.base_rate_mult
    new_peak_rate = base_P.peak_arrival_rate * day.peak_rate_mult
    return Params(
        minutes=base_P.minutes,
        seed=base_P.seed,  # to be overridden per run
        base_arrival_rate=new_base_rate,
        peak_arrival_rate=new_peak_rate,
        peak_start=base_P.peak_start,
        peak_end=base_P.peak_end,
        use_peak=base_P.use_peak if day.use_peak is None else day.use_peak,
        second_barista_in_peak=(
            base_P.second_barista_in_peak if day.second_barista_in_peak is None else day.second_barista_in_peak
        ),
        plot=False,
    )

def run_day_multi_seed(day: DayPreset,
                       base_P: Params,
                       base_mix: Dict[str, float],
                       seeds: List[int]) -> Dict:
    """
    Run the same 'day' multiple times with different random seeds.
    Why? To see variability and get better averages.
    Returns an aggregate object with per-seed results and per-day summaries.
    """
    P_day_template = preset_params_for_day(base_P, day)
    mix_day = apply_mix_delta(base_mix, day.mix_delta)

    runs = []
    queue_ts_all: List[List[Tuple[float, int]]] = []  # list of time series, one per seed

    for s in seeds:
        # Each seed gets its own Params object
        P_run = Params(
            minutes=P_day_template.minutes,
            seed=s,
            base_arrival_rate=P_day_template.base_arrival_rate,
            peak_arrival_rate=P_day_template.peak_arrival_rate,
            peak_start=P_day_template.peak_start,
            peak_end=P_day_template.peak_end,
            use_peak=P_day_template.use_peak,
            second_barista_in_peak=P_day_template.second_barista_in_peak,
            plot=False,
        )
        res = simulate(P_run, mix_day)
        summ = summarise(res)
        runs.append({"seed": s, "summary": summ, "results": res})
        queue_ts_all.append(res["queue_len"])

    # Aggregate some handy cross-seed stats
    avg_waits = [r["summary"]["avg_wait"] for r in runs]
    med_waits = [r["summary"]["med_wait"] for r in runs]
    p95_waits = [r["summary"]["p95_wait"] for r in runs]
    served = [r["summary"]["served"] for r in runs]

    agg = {
        "day": day.name,
        "mix": mix_day,
        "avg_wait_mean": statistics.mean(avg_waits) if avg_waits else 0.0,
        "avg_wait_p95": p95(avg_waits) if avg_waits else 0.0,
        "avg_waits": avg_waits,                 # keep raw values for boxplot
        "med_wait_mean": statistics.mean(med_waits) if med_waits else 0.0,
        "p95_wait_mean": statistics.mean(p95_waits) if p95_waits else 0.0,
        "served_mean": statistics.mean(served) if served else 0.0,
        "served": served,
        "runs": runs,                           # full per-seed payload
        "queue_ts_all": queue_ts_all,           # time series per seed
        "params": P_day_template,               # the day-adjusted Params
    }
    return agg



# -----------------------------------------------------------
# Visualisation helpers
# -----------------------------------------------------------

def frange(start: float, stop: float, step: float):
    """
    float-range helper for regular sampling (e.g., every minute).
    """
    t = start
    while t <= stop + 1e-9:
        yield round(t, 6)
        t += step


def align_queue_series(series_list: List[List[Tuple[float, int]]],
                       max_time: float,
                       dt: float = 1.0) -> Tuple[List[float], List[List[float]]]:
    """
    Our queue length snapshots occur at irregular times (events).
    To plot tidy lines, we resample onto a regular grid:
      times = 0, dt, 2dt, ...
    For each time, we take the most recent observed queue length.
    """
    times = [t for t in frange(0.0, max_time, dt)]
    values_per_seed = []

    for series in series_list:
        series_sorted = sorted(series, key=lambda x: x[0])
        q = 0
        j = 0
        vals = []
        for t in times:
            # Move forward through the event list up to the current time
            while j < len(series_sorted) and series_sorted[j][0] <= t:
                q = series_sorted[j][1]
                j += 1
            vals.append(q)
        values_per_seed.append(vals)

    return times, values_per_seed


def plot_across_days(day_aggs: List[Dict], minutes: int):
    """
    Two side-by-side plots:
    - Left: Boxplots of average wait by day (shows variability across seeds).
    - Right: Mean queue length over time by day (averaged across seeds).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    # Boxplot of average waits
    day_names = [d["day"] for d in day_aggs]
    avg_waits_per_day = [d["avg_waits"] for d in day_aggs]
    axes[0].boxplot(avg_waits_per_day, labels=day_names, showmeans=True)
    axes[0].set_title("Average wait by day (across seeds)")
    axes[0].set_ylabel("Minutes")

    # Mean queue length over time
    for d in day_aggs:
        times, vals_per_seed = align_queue_series(d["queue_ts_all"], minutes, dt=1.0)
        if not vals_per_seed:
            continue
        # Average across seeds at each time point
        mean_q = [statistics.mean(xs) for xs in zip(*vals_per_seed)]
        axes[1].plot(times, mean_q, label=d["day"])

    axes[1].set_title("Mean queue length over time")
    axes[1].set_xlabel("Minutes since open")
    axes[1].set_ylabel("Queue length")
    axes[1].legend()

    plt.show()

# -----------------------------------------------------------
# Weekday presets (feel free to tweak the story)
# -----------------------------------------------------------

WEEKDAY_PRESETS: List[DayPreset] = [
    # Monday: cold weather → push towards simpler hot drinks
    DayPreset(
        name="Mon",
        base_rate_mult=1.0,
        peak_rate_mult=1.0,
        mix_delta={"simple": +0.10, "standard": -0.08, "slow": -0.02},
        use_peak=True,
        second_barista_in_peak=True,
    ),
    # Tuesday: similar to Monday but a touch lighter bias
    DayPreset(
        name="Tue",
        base_rate_mult=1.0,
        peak_rate_mult=1.0,
        mix_delta={"simple": +0.08, "standard": -0.06, "slow": -0.02},
        use_peak=True,
        second_barista_in_peak=True,
    ),
    # Wednesday: more people in the city → higher arrival rates
    DayPreset(
        name="Wed",
        base_rate_mult=1.25,
        peak_rate_mult=1.25,
        mix_delta=None,
        use_peak=True,
        second_barista_in_peak=True,
    ),
    # Thursday: baseline day
    DayPreset(
        name="Thu",
        base_rate_mult=1.0,
        peak_rate_mult=1.0,
        mix_delta=None,
        use_peak=True,
        second_barista_in_peak=True,
    ),
    # Friday: more WFH → fewer arrivals
    DayPreset(
        name="Fri",
        base_rate_mult=0.75,
        peak_rate_mult=0.75,
        mix_delta=None,
        use_peak=True,
        second_barista_in_peak=True,
    ),
]



# -----------------------------------------------------------
# Main: run a whole "week" across many seeds
# -----------------------------------------------------------

if __name__ == "__main__":
    # Base scenario shared by all days (the presets will tweak it)
    base_P = Params(
        minutes=60,                 # Try 90 or 120 to stretch the morning
        seed=42,                    # Just a placeholder; each run sets its own seed
        base_arrival_rate=1/2,      # Off-peak arrivals per minute
        peak_arrival_rate=1/0.5,    # Faster arrivals in peak
        peak_start=30.0,            # Start of rush
        peak_end=45.0,              # End of rush
        use_peak=True,              # Peak ON by default
        second_barista_in_peak=True,# Bring in a second barista during the rush
        plot=False,
    )

    # Base drink mix (per your original)
    base_mix = {
        "simple": 0.3,
        "standard": 0.6,
        "slow": 0.1,
    }

    # Multi-seed controls:
    # - start_seed: choose where the seed sequence begins (e.g., 42)
    # - num_seeds: how many seeds to run (e.g., 50 → seeds 42..91)
    start_seed = 42
    num_seeds = 50
    seeds = list(range(start_seed, start_seed + num_seeds))

    # Run all weekdays with the same seed list
    day_aggs: List[Dict] = []
    for day in WEEKDAY_PRESETS:
        agg = run_day_multi_seed(day, base_P, base_mix, seeds)
        day_aggs.append(agg)

    # Print a compact weekly summary
    print("=== Weekly summary across seeds ===")
    print(f"Seeds: {seeds[0]}..{seeds[-1]} (n={len(seeds)})")
    for d in day_aggs:
        print(f"{d['day']}: served_mean={d['served_mean']:.1f}, "
              f"avg_wait_mean={d['avg_wait_mean']:.2f} min, "
              f"avg_wait_p95={d['avg_wait_p95']:.2f} min, "
              f"mix={d['mix']}")

    # Visualise the results across days
    plot_across_days(day_aggs, minutes=base_P.minutes)

    # OPTIONAL: also show a representative single-seed timeline for each day using visual.py
    # Comment this block out if you don't want per-day individual run plots.
    try:
        from visual import plot_queue_and_waits

        for d in day_aggs:
            runs = d["runs"]
            if not runs:
                continue
            # Choose the run whose average wait is closest to the day's mean (a "typical" seed)
            target = min(runs, key=lambda r: abs(r["summary"]["avg_wait"] - d["avg_wait_mean"]))

            title = f"{d['day']} representative (seed={target['seed']})"
            plot_queue_and_waits(
                target["results"]["queue_len"],
                target["results"]["waits_over_time"],
                title=title,
                figsize=(9, 6),
                show=True,
                save_path=None,  # e.g., f"{d['day']}_rep.png"
            )
    except ImportError:
        # visual.py or matplotlib not available; skip these per-day plots gracefully.
        pass