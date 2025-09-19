# DDD Perth 2025 — SimPy Coffee Shop Demos

G’day! If you’ve landed here, you probably came from my DDD Perth 2025 talk. This repo is a friendly nudge to give SimPy a go — today. SimPy is a small, powerful library for discrete‑event simulation in Python. You can model queues, servers, and variability with just a few lines.

I’d love you to start using SimPy — especially if you haven’t before. This repo gives you two coffee‑shop simulations you can run and tweak, these are the simulations I used during my DDD 2025 talk.

## Contents

- `coffee_shop_minimal.py` — single‑barista model with event logs and a simple visual.
- `coffee_shop_advanced.py` — shared‑queue model with drink mix, optional peak window, weekday presets, multi‑seed runs, and cross‑day visuals.
- `visual.py` — plotting helpers.
- `requirements.txt` — pinned dependencies.

## Short Description

**Minimal:** simulate one barista with exponential arrivals and service. Start with a verbose event log (Stage 1), then flip to concise KPIs + a quick visual (Stage 2).

**Advanced:** one shared queue, drink mix (simple/standard/slow), optional peak window with a second barista during the rush, weekday presets (Mon–Fri), multiple random seeds to show variability, and comparison plots.

**Guiding star:** SimPy is free, easy, and you can do it today.

## How to Run

### Option A: Quick Start (no plotting)

1. Install Python 3.10+.
2. Create a virtual environment (recommended):

    ```bash
    python -m venv .venv
    # macOS/Linux:
    source .venv/bin/activate
    # Windows (PowerShell):
    .venv\\Scripts\\Activate.ps1
    ```

3. Install requirements:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the minimal model, Stage 1 (event log):

    - In `coffee_shop_minimal.py`, at the bottom ensure:
      ```python
      run(plot=False, verbose=True)
      ```
    - Then run:
      ```bash
      python coffee_shop_minimal.py
      ```

    - You’ll see arrivals, service starts, and finishes printed to the console.

### Option B: Minimal Model with Visuals (Stage 2)

1. In `coffee_shop_minimal.py`:
    - Uncomment:
      ```python
      from visual import plot_queue_and_waits
      ```
    - Comment the Stage 1 line and uncomment:
      ```python
      run(plot=True, verbose=False)
      ```

2. Run:
    ```bash
    python coffee_shop_minimal.py
    ```

3. A simple two‑panel plot will show queue length and wait times.

### Option C: Advanced Model — Weekdays and Multi‑Seed Comparison

1. Ensure matplotlib is installed (it’s in `requirements.txt`).

2. Run:
    ```bash
    python coffee_shop_advanced.py
    ```

3. You’ll get:
    - A printed weekly summary across seeds (served, average waits, p95).
    - A figure comparing average waits across days and mean queue profiles.

4. To tweak the story:
    - Edit `Params` in `main` for minutes, arrival rates, peak window, and second barista.
    - Adjust `WEEKDAY_PRESETS` to change arrival multipliers and drink mix.
    - Change `start_seed` and `num_seeds` for more or fewer replications.

## Pro Tips to Try Live

- In `coffee_shop_minimal.py`, nudge `ARRIVAL_RATE` and `SERVICE_MEAN` to show congestion tipping points.
- In `coffee_shop_advanced.py`, toggle `use_peak` or `second_barista_in_peak` to show staffing impact.

## Requirements

- **Python 3.10+ recommended**
- Dependencies (pinned in `requirements.txt`):
    - `simpy==4.1.1`
    - `numpy==2.0.1`
    - `matplotlib==3.9.0`
    - `pandas==2.2.2`

Install with:

```bash
pip install -r requirements.txt
```

## Contact

Questions, ideas, or wins from your own simulations? Reach out:

Name: David Whebell

Email: dmwhebell@gmail.com

Links: https://au.linkedin.com/in/david-whebell


If this helped you get started with SimPy, I’d love to hear about it. Cheers!
