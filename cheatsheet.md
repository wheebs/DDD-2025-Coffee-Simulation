# SimPy Cheatsheet (DDD Perth 2025)

A quick, practical guide to modelling queues and resources with SimPy. Use this as a companion to the coffee shop demos.

---

## Core ideas

- **Discrete‑event simulation**: jump from event to event (arrivals, service starts/finishes) instead of ticking every second.  
- **Environment**: the simulation clock and scheduler (`simpy.Environment`).  
- **Processes**: Python generator functions that yield events (e.g., timeouts, resource requests).  
- **Resources**: capacity constraints (servers, machines, baristas).  
- **Randomness**: use distributions (e.g., exponential) to model interarrival and service times.  
- **Replications (seeds)**: run multiple times with different seeds to see variability and reduce fluke results.  

---

## Minimal building blocks

**Create an environment:**

```python
env = simpy.Environment()
```

**Define a process (a generator that yields events):**

```python
def customer(env):
    yield env.timeout(1.0)  # do something for 1 minute
```

**Start processes:**

```python
env.process(customer(env))
```

**Advance time:**

```python
yield env.timeout(3.5)  # advance 3.5 minutes
```

**Run the simulation:**

```python
env.run(until=60)  # run until t = 60 minutes
```

---

## Resources (queues and servers)

**One server:**

```python
barista = simpy.Resource(env, capacity=1)
```

**Request/hold/release pattern (managed by a context manager):**

```python
with barista.request() as req:
    yield req              # wait in queue until you get the resource
    yield env.timeout(s)   # service takes s minutes
```

**Shared FIFO queue (alternative to Resource):**

```python
queue = simpy.Store(env)
yield queue.put(job)       # producer adds jobs
job = yield queue.get()    # consumer pulls next job
```

**When to use which?**  

- `Resource`: simpler single‑server or limited‑capacity server semantics.  
- `Store`: explicit queue objects you can share across multiple servers, or where you want direct access to queue length/items.  

---

## Time‑varying arrivals

**Constant (Poisson) arrivals:**

```python
inter = random.expovariate(lam)  # lam = arrivals per minute
yield env.timeout(inter)
```

**Time‑varying rate (piecewise):**

```python
lam = peak_rate if peak_start <= env.now <= peak_end else base_rate
inter = random.expovariate(lam)
yield env.timeout(inter)
```

---

## Useful stats

**Average/median waits:**

```python
import statistics as stats
avg = stats.mean(waits) if waits else 0.0
med = stats.median(waits) if waits else 0.0
```

**Quick p95:**

```python
def p95(xs):
    xs = sorted(xs)
    return xs[int(0.95 * (len(xs) - 1))] if xs else 0.0
```

**Max queue length from snapshots:**

```python
max_q = max((q for _, q in queue_len), default=0)
```

---

## Structuring a small model

- Keep parameters together (dataclass works well).  
- Separate “arrival_process”, “server(s)”, and “run/summarise” functions.  
- Record metrics as you go: waits, system times, queue length snapshots.  
- Seed the RNG for reproducibility:  

```python
random.seed(42)
```

---

## Multi‑seed runs (why and how)

**Why:**  

- Simulations are stochastic. One run can be lucky/unlucky.  
- Multiple seeds show variability and give more robust averages. 
If you wanted to have a more robust number of seeds you should look to probabilistic hypothesis testing to
discern if 40 seeds or 100 seeds is the right number of seeds to get a consistent answer.
If you would like to chat about this stuff (it really excites me), please reach out. My contact details
are in the README.md 

**How:**

```python
seeds = range(42, 92)  # 50 seeds
for s in seeds:
    random.seed(s)
    run_simulate(...)
```

**In this repo:**  

- `coffee_shop_advanced.py` has `run_day_multi_seed` and cross‑day plots.  

---

## Plotting (quick visual workflow)

**Use matplotlib for simple timelines:**

```python
plt.step(times, q, where="post")  # queue length
plt.plot(t, waits, ".")           # dots for individual waits
```

**Or use the helper:**

```python
from visual import plot_queue_and_waits
plot_queue_and_waits(queue_len, waits_over_time, title="My run")
```

---

## Common tweaks to try

- Increase `ARRIVAL_RATE` to create congestion and watch waits grow.  
- Add a second barista during the peak and compare waits/queues.  
- Change the drink mix to skew towards “slow” drinks and see the tail (p95) rise.  
- Extend minutes to see steady‑state vs transient effects.  

---

## Gotchas

- Don’t commit `__pycache__/` or `*.pyc` — use a `.gitignore`.  
- If nothing seems to happen, you might have forgotten a `yield`.  
- If plots don’t show, check that matplotlib is installed and your backend supports showing windows (or save to file).  
- For exponential draws, ensure you pass the **rate** to `expovariate` (λ per minute), or invert a mean appropriately.  

---

## Handy references

- SimPy docs: <https://simpy.readthedocs.io/>  
- GitHub Python .gitignore: <https://github.com/github/gitignore/blob/main/Python.gitignore>  

---

**Happy simming — and cheers for coming to DDD Perth!**
