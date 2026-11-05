# AI-Driven Heterogeneous Resource Scheduling for Baseband

A small Python prototype for assigning time-critical baseband processing workloads to heterogeneous compute resources. It combines runtime feedback with a deterministic policy agent that selects a latency- or energy-oriented scheduling objective.

> **What this project schedules:** compute tasks such as channel estimation, beamforming, and decoding across CPUs, DSPs, FPGAs, GPUs, or dedicated accelerators. It is not a 3GPP MAC scheduler and does not allocate radio resource blocks to UEs. This is a scheduling simulation/control-plane foundation, not a production real-time executor.

## What is implemented

- **Deadline-aware placement:** tasks are admitted only when an estimated completion fits the absolute deadline.
- **Heterogeneous profiles:** each accelerator supports explicit workload kinds with its own runtime and power estimate.
- **Runtime adaptation:** execution feedback updates per-device, per-workload runtime estimates using EWMA.
- **Explainable policy agent:** `SchedulingAgent` selects a per-batch latency, balanced, or energy objective based on estimated deadline slack, and returns the selected mode and rationale.
- **Hard safety checks:** the scheduler remains authoritative for device compatibility, serial execution, deadline feasibility, and the aggregate active-power limit. The agent's objective changes candidate ranking only; it cannot bypass these checks.
- **Explicit outcomes:** the plan reports assignments and rejection reasons.

The agent is a deterministic local policy controller, **not an LLM** and not a trained neural-network policy. It estimates the tightest deadline-slack ratio in the submitted batch as `(deadline_us - now_us) / fastest_estimated_duration_us`. By default, ratios at or below `1.5` select latency mode (energy weight `0`), ratios at or above `4.0` select energy mode (energy weight `10`), and intermediate ratios retain the scheduler's configured default. If no submitted workload has a compatible profile, it also retains the default. These thresholds are policy defaults, not learned or benchmark-derived values.

## Repository structure

```text
baseband_scheduler/
  __init__.py
  __main__.py   # package simulation entry point
  agent.py      # explainable policy selection
  scheduler.py  # EDF admission, placement, reservations, and EWMA feedback
tests/
  test_agent.py
  test_scheduler.py
scheduler.py    # compatibility entry point for earlier imports and simulation
```

## Usage

```python
from baseband_scheduler import (
    Accelerator,
    AdaptiveScheduler,
    Performance,
    SchedulingAgent,
    Workload,
)

scheduler = AdaptiveScheduler(
    [
        Accelerator("gpu", {"beamforming": Performance(80, 120)}),
        Accelerator("dsp", {"beamforming": Performance(140, 25)}),
    ],
    power_budget_w=150,
)
agent = SchedulingAgent(scheduler)

result = agent.plan(
    [Workload("beam-1", "beamforming", work=1, deadline_us=200)],
    now_us=0,
)
print(result.mode, result.rationale)
for assignment in result.schedule.assignments:
    print(assignment.accelerator, assignment.start_us, assignment.finish_us)
for rejection in result.schedule.rejected:
    print(rejection.workload.task_id, rejection.reason)

# Feed measured execution time back into the runtime estimator.
scheduler.observe("gpu", "beamforming", work=1, duration_us=90)
```

The scheduler can also be used directly, without the policy agent:

```python
from baseband_scheduler import AdaptiveScheduler, Workload

plan = scheduler.schedule([Workload("beam-2", "beamforming", 1, 200)])
```

Run the illustrative simulation and standard-library test suite from the repository root (Python 3.10+; no installation required):

```sh
python -m baseband_scheduler
python -m unittest discover -s tests -v
```

The root-level `scheduler.py` remains as a compatibility shim for existing code using `from scheduler import ...` and for `python scheduler.py`.

## Scheduling model and boundaries

Workload deadlines are absolute timestamps in the caller's monotonic microsecond clock. Work units must be consistent within a workload kind. The scheduler processes each submitted batch in earliest-deadline-first order, with task ID as a deterministic tie-breaker. It estimates candidate duration from per-device runtime profiles, checks device serialization, deadlines, and active-power budget, then ranks only feasible candidates using completion time plus a normalized energy penalty.

Accepted reservations persist across calls until their predicted finish time. `observe()` updates the EWMA runtime estimate for future plans; it does not revise an existing reservation. Calls to one scheduler instance must be serialized by the caller.

The prototype models non-preemptible tasks, one task at a time per accelerator, simple workload profiles, and a single aggregate active-power limit. It does not model task dependencies, queues beyond reservations, preemption, memory/bandwidth, data movement, device health, simultaneous kernels, idle power, thermal limits, or dispatch overhead unless reflected in supplied profiles. It provides no hard real-time guarantee: profile error, workload drift, and overruns can invalidate predicted feasibility. A production integration needs actual device dispatch, completion reporting, watchdog/overrun handling, clock-domain discipline, and an admission/fallback policy.

The supplied accelerator profiles and simulation values are illustrative, not benchmark results or hardware guarantees. The runtime estimator is in-memory EWMA state. There is no model registry, durable telemetry store, training job, external LLM, deployment service, or automated rollback.

## Future production considerations

For production use, calibrate profiles by hardware, software/firmware, and workload configuration; validate telemetry before updating estimates; monitor predicted-versus-measured duration, deadline misses, rejection rates, utilization, and power; and define failure behavior for degraded devices and overruns. Any future learned policy should be evaluated on representative traces and retain deterministic compatibility, deadline, and power checks outside the model, with replay/shadow evaluation, constrained rollout, and rollback.
