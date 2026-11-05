# AI-Driven Heterogeneous Resource Scheduling for Baseband

A small Python prototype for assigning time-critical baseband processing workloads to heterogeneous compute resources. It combines runtime feedback with a deterministic policy agent that selects a latency- or energy-oriented scheduling objective.

> **What this project schedules:** compute tasks such as channel estimation, beamforming, and decoding across CPUs, DSPs, FPGAs, GPUs, or dedicated accelerators. It is not a 3GPP MAC scheduler and does not allocate radio resource blocks to UEs. This is a scheduling simulation/control-plane foundation, not a production real-time executor.

## Bottlenecks in traditional scheduling approaches

Modern radio access networks have scheduling decisions at more than one level. A radio (UE) scheduler decides which user equipment (UE) can use radio resources in each transmission time interval (TTI), balancing throughput, fairness, quality of service (QoS), buffer backlog, and retransmission state such as HARQ ([3GPP TS 38.300](https://www.3gpp.org/DynaReport/38300.htm), [TS 38.321](https://www.3gpp.org/DynaReport/38321.htm)). Traditional radio schedulers often use handcrafted heuristics such as Proportional Fair (PF), Round Robin (RR), or weighted PF. Those approaches address radio-resource allocation; they are not the same problem as assigning baseband compute tasks to processors, and this repository does **not** implement a UE or resource-block scheduler.

The broader RAN environment also carries different service profiles at the same time:

- **eMBB (Enhanced Mobile Broadband):** emphasizes high throughput and large data volumes.
- **URLLC (Ultra-Reliable Low-Latency Communications):** emphasizes very low latency and high reliability.
- **mMTC (Massive Machine-Type Communications):** involves many devices sending small, often sporadic messages.

These are broad service-category descriptions, not per-task timing guarantees ([ITU-R M.2083](https://www.itu.int/rec/R-REC-M.2083-0-201509-I/en), [3GPP TS 22.261](https://www.3gpp.org/DynaReport/22261.htm)).

That changing traffic and radio environment also changes the mix and urgency of baseband Layer 1 (L1) work, including channel estimation, massive-MIMO beamforming, and decoding. These are examples of PHY processing functions; applicable NR physical-layer specifications include [3GPP TS 38.211](https://www.3gpp.org/DynaReport/38211.htm), [TS 38.212](https://www.3gpp.org/DynaReport/38212.htm), and [TS 38.214](https://www.3gpp.org/DynaReport/38214.htm). The exact algorithms and mapping of these operations onto CPUs, GPUs, DSPs, FPGAs, or ASICs depend on the implementation; standards compliance does not prescribe this repository's compute placement policy. Heterogeneous baseband task mapping is an implementation-level research topic, for example [NEXUS](https://arxiv.org/abs/2509.04625). At the compute-scheduling layer that this project models, important challenges include:

- **Static heuristics:** a fixed device choice or fixed placement rule may not adapt well when workload size, configuration, or execution conditions change.
- **Hardware underutilization:** without workload-aware placement, some compatible CPUs, GPUs, DSPs, FPGAs, or ASICs can be idle while another resource is overloaded.
- **Deadline and service-level objective misses:** a task that finishes too late can contribute to missed processing deadlines and, in an integrated pipeline, may ultimately affect packet delivery or service objectives.
- **Power constraints:** selecting a fast but power-hungry device without accounting for concurrent work can exceed the platform's intended power budget.

These are motivating challenges, not measured results from this repository. The prototype addresses a limited part of the compute-placement problem with runtime profiles, deadline and power feasibility checks, and an adaptive policy objective. It does not claim to solve radio-resource allocation, guarantee service-level agreements, or demonstrate performance improvements on real hardware.

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

Workload deadlines are absolute timestamps in the caller's monotonic microsecond clock. They represent task-level deadlines supplied by the integrating application; they are **not automatically equal to a service-level packet delay budget (PDB), a UE scheduling/grant window, or a HARQ timing value**. A PDB is a QoS characteristic for packet delay across a defined network portion ([3GPP TS 23.501, section 5.7.4](https://www.3gpp.org/DynaReport/23501.htm)). An integration must derive each compute-task deadline from the actual processing stage, radio timeline, and remaining end-to-end budget; this prototype does not derive those deadlines. Work units must be consistent within a workload kind. The scheduler processes each submitted batch in earliest-deadline-first order, with task ID as a deterministic tie-breaker. It estimates candidate duration from per-device runtime profiles, checks device serialization, deadlines, and active-power budget, then ranks only feasible candidates using completion time plus a normalized energy penalty.

Accepted reservations persist across calls until their predicted finish time. `observe()` updates the EWMA runtime estimate for future plans; it does not revise an existing reservation. Calls to one scheduler instance must be serialized by the caller.

The prototype models non-preemptible tasks, one task at a time per accelerator, simple workload profiles, and a single aggregate active-power limit. It does not model task dependencies, queues beyond reservations, preemption, memory/bandwidth, data movement, device health, simultaneous kernels, idle power, thermal limits, or dispatch overhead unless reflected in supplied profiles. It provides no hard real-time guarantee: profile error, workload drift, and overruns can invalidate predicted feasibility. A production integration needs actual device dispatch, completion reporting, watchdog/overrun handling, clock-domain discipline, and an admission/fallback policy.

The supplied accelerator profiles and simulation values are illustrative, not benchmark results or hardware guarantees. The runtime estimator is in-memory EWMA state. There is no model registry, durable telemetry store, training job, external LLM, deployment service, or automated rollback.

## Future production considerations

This section describes work that would be needed before connecting this prototype to a real baseband system. **These capabilities are not implemented in this repository.** The current code plans tasks from supplied profiles and simulated/runtime feedback; it does not dispatch work to real hardware, collect platform telemetry, or guarantee deadlines.

### 1. Measure and calibrate device profiles

The scheduler needs a realistic estimate of how long each type of task takes on each compatible accelerator. The example profiles in this project are illustrative only. For a real platform, measure representative tasks on the target hardware and software stack, then use those measurements to populate `Performance(time_per_unit_us, power_w)`.

- Measure the actual workload configurations that matter (for example, relevant input sizes, antenna counts, decoder modes, or batch sizes). A single profile for a broad task name may hide large differences in execution cost.
- Keep work units consistent for a workload kind. If one profile defines a unit as one frame, every matching task and measurement must use that same definition; do not compare a frame-based profile with a sample-based workload.
- Record which hardware model, firmware, driver, compiler, and application version produced each measurement. Recalibrate when those change, because performance may change too.
- Measure execution duration separately from time spent waiting in a queue. The scheduler's `observe()` method expects task execution duration divided by task work; feeding it queue delay would teach it the wrong device speed.
- Measure power using a clearly defined method and scope. This prototype checks the sum of active profile power against one configured budget; it does not include idle/platform power or model detailed power fluctuations.
- Repeat measurements under representative operating conditions, and use conservative profile values if occasional slow executions could cause a deadline miss. An average measurement alone may not represent tail latency.

For example, if a workload has `work=4` and measured device execution takes `200` microseconds, the observation is `duration_us=200`; the measured runtime per work unit is `50` microseconds. The current EWMA then blends that sample with the prior estimate. This update changes later estimates; it does not alter tasks that have already been assigned.

### 2. Validate measurements before learning from them

Runtime feedback affects future placement decisions, so it should be checked before passing it to `observe()`.

- Confirm that the reported task really completed on the named accelerator and that the workload kind, work amount, and profile version match the task that ran.
- Reject missing, non-finite, zero, or negative work and duration values. The current method validates positive finite work and duration, but a production collector should also validate identity, provenance, units, and freshness.
- Detect duplicate reports and late or stale telemetry. A completion should update the model at most once and should be associated with the right task and configuration.
- Keep execution time distinct from queue wait, data transfer, dispatch delay, and completion-reporting delay. If production scheduling must account for those costs, measure and model them explicitly rather than accidentally mixing them into device execution time.
- Track estimate quality. If observed values differ sharply from predictions, flag the profile or device for investigation rather than allowing a bad sample stream to silently distort decisions.

The repository currently has no telemetry collector, durable history, sample confidence tracking, or automatic outlier policy; these checks would belong in the integrating service or a future dedicated telemetry layer.

### 3. Monitor outcomes, not just average speed

A production system should record enough information to tell whether scheduling is meeting its goals and where failures occur. Useful measurements include:

- **Prediction error:** compare predicted runtime with measured execution time for each device and workload profile. Break results down by configuration and report tail errors, not only the average.
- **Deadline results:** count completed tasks that met their deadline, tasks that missed it, and tasks rejected before dispatch. Include rates and trends over time.
- **Decision time:** measure how long scheduling itself takes, separately from task execution. A fast workload is not useful if its scheduling decision arrives too late.
- **Device utilization:** measure how busy each accelerator is, how often it is idle, and whether work is distributed as intended.
- **Power:** compare the scheduler's modeled active-power reservations with platform power telemetry, and count any measured or estimated budget violations. The prototype's profile-based reservation is not a physical power guarantee.
- **Operational health:** count device errors, task overruns, invalid or late measurements, and fallback/retry actions.

These metrics help answer practical questions: Are tasks missing deadlines because estimates are too optimistic? Is one accelerator overloaded? Are power readings inconsistent with the configured profiles? The prototype does not collect or publish these metrics itself.

### 4. Define what happens when predictions or devices fail

Predicted finish times are estimates. A real device can run slowly, report an error, or become unavailable after a task was assigned. Decide the response before deployment rather than assuming the scheduler's plan is a guarantee.

- **Task overrun:** detect when actual execution exceeds its reservation or approaches its deadline. Specify whether to wait, cancel if safe, reduce work, alert, or mark dependent work as failed. The current scheduler does not extend or revise an active reservation after an overrun.
- **Device degradation or loss:** define how health is reported and how an unhealthy accelerator is removed or derated for future plans. Also define what happens to work already dispatched to that device.
- **No feasible assignment:** choose whether the caller rejects, retries, routes to a fallback, degrades the task, or drops it. Do not retry blindly after its absolute deadline has expired.
- **Power pressure:** define whether new work is rejected or delayed when the power budget leaves no feasible slot. The prototype reports a rejection; it does not decide the application's recovery behavior.
- **Clock consistency:** use one monotonic clock domain for `now_us` and absolute deadlines. If components use different clocks, translate timestamps carefully; wall-clock adjustments must not make deadlines move backward or forward unexpectedly.
- **Concurrency and ownership:** serialize calls to one scheduler instance, as required by the current implementation, or add synchronization in a service wrapper. Define which component owns admission and task lifecycle state.

The right response depends on the baseband pipeline's reliability and timing requirements; this project does not prescribe one universal fallback policy.

### 5. Introduce future learned policies cautiously

The current `SchedulingAgent` is a deterministic threshold policy, not a trained model. If a learned policy is added later, treat its output as a suggestion for ranking choices—not permission to violate platform constraints.

1. **Start with recorded-trace evaluation.** Replay representative past workloads and compare the candidate policy with the current deterministic policy. Measure deadline misses, tail latency, rejection rate, utilization, and power, not only average throughput.
2. **Run in shadow mode.** Let the candidate produce choices without controlling actual dispatch. Compare its proposed choices with the live scheduler and check whether it would have improved results using the same trace and constraints.
3. **Use a limited canary.** If shadow results are acceptable, enable the candidate for a small, controlled portion of workloads or devices, with explicit monitoring and a predefined stop condition.
4. **Keep hard checks outside the model.** Validate workload inputs and device compatibility, and enforce deadline, serialization, and power constraints in trusted scheduling code after any model proposal. An invalid or infeasible model suggestion must never be dispatched.
5. **Keep a tested fallback and rollback path.** Preserve a known safe deterministic policy and a way for operators to switch back quickly if deadline, power, or error metrics regress.
6. **Version the model and its inputs.** Record which model, profiles, software, and policy configuration produced each decision so that results can be reproduced and problematic releases can be identified.

This repository currently has no model training, model registry, policy deployment, shadow mode, canary rollout, or rollback implementation. These are future integration and operational responsibilities, not features of the example agent.

### Technical references

- [ITU-R Recommendation M.2083](https://www.itu.int/rec/R-REC-M.2083-0-201509-I/en): IMT-2020 vision and broad usage scenarios.
- [3GPP TS 22.261](https://www.3gpp.org/DynaReport/22261.htm): 5G service requirements.
- [3GPP TS 38.300](https://www.3gpp.org/DynaReport/38300.htm): NR/NG-RAN overall description.
- [3GPP TS 38.321](https://www.3gpp.org/DynaReport/38321.htm): NR Medium Access Control (MAC), including MAC procedures and HARQ-related behavior.
- [3GPP TS 38.211](https://www.3gpp.org/DynaReport/38211.htm), [TS 38.212](https://www.3gpp.org/DynaReport/38212.htm), and [TS 38.214](https://www.3gpp.org/DynaReport/38214.htm): NR physical channels/modulation, channel coding, and physical-layer data procedures.
- [3GPP TS 23.501](https://www.3gpp.org/DynaReport/23501.htm): 5G system architecture and QoS characteristics, including packet delay budget.
- [NEXUS: Efficient and Scalable Multi-Cell mmWave Baseband Processing with Heterogeneous Compute](https://arxiv.org/abs/2509.04625): research on implementation-level heterogeneous baseband processing; it is an example of research, not a normative 3GPP specification.

The standards evolve across releases; consult the applicable release and deployment profile when using them for engineering requirements. The references provide technical context for this README and do not imply that this prototype implements all procedures defined by those standards.
