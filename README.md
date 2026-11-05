# AI-Driven-Heterogeneous-Resource-Scheduling-in-Baseband
AI-Driven Heterogeneous Resource Scheduling for 5G or 6G Baseband Processing as next-generation networks demand unprecedented data rates.

## The Bottleneck of Traditional/Conventional Scheduling

In modern RAN systems, user scheduling is the heartbeat of uplink and downlink efficiency. 
Every TTI, the scheduler must decide which UEs to serve — balancing throughput, fairness, 
QoS, buffer backlog, and HARQ state. Traditional schedulers rely on handcrafted heuristics 
(PF, RR, weighted PF), but the complexity of real‑world radio conditions makes those rules 
brittle.

Then, the clash of Heterogeneous Traffic (Coexistence Bottleneck).

So, traditional scheduling is structurally blind to multi-service multiplexing. 5G requires the 
simultaneous management of three wildly different traffic profiles:
* eMBB (Enhanced Mobile Broadband): Needs massive data blocks and high throughput.
* URLLC (Ultra-Reliable Low-Latency Communications)/TCC: Needs sub-millisecond priority access. low latency communications.
* mMTC (Massive Machine-Type Communications): Consists of millions of low-power, sporadic IoT pings. massive IoT connectivity.

At the baseband L1 physical layer, different workloads like channel estimation, massive MIMO 
beamforming, and decoding fluctuate rapidly based on user traffic and radio channel conditions. 
Traditional scheduling methods face severe limitations like below:
*  **Static heuristics**: Rule-based schedulers (e.g., Proportional Fair, Round Robin) fail under highly dynamic network conditions.
*  **Hardware underutilization**: Inability to fluidly distribute tasks across heterogeneous chips (CPUs, GPUs, FPGAs, and ASICs).
*  **Strict SLA violations**: Missing tight sub-millisecond deadlines, leading to dropped packets and high power consumption (power budget is higher).

## Adaptive scheduling prototype

`scheduler.py` provides a dependency-free Python 3.10+ scheduling model for
beamforming, channel estimation, decoding, or any caller-defined workload kind.
It maps non-preemptible tasks onto compatible, serial accelerators using:

- Earliest-deadline-first admission with explicit rejection of infeasible tasks.
- Per-accelerator, per-workload runtime estimates, updated by an exponentially
  weighted moving average of measured execution time per work unit.
- Interval reservations that prevent accelerator overbooking and keep summed
  active power within a configurable budget, including across submission batches.
- A completion-time/energy score to favor prompt, energy-efficient mappings.
  `energy_weight=0` favors earliest completion; larger values favor lower energy.

Run the illustrative simulation and the standard-library tests from the repository
root (no installation required):

```sh
python scheduler.py
python -m unittest -v test_scheduler
```

### Integrating measured feedback

```python
from scheduler import Accelerator, AdaptiveScheduler, Performance, Workload

scheduler = AdaptiveScheduler([
    Accelerator("gpu", {"beamforming": Performance(80, 120)}),
    Accelerator("dsp", {"beamforming": Performance(140, 25)}),
], power_budget_w=150)

plan = scheduler.schedule(
    [Workload("beam-1", "beamforming", work=1, deadline_us=200)],
    now_us=0,
)
for assignment in plan.assignments:
    print(assignment.accelerator, assignment.start_us, assignment.finish_us)
for rejection in plan.rejected:
    print(rejection.workload.task_id, rejection.reason)

# Feed actual accelerator measurements back before subsequent scheduling calls.
scheduler.observe("gpu", "beamforming", work=1, duration_us=240)
next_plan = scheduler.schedule(
    [Workload("beam-2", "beamforming", work=1, deadline_us=1000)],
    now_us=700,
)
```

All times are microseconds on the same caller-controlled, monotonic clock;
deadlines are absolute. Work units must be consistent within each workload kind,
and measured durations should capture execution, not time waiting in a queue.
Profiles describe active power in watts and initial execution time per work unit.
The default learning rate is 0.3. Accepted reservations persist across calls until
their predicted finish time; feedback changes future estimates, not existing slots.
Task IDs must be unique within a batch and among active reservations.
Calls must be serialized.

### Scope and limitations

This is a scheduling simulation/control-plane foundation, not a hardware executor
or a trained offline neural policy. Online runtime learning allows subsequent
mappings to adapt to measured workload/channel-condition changes. The example
uses synthetic profiles and sub-millisecond **predicted execution** windows;
it does not establish sub-millisecond scheduler overhead or real-time guarantees.

Deadline and power checks are only as reliable as supplied profiles and runtime
estimates. The active-power budget excludes idle/platform power and assumes fixed
active power per accelerator/workload profile. An integration must reserve idle
power separately, measure hardware under representative loads, account for transfer
and dispatch overhead in profiles, and enforce limits at runtime. Overruns need
external handling before dispatching dependent reservations. This greedy planner
does not guarantee globally optimal utilization or acceptance, model dependencies,
preemption, memory limits, or simultaneous kernels on one accelerator.
