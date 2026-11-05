"""Deterministic policy agent for explainable scheduling decisions."""

from dataclasses import dataclass
from math import isfinite

from .scheduler import AdaptiveScheduler, Schedule, Workload


@dataclass(frozen=True)
class SchedulingResult:
    """A schedule together with the agent's selected policy and rationale."""

    schedule: Schedule
    mode: str
    energy_weight: float
    tightest_slack_ratio: float | None
    rationale: str


class SchedulingAgent:
    """Select a per-batch scheduling objective from estimated deadline slack.

    This is a deterministic policy layer, not an LLM. The scheduler remains the
    authority for compatibility, device serialization, deadline, and power checks.
    """

    def __init__(self, scheduler: AdaptiveScheduler,
                 urgent_slack_ratio: float = 1.5,
                 relaxed_slack_ratio: float = 4.0,
                 energy_weight: float = 10.0):
        if (not isfinite(urgent_slack_ratio) or urgent_slack_ratio <= 0
                or not isfinite(relaxed_slack_ratio)
                or relaxed_slack_ratio <= urgent_slack_ratio):
            raise ValueError("slack thresholds must be finite, positive, and increasing")
        if not isfinite(energy_weight) or energy_weight < 0:
            raise ValueError("energy_weight must be finite and nonnegative")
        self.scheduler = scheduler
        self.urgent_slack_ratio = urgent_slack_ratio
        self.relaxed_slack_ratio = relaxed_slack_ratio
        self.energy_weight = energy_weight

    def plan(self, workloads, now_us: float = 0.0) -> SchedulingResult:
        """Choose latency, balanced, or energy preference, then request a safe plan."""
        workloads = tuple(workloads)
        ratios = []
        for workload in workloads:
            estimate = self.scheduler.estimated_duration_us(workload)
            if estimate is None:
                continue
            slack = workload.deadline_us - now_us
            ratio = slack / estimate if isfinite(estimate) and estimate > 0 else 0.0
            ratios.append(ratio)

        tightest = min(ratios) if ratios else None
        if tightest is None:
            mode = "balanced"
            weight = self.scheduler.energy_weight
            rationale = "No workload has a compatible device profile; using the scheduler default."
        elif tightest <= self.urgent_slack_ratio:
            mode = "latency"
            weight = 0.0
            rationale = (
                f"Tightest estimated deadline-slack ratio is {tightest:.2f}; "
                "prioritize earliest completion."
            )
        elif tightest >= self.relaxed_slack_ratio:
            mode = "energy"
            weight = self.energy_weight
            rationale = (
                f"Tightest estimated deadline-slack ratio is {tightest:.2f}; "
                "favor lower-energy feasible placements."
            )
        else:
            mode = "balanced"
            weight = self.scheduler.energy_weight
            rationale = (
                f"Tightest estimated deadline-slack ratio is {tightest:.2f}; "
                "retain the scheduler's balanced default."
            )

        schedule = self.scheduler.schedule(
            workloads, now_us=now_us, energy_weight=weight
        )
        return SchedulingResult(schedule, mode, weight, tightest, rationale)
