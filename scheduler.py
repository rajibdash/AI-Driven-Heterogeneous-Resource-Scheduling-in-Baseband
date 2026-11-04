"""Online, model-based scheduling for serial heterogeneous accelerators."""

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Mapping


def _positive(value: float, name: str) -> None:
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class Performance:
    """Runtime per work unit (microseconds) and active accelerator power (watts)."""

    time_per_unit_us: float
    power_w: float

    def __post_init__(self):
        _positive(self.time_per_unit_us, "time_per_unit_us")
        _positive(self.power_w, "power_w")


@dataclass(frozen=True)
class Accelerator:
    name: str
    profiles: Mapping[str, Performance]

    def __post_init__(self):
        if not self.name or not self.profiles:
            raise ValueError("accelerator name and workload profiles are required")
        if any(not kind or not isinstance(profile, Performance)
               for kind, profile in self.profiles.items()):
            raise ValueError("profiles must map workload kinds to Performance objects")
        object.__setattr__(self, "profiles", MappingProxyType(dict(self.profiles)))


@dataclass(frozen=True)
class Workload:
    """A non-preemptible task with an absolute deadline on the caller's clock."""

    task_id: str
    kind: str
    work: float
    deadline_us: float

    def __post_init__(self):
        if not self.task_id or not self.kind:
            raise ValueError("task_id and kind are required")
        _positive(self.work, "work")
        _positive(self.deadline_us, "deadline_us")


@dataclass(frozen=True)
class Assignment:
    workload: Workload
    accelerator: str
    start_us: float
    finish_us: float
    power_w: float

    @property
    def energy_uj(self) -> float:
        return (self.finish_us - self.start_us) * self.power_w


@dataclass(frozen=True)
class Rejection:
    workload: Workload
    reason: str


@dataclass(frozen=True)
class Schedule:
    assignments: tuple[Assignment, ...]
    rejected: tuple[Rejection, ...]


class AdaptiveScheduler:
    """EDF admission with learned runtimes and power-aware interval reservations.

    Each accelerator executes one task at a time. Feedback updates an EWMA
    runtime model per (accelerator, workload kind); committed slots are immutable.
    Calls must be serialized by the caller.
    """

    def __init__(self, accelerators, power_budget_w: float,
                 learning_rate: float = 0.3, energy_weight: float = 1.0):
        _positive(power_budget_w, "power_budget_w")
        if not isfinite(learning_rate) or not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if not isfinite(energy_weight) or energy_weight < 0:
            raise ValueError("energy_weight must be finite and nonnegative")
        accelerators = tuple(accelerators)
        if not accelerators or len({a.name for a in accelerators}) != len(accelerators):
            raise ValueError("provide accelerators with unique names")
        self.accelerators = MappingProxyType({a.name: a for a in accelerators})
        self.power_budget_w = power_budget_w
        self.learning_rate = learning_rate
        self.energy_weight = energy_weight
        self._estimates = {
            (a.name, kind): profile.time_per_unit_us
            for a in accelerators for kind, profile in a.profiles.items()
        }
        self._reservations: list[Assignment] = []
        self._now_us = 0.0

    def observe(self, accelerator: str, kind: str, work: float,
                duration_us: float) -> None:
        """Learn from a measured execution, normalized by its work size."""
        _positive(work, "work")
        _positive(duration_us, "duration_us")
        key = (accelerator, kind)
        if key not in self._estimates:
            raise ValueError("unknown accelerator/workload combination")
        sample = duration_us / work
        _positive(sample, "measured time per unit")
        previous = self._estimates[key]
        self._estimates[key] = (
            (1 - self.learning_rate) * previous + self.learning_rate * sample
        )

    def _fits(self, name, start, finish, power):
        overlapping = [
            r for r in self._reservations
            if r.start_us < finish and start < r.finish_us
        ]
        if any(r.accelerator == name for r in overlapping):
            return False
        # Active power changes only at interval boundaries (half-open slots).
        checkpoints = {start} | {
            r.start_us for r in overlapping if start < r.start_us < finish
        }
        return all(
            power + sum(r.power_w for r in overlapping
                        if r.start_us <= t < r.finish_us) <= self.power_budget_w
            for t in checkpoints
        )

    def schedule(self, workloads, now_us: float = 0.0) -> Schedule:
        """Reserve feasible slots; report unschedulable tasks without overbooking.

        Time is absolute, finite, nonnegative, and monotonic across calls.
        Task identifiers must be unique within a batch and active reservations.
        """
        if not isfinite(now_us) or now_us < self._now_us:
            raise ValueError("now_us must be finite and monotonic")
        workloads = tuple(workloads)
        ids = [w.task_id for w in workloads]
        active = [r for r in self._reservations if r.finish_us > now_us]
        if len(set(ids)) != len(ids) or set(ids) & {r.workload.task_id for r in active}:
            raise ValueError("task identifiers must be unique")
        self._now_us = now_us
        self._reservations = active
        assigned, rejected = [], []
        for workload in sorted(workloads, key=lambda w: (w.deadline_us, w.task_id)):
            candidates = []
            starts = sorted({now_us} | {r.finish_us for r in self._reservations
                                       if r.finish_us >= now_us})
            supported = False
            for name, accelerator in self.accelerators.items():
                profile = accelerator.profiles.get(workload.kind)
                if profile is None:
                    continue
                supported = True
                if profile.power_w > self.power_budget_w:
                    continue
                duration = self._estimates[name, workload.kind] * workload.work
                for start in starts:
                    finish = start + duration
                    if not isfinite(finish) or finish > workload.deadline_us:
                        break
                    if finish <= start:
                        continue
                    if self._fits(name, start, finish, profile.power_w):
                        assignment = Assignment(
                            workload, name, start, finish, profile.power_w
                        )
                        # Normalize energy by budget to retain microsecond units.
                        score = (finish - now_us + self.energy_weight
                                 * duration * (profile.power_w / self.power_budget_w))
                        candidates.append((score, finish, name, assignment))
                        break
            if candidates:
                chosen = min(candidates, key=lambda c: c[:3])[3]
                assigned.append(chosen)
                self._reservations.append(chosen)
            else:
                reason = ("no compatible accelerator" if not supported else
                          "no slot meets deadline and power budget")
                rejected.append(Rejection(workload, reason))
        return Schedule(tuple(assigned), tuple(rejected))


def main():
    """Run an illustrative sub-millisecond baseband scheduling simulation."""
    scheduler = AdaptiveScheduler([
        Accelerator("gpu", {
            "beamforming": Performance(80, 120),
            "channel_estimation": Performance(60, 100),
        }),
        Accelerator("fpga", {
            "channel_estimation": Performance(50, 35),
            "decoding": Performance(110, 45),
        }),
        Accelerator("dsp", {
            "beamforming": Performance(140, 25),
            "decoding": Performance(180, 20),
        }),
    ], power_budget_w=150)
    tasks = [
        Workload("beam-1", "beamforming", 1, 200),
        Workload("channel-1", "channel_estimation", 1, 250),
        Workload("decode-1", "decoding", 2, 600),
    ]
    result = scheduler.schedule(tasks)
    for assignment in result.assignments:
        print(f"{assignment.workload.task_id}: {assignment.accelerator} "
              f"[{assignment.start_us:.1f}, {assignment.finish_us:.1f}) us, "
              f"{assignment.power_w:.1f} W")
    for rejection in result.rejected:
        print(f"{rejection.workload.task_id}: rejected ({rejection.reason})")
    scheduler.observe("gpu", "beamforming", work=1, duration_us=240)
    updated = scheduler.schedule(
        [Workload("beam-2", "beamforming", 1, 1000)], now_us=700
    )
    for assignment in updated.assignments:
        print(f"After feedback: {assignment.workload.task_id} -> "
              f"{assignment.accelerator}")


if __name__ == "__main__":
    main()
