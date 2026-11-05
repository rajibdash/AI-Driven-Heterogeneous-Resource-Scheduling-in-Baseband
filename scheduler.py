"""Compatibility exports for the baseband_scheduler package."""

from baseband_scheduler.scheduler import (
    Accelerator,
    AdaptiveScheduler,
    Assignment,
    Performance,
    Rejection,
    Schedule,
    Workload,
    main,
)

__all__ = [
    "Accelerator",
    "AdaptiveScheduler",
    "Assignment",
    "Performance",
    "Rejection",
    "Schedule",
    "Workload",
]


if __name__ == "__main__":
    main()
