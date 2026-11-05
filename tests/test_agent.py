import unittest

from baseband_scheduler import (
    Accelerator,
    AdaptiveScheduler,
    Performance,
    SchedulingAgent,
    Workload,
)


class SchedulingAgentTests(unittest.TestCase):
    def make_agent(self):
        scheduler = AdaptiveScheduler([
            Accelerator("gpu", {"beam": Performance(10, 80)}),
            Accelerator("dsp", {"beam": Performance(20, 30)}),
        ], power_budget_w=100)
        return SchedulingAgent(scheduler), scheduler

    def test_tight_slack_selects_latency_and_fastest_device(self):
        agent, _ = self.make_agent()
        result = agent.plan([Workload("urgent", "beam", 1, 12)])
        self.assertEqual(result.mode, "latency")
        self.assertEqual(result.energy_weight, 0)
        self.assertEqual(result.schedule.assignments[0].accelerator, "gpu")
        self.assertIn("prioritize earliest completion", result.rationale)

    def test_relaxed_slack_selects_lower_energy_feasible_device(self):
        agent, _ = self.make_agent()
        result = agent.plan([Workload("relaxed", "beam", 1, 100)])
        self.assertEqual(result.mode, "energy")
        self.assertEqual(result.schedule.assignments[0].accelerator, "dsp")

    def test_agent_cannot_bypass_deadline_feasibility(self):
        agent, _ = self.make_agent()
        result = agent.plan([Workload("impossible", "beam", 1, 5)])
        self.assertEqual(result.mode, "latency")
        self.assertFalse(result.schedule.assignments)
        self.assertEqual(result.schedule.rejected[0].reason,
                         "no slot meets deadline and power budget")

    def test_unsupported_workload_uses_default_and_is_rejected(self):
        agent, _ = self.make_agent()
        result = agent.plan([Workload("unsupported", "fft", 1, 100)])
        self.assertEqual(result.mode, "balanced")
        self.assertIsNone(result.tightest_slack_ratio)
        self.assertFalse(result.schedule.assignments)
        self.assertEqual(result.schedule.rejected[0].reason,
                         "no compatible accelerator")

    def test_empty_batch_uses_default(self):
        agent, _ = self.make_agent()
        result = agent.plan([])
        self.assertEqual(result.mode, "balanced")
        self.assertFalse(result.schedule.assignments)
        self.assertFalse(result.schedule.rejected)

    def test_invalid_thresholds(self):
        agent, scheduler = self.make_agent()
        with self.assertRaises(ValueError):
            SchedulingAgent(scheduler, urgent_slack_ratio=4,
                            relaxed_slack_ratio=3)
        with self.assertRaises(ValueError):
            SchedulingAgent(scheduler, energy_weight=-1)


if __name__ == "__main__":
    unittest.main()
