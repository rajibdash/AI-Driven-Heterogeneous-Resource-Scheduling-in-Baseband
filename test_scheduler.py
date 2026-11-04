import unittest

from scheduler import Accelerator, AdaptiveScheduler, Performance, Workload


class SchedulerTests(unittest.TestCase):
    def make_scheduler(self, budget=100, **kwargs):
        return AdaptiveScheduler([
            Accelerator("gpu", {"beam": Performance(10, 80)}),
            Accelerator("dsp", {"beam": Performance(20, 30),
                                "decode": Performance(15, 30)}),
        ], budget, **kwargs)

    def test_deadlines_and_serial_execution(self):
        scheduler = self.make_scheduler()
        result = scheduler.schedule([
            Workload("later", "beam", 1, 40),
            Workload("urgent", "beam", 1, 10),
        ])
        self.assertEqual([a.workload.task_id for a in result.assignments],
                         ["urgent", "later"])
        self.assertEqual([(a.start_us, a.finish_us) for a in result.assignments],
                         [(0, 10), (10, 20)])

    def test_power_budget_serializes_otherwise_parallel_resources(self):
        result = self.make_scheduler().schedule([
            Workload("beam", "beam", 1, 10),
            Workload("decode", "decode", 1, 30),
        ])
        self.assertEqual([a.start_us for a in result.assignments], [0, 10])

    def test_parallel_execution_within_budget(self):
        result = self.make_scheduler(budget=110).schedule([
            Workload("beam", "beam", 1, 10),
            Workload("decode", "decode", 1, 20),
        ])
        self.assertEqual([a.start_us for a in result.assignments], [0, 0])

    def test_power_check_includes_future_reservation_starts(self):
        scheduler = AdaptiveScheduler([
            Accelerator("a", {"first": Performance(10, 20),
                              "second": Performance(10, 80)}),
            Accelerator("b", {"long": Performance(15, 30)}),
        ], 100)
        result = scheduler.schedule([
            Workload("first", "first", 1, 10),
            Workload("second", "second", 1, 20),
            Workload("long", "long", 1, 40),
        ])
        self.assertEqual(result.assignments[-1].start_us, 20)

    def test_feedback_changes_mapping_and_scales_by_work(self):
        scheduler = self.make_scheduler(learning_rate=1)
        first = scheduler.schedule([Workload("first", "beam", 1, 50)])
        self.assertEqual(first.assignments[0].accelerator, "gpu")
        scheduler.observe("gpu", "beam", work=2, duration_us=100)
        second = scheduler.schedule(
            [Workload("second", "beam", 2, 100)], now_us=50
        )
        self.assertEqual(second.assignments[0].accelerator, "dsp")
        self.assertEqual(second.assignments[0].finish_us, 90)

    def test_energy_preference(self):
        result = self.make_scheduler(energy_weight=10).schedule([
            Workload("beam", "beam", 1, 40)
        ])
        self.assertEqual(result.assignments[0].accelerator, "dsp")
        self.assertEqual(result.assignments[0].energy_uj, 600)

    def test_rejections_do_not_reserve_resources(self):
        result = self.make_scheduler(budget=25).schedule([
            Workload("unsupported", "fft", 1, 100),
            Workload("power", "beam", 1, 100),
        ])
        self.assertFalse(result.assignments)
        self.assertEqual(len(result.rejected), 2)
        scheduler = self.make_scheduler()
        result = scheduler.schedule([
            Workload("expired", "beam", 1, 5),
            Workload("feasible", "beam", 1, 10),
        ])
        self.assertEqual(result.assignments[0].start_us, 0)
        self.assertEqual(len(result.rejected), 1)

    def test_reservations_persist_and_expire_between_batches(self):
        scheduler = self.make_scheduler()
        scheduler.schedule([Workload("first", "beam", 1, 10)])
        result = scheduler.schedule([Workload("next", "beam", 1, 30)], now_us=5)
        self.assertEqual(result.assignments[0].start_us, 10)
        result = scheduler.schedule([Workload("first", "beam", 1, 40)], now_us=20)
        self.assertEqual(result.assignments[0].start_us, 20)
        with self.assertRaises(ValueError):
            scheduler.schedule([], now_us=19)

    def test_duplicate_ids_rejected_without_state_change(self):
        scheduler = self.make_scheduler()
        workload = Workload("one", "beam", 1, 100)
        with self.assertRaises(ValueError):
            scheduler.schedule([workload, workload], now_us=50)
        scheduler.schedule([workload])
        with self.assertRaises(ValueError):
            scheduler.schedule([workload])

    def test_invalid_inputs(self):
        for invalid in [0, -1, float("inf"), float("nan")]:
            with self.subTest(value=invalid):
                with self.assertRaises(ValueError):
                    Performance(invalid, 10)
                with self.assertRaises(ValueError):
                    Performance(10, invalid)
                with self.assertRaises(ValueError):
                    Workload("id", "beam", invalid, 100)
                with self.assertRaises(ValueError):
                    self.make_scheduler(budget=invalid)
                with self.assertRaises(ValueError):
                    self.make_scheduler().observe("gpu", "beam", 1, invalid)
        with self.assertRaises(ValueError):
            self.make_scheduler().observe("unknown", "beam", 1, 10)
        with self.assertRaises(ValueError):
            self.make_scheduler().schedule([], now_us=-1)
        with self.assertRaises(ValueError):
            self.make_scheduler(learning_rate=2)
        with self.assertRaises(ValueError):
            self.make_scheduler(energy_weight=-1)

    def test_profile_mapping_is_copied(self):
        profiles = {"beam": Performance(10, 20)}
        accelerator = Accelerator("gpu", profiles)
        profiles.clear()
        self.assertIn("beam", accelerator.profiles)

    def test_empty_batch(self):
        result = self.make_scheduler().schedule([])
        self.assertFalse(result.assignments)
        self.assertFalse(result.rejected)


if __name__ == "__main__":
    unittest.main()
