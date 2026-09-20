import sys
import unittest
from pathlib import Path
from unittest.mock import patch


CORE = Path(__file__).resolve().parents[1] / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from processing.controller import ProcessingController
from tasks import CancellationToken, TaskContextQueue, TaskCoordinator, TaskStatus
from DataProcessor import MassDataProcessor
from LifetimeCalculator import CalculationThread


class _Worker:
    def __init__(self):
        self.contexts = TaskContextQueue()
        self.cancel_calls = 0

    def enqueue_task_context(self, task_id, token):
        self.contexts.enqueue(task_id, token)

    def cancel(self):
        self.cancel_calls += 1


class _Thread:
    def __init__(self):
        self.start_calls = 0
        self.interruption_calls = 0

    def start(self):
        self.start_calls += 1

    def requestInterruption(self):
        self.interruption_calls += 1


class _Window:
    def __init__(self):
        self.mass_data_processor = _Worker()
        self.avi_thread = _Thread()

    def update_progress(self, *_args):
        pass


class ProcessingTaskContextTests(unittest.TestCase):
    def test_cancelling_first_task_does_not_cancel_next_context(self):
        window = _Window()
        coordinator = TaskCoordinator()
        controller = ProcessingController(window, coordinator)
        with patch("processing.controller.is_thread_active", return_value=True):
            first = controller.begin("avi_thread", "em_processing", "first")
            second = controller.begin("avi_thread", "em_processing", "second")

        self.assertTrue(coordinator.cancel_task(first.task_id))
        first_context = window.mass_data_processor.contexts.next()
        second_context = window.mass_data_processor.contexts.next()

        self.assertTrue(first_context.token.is_cancelled)
        self.assertFalse(second_context.token.is_cancelled)
        self.assertEqual(window.mass_data_processor.cancel_calls, 0)
        self.assertEqual(window.avi_thread.interruption_calls, 0)
        self.assertEqual(first.status, TaskStatus.CANCELLING)
        self.assertEqual(second.status, TaskStatus.RUNNING)

    def test_cancelled_head_advances_to_independent_second_task(self):
        window = _Window()
        coordinator = TaskCoordinator()
        controller = ProcessingController(window, coordinator)
        with patch("processing.controller.is_thread_active", return_value=True):
            first = controller.begin("avi_thread", "em_processing", "first")
            second = controller.begin("avi_thread", "em_processing", "second")

        coordinator.cancel_task(first.task_id)
        controller.cancelled("em_processing")

        self.assertEqual(first.status, TaskStatus.CANCELLED)
        self.assertIs(controller.active("em_processing"), second)
        self.assertFalse(second.token.is_cancelled)

    def test_context_queue_keeps_tokens_in_submission_order(self):
        queue = TaskContextQueue()
        first = CancellationToken()
        second = CancellationToken()
        queue.enqueue("first", first)
        queue.enqueue("second", second)
        first.cancel()

        self.assertEqual(queue.next().task_id, "first")
        next_context = queue.next()
        self.assertEqual(next_context.task_id, "second")
        self.assertFalse(next_context.token.is_cancelled)

    def test_mass_worker_skips_cancelled_head_then_activates_second(self):
        worker = MassDataProcessor()
        first = CancellationToken()
        second = CancellationToken()
        worker.enqueue_task_context("first", first)
        worker.enqueue_task_context("second", second)
        first.cancel()

        self.assertFalse(worker._activate_task_context())
        self.assertTrue(worker._activate_task_context())
        self.assertEqual(worker.task_id, "second")
        self.assertIs(worker.cancellation_token, second)
        self.assertFalse(second.is_cancelled)

    def test_lifetime_worker_activates_each_task_owned_token(self):
        worker = CalculationThread()
        first = CancellationToken()
        second = CancellationToken()
        worker.enqueue_task_context("first", first)
        worker.enqueue_task_context("second", second)
        first.cancel()

        self.assertFalse(worker._activate_task_context())
        self.assertTrue(worker._activate_task_context())
        self.assertEqual(worker.task_id, "second")
        self.assertIs(worker.cancellation_token, second)


if __name__ == "__main__":
    unittest.main()
