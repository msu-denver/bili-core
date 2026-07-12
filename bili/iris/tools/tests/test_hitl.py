"""Tests for bili.iris.tools.hitl.

Covers:
- NullHitlResponder returns the no-response sentinel without blocking.
- ScriptedHitlResponder cycles scripted answers and records calls.
- ScriptedHitlResponder raises when the script is exhausted (a test-authoring
  bug, not a runtime condition to degrade from).
- ScriptedHitlResponder is genuinely thread-safe under real concurrent
  access (not just "has a lock") -- the documented contract every
  HitlResponder implementation must honor, since a fan-out batch of agent
  runs can have more than one ask_user call in flight against one
  responder at once.
- HitlResponder is runtime_checkable and isinstance-matches any object with
  a compatible ask() method (structural typing, not a required base class).
"""

# pylint: disable=missing-function-docstring

import threading
import time

import pytest

from bili.iris.tools.hitl import (
    NO_RESPONSE_PREFIX,
    HitlResponder,
    NullHitlResponder,
    ScriptedHitlResponder,
)


class _NoOpLock:
    """Context-manager stand-in for threading.Lock() that acquires nothing.

    Swapped onto a ScriptedHitlResponder instance's _lock attribute in
    test_lock_prevents_the_append_then_read_race to simulate "the lock was
    never added" while running the exact same ask() method body -- proves
    the real lock is what prevents the race, not a synthetic twin class.
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class TestNullHitlResponder:
    """Tests for NullHitlResponder."""

    def test_returns_no_response_sentinel(self):
        responder = NullHitlResponder()
        answer = responder.ask("Which environment?")
        assert answer.startswith(NO_RESPONSE_PREFIX)

    def test_ignores_options(self):
        responder = NullHitlResponder()
        answer = responder.ask("Pick one", options=["a", "b"])
        assert answer.startswith(NO_RESPONSE_PREFIX)

    def test_satisfies_hitl_responder_protocol(self):
        assert isinstance(NullHitlResponder(), HitlResponder)


class TestScriptedHitlResponder:
    """Tests for ScriptedHitlResponder."""

    def test_returns_answers_in_order(self):
        responder = ScriptedHitlResponder(["staging", "yes"])
        assert responder.ask("Which environment?") == "staging"
        assert responder.ask("Proceed?") == "yes"

    def test_records_calls(self):
        responder = ScriptedHitlResponder(["staging"])
        responder.ask("Which environment?", options=["staging", "prod"])
        assert responder.calls == [
            {"question": "Which environment?", "options": ["staging", "prod"]}
        ]

    def test_calls_property_returns_a_copy(self):
        responder = ScriptedHitlResponder(["staging"])
        responder.ask("q")
        calls = responder.calls
        calls.append({"question": "tampered", "options": None})
        assert len(responder.calls) == 1

    def test_raises_when_script_exhausted(self):
        responder = ScriptedHitlResponder(["only-one"])
        responder.ask("first question")
        with pytest.raises(IndexError):
            responder.ask("second question, no script left")

    def test_satisfies_hitl_responder_protocol(self):
        assert isinstance(ScriptedHitlResponder([]), HitlResponder)

    def test_concurrent_calls_get_distinct_correctly_ordered_answers(self):
        """N threads calling ask() at once against ONE responder each get a
        distinct answer with no lost update or duplicate index.
        """
        n = 20
        answers = [str(i) for i in range(n)]
        responder = ScriptedHitlResponder(answers)
        results = []
        results_lock = threading.Lock()

        def worker(i):
            answer = responder.ask(f"question {i}")
            with results_lock:
                results.append(answer)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(results) == n
        assert len(set(results)) == n, "expected all answers distinct, got a collision"
        assert len(responder.calls) == n

    def test_lock_prevents_the_append_then_read_race(self):
        """Directly proves the lock in ScriptedHitlResponder.ask is load-bearing.

        A bare append-then-read race is often NOT observable under CPython's
        GIL without deliberately widening the window (individual list
        operations are fast enough to rarely get preempted), so a plain
        concurrent-calls test can pass by GIL-timing luck without the lock
        actually doing anything -- a false sense of coverage. This test
        widens the window INSIDE the real class's real critical section (a
        slow-appending list stands in for self._calls, so the delay happens
        between the append and the length-read the lock is meant to guard)
        and shows the class collides with the lock removed but not with it
        present, using the actual production code path both times -- not a
        synthetic stand-in class.
        """
        n = 20
        answers = [str(i) for i in range(n)]

        class _SlowAppendList(list):
            """A list whose append() sleeps, widening the race window that
            follows it (the read of len(self)) for whichever code holds
            (or fails to hold) the responder's lock across both operations.
            """

            def append(self, item):
                super().append(item)
                time.sleep(0.005)

        def _run_concurrently(responder):
            results = []
            results_lock = threading.Lock()

            def worker(i):
                answer = responder.ask(f"question {i}")
                with results_lock:
                    results.append(answer)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)
            return results

        locked = ScriptedHitlResponder(answers)
        locked._calls = _SlowAppendList()  # pylint: disable=protected-access
        locked_results = _run_concurrently(locked)
        assert len(set(locked_results)) == n, (
            "the real, locked ScriptedHitlResponder must not collide even "
            "with the append-then-read window deliberately widened"
        )

        unlocked = ScriptedHitlResponder(answers)
        unlocked._calls = _SlowAppendList()  # pylint: disable=protected-access
        unlocked._lock = _NoOpLock()  # pylint: disable=protected-access
        unlocked_results = _run_concurrently(unlocked)
        assert len(set(unlocked_results)) < n, (
            "expected the lock-defeated instance to collide under the "
            "widened race window -- if it did not, the race-window "
            "widening stopped working and this test needs a wider delay, "
            "not a passing grade"
        )
