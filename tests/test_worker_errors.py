"""Guard the one invariant that keeps worker failures visible. Plain asserts, no framework.

    python tests/test_worker_errors.py

Background, measured against the installed superqt: a @thread_worker GENERATOR that raises
RuntimeError emits neither `errored` nor `finished`. GeneratorWorker.work() catches RuntimeError
and returns it, and WorkerBase.run() then warns and returns before either signal. The dialog never
appears, the button stays disabled, and the worker still reports is_running == True. Every DL
failure message in pipeline.py is a RuntimeError, so without FenestraError they are all invisible.

If someone ever makes FenestraError inherit RuntimeError, this file is what notices.
"""

import inspect
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from fenestra.pipeline import FenestraError, _reporting


def test_fenestra_error_is_not_a_runtimeerror():
    # The entire point. superqt special-cases RuntimeError and swallows it.
    assert not issubclass(FenestraError, RuntimeError)
    assert issubclass(FenestraError, Exception)


def test_reporting_converts_runtimeerror_and_keeps_the_cause():
    @_reporting
    def boom():
        raise RuntimeError("Container DL Inference failed: stderr here")
        yield

    try:
        list(boom())
    except FenestraError as e:
        assert "Container DL Inference failed" in str(e)
        assert isinstance(e.__cause__, RuntimeError)
    else:
        raise AssertionError("a RuntimeError in a worker must surface as FenestraError")


def test_reporting_still_produces_a_generator_function():
    # thread_worker picks GeneratorWorker via inspect.isgeneratorfunction. If the wrapper stopped
    # being a generator function the worker type would change underneath us.
    @_reporting
    def gen():
        yield 1
        yield 2

    assert inspect.isgeneratorfunction(gen)
    assert list(gen()) == [1, 2]


def test_reporting_does_not_touch_other_exceptions():
    @_reporting
    def bad_engine():
        raise ValueError("Unknown engine: Podman")
        yield

    try:
        list(bad_engine())
    except ValueError:
        pass
    else:
        raise AssertionError("non-RuntimeError must pass through untouched")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")
