"""The one check behind _build_dl_cmd. Plain asserts, no test framework.

There is no CI in this repo, so run it by hand in the host environment:

    python tests/test_dl_cmd.py

It exists because the DL argv used to be copy-pasted into two functions that had to be kept in
sync by hand. These asserts fail if the three engines ever stop agreeing on the contract.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from fenestra.pipeline import _build_dl_cmd, _engine_key

ARGS = dict(
    temp_in_path="/tmp/run/temp_in.tif",
    temp_out_dir="/tmp/run/out",
    model_path="/models/best_model_ema.pth",
    architecture="hat",
)


def test_engine_key_normalises_ui_labels():
    assert _engine_key("Local (bundled)") == "local"
    assert _engine_key(" Singularity ") == "singularity"
    assert _engine_key("Docker") == "docker"
    assert _engine_key("") == ""


def test_singularity_binds_all_four_paths():
    cmd, env = _build_dl_cmd("Singularity", container_path="/x/dl.sif", **ARGS)
    assert env is None
    assert cmd[:3] == ["singularity", "exec", "--nv"]
    assert cmd.count("--bind") == 4
    # The image sees the mount points, never the host paths.
    assert "/tmp_model/best_model_ema.pth" in cmd
    assert "/models/best_model_ema.pth" not in cmd
    # singularity exec bypasses %runscript, so the explicit interpreter is required.
    assert cmd[cmd.index("/x/dl.sif") + 1] == "python"


def test_docker_and_singularity_pass_the_same_inner_command():
    sing, _ = _build_dl_cmd("Singularity", container_path="/x/dl.sif", **ARGS)
    dock, _ = _build_dl_cmd("Docker", container_path="tag:latest", **ARGS)
    assert dock[:5] == ["docker", "run", "--rm", "--gpus", "all"]
    assert dock.count("-v") == 4
    # This equality is the whole point of the shared builder.
    assert sing[sing.index("python"):] == dock[dock.index("python"):]


def test_local_uses_real_paths_and_scrubs_the_gui_interpreter():
    with tempfile.TemporaryDirectory() as d:
        fake_python = os.path.join(d, "python")
        open(fake_python, "w").close()
        old = dict(os.environ)
        try:
            os.environ["FENESTRA_DL_PYTHON"] = fake_python
            os.environ["PYTHONPATH"] = "/opt/venv-gui/lib/python3.10/site-packages"
            cmd, env = _build_dl_cmd("Local (bundled)", container_path="", **ARGS)
        finally:
            os.environ.clear()
            os.environ.update(old)

    assert cmd[0] == fake_python
    assert cmd[1].endswith(os.path.join("backend", "inference.py"))
    # No bind mounts, no /tmp_in indirection — one filesystem.
    assert "--bind" not in cmd and "-v" not in cmd
    assert "/tmp/run/out" in cmd
    assert cmd[cmd.index("--model_path") + 1] == "/models/best_model_ema.pth"
    # The GUI venv must not leak into the DL interpreter.
    assert "PYTHONPATH" not in env and "PYTHONHOME" not in env


def test_local_without_the_bundled_env_fails_loudly():
    old = os.environ.get("FENESTRA_DL_PYTHON")
    os.environ["FENESTRA_DL_PYTHON"] = "/nonexistent/python"
    try:
        _build_dl_cmd("Local (bundled)", container_path="", **ARGS)
    except RuntimeError as e:
        assert "bundled deep-learning environment" in str(e)
    else:
        raise AssertionError("a missing bundled environment must raise")
    finally:
        if old is None:
            del os.environ["FENESTRA_DL_PYTHON"]
        else:
            os.environ["FENESTRA_DL_PYTHON"] = old


def test_unknown_engine_still_raises():
    try:
        _build_dl_cmd("Podman", container_path="", **ARGS)
    except ValueError as e:
        assert "Unknown engine" in str(e)
    else:
        raise AssertionError("an unknown engine must raise")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")
