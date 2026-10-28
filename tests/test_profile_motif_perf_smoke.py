import importlib.util
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "benchmark" / "ibm-aml" / "profile_motif_perf.py"
SPHERE_PATH = REPO_ROOT / "benchmark" / "ibm-aml" / "hi_small_sphere" / "gds_aml_hi_small"


def _sphere_unavailable() -> bool:
    sphere_json = SPHERE_PATH / "_gds_meta" / "sphere.json"
    if not sphere_json.exists():
        return True
    import json

    return json.loads(sphere_json.read_text()).get("format_version", "").split(".")[0] != "4"


@pytest.mark.skipif(
    _sphere_unavailable(),
    reason="AML HI-Small sphere not present (format 3.x)",
)
@pytest.mark.skipif(
    os.environ.get("HYPERTOPOS_RUN_PROFILE_SMOKE") != "1",
    reason="runs the full profiling script (minutes); set HYPERTOPOS_RUN_PROFILE_SMOKE=1",
)
def test_profile_script_produces_output(tmp_path, monkeypatch):
    """Integration smoke: cProfile script runs end-to-end and writes output."""
    spec = importlib.util.spec_from_file_location("profile_motif_perf", SCRIPT_PATH)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(script, "OUTPUT_DIR", tmp_path)
    script.main()
    files = list(tmp_path.glob("*-motif-fhpm-cprofile.txt"))
    assert len(files) == 1
    content = files[0].read_text(encoding="utf-8")
    assert "bipartite_burst" in content
    assert "cycle_3" in content
    assert "chain_k" in content
    assert "adj build (one-time)" in content
