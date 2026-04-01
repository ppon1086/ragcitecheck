from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "tests" / "fixtures" / "runs_min"


def test_cli_report_doc_smoke(tmp_path):
    out_dir = tmp_path / "out_report_doc"

    cmd = [
        sys.executable,
        "-m",
        "ragcitecheck.cli",
        "report",
        "--runs",
        str(RUNS_DIR),
        "--out",
        str(out_dir),
        "--evidence-key",
        "doc",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert (out_dir / "validation_summary.json").exists()
    assert (out_dir /"doc"/ "report_meta.json").exists()