from pathlib import Path
import subprocess
import sys


def test_pipeline_runs():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, 'run_all.py'], cwd=root, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (root / 'outputs' / 'pre_layer' / 'prelayer_result.json').exists()
    assert (root / 'outputs' / 'layer1' / 'scenarios_long.csv').exists()
    assert (root / 'outputs' / 'layer2' / 'recommended_candidate.json').exists()
    assert (root / 'outputs' / 'layer3' / 'explanation_report.md').exists()
