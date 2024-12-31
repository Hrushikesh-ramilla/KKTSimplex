"""
Unit tests for KKTSimplex CLI subcommands (audit, gate, certify, export-dpo).
"""

import os
import sys
import json
import tempfile
import subprocess
import pytest


def run_kkt_simplex_cli(args: list[str]) -> subprocess.CompletedProcess:
    """Run kkt_simplex CLI module via subprocess."""
    cmd = [sys.executable, "-m", "kkt_simplex.cli"] + args
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


def test_cli_audit():
    """Verify kkt_simplex audit subcommand."""
    res = run_kkt_simplex_cli(["audit", "--passed", "--critique", "You made a terrible mistake!"])
    assert res.returncode == 0
    assert "KKTSIMPLEX AUDIT ADJUDICATION: REJECT_HOSTILE" in res.stdout
    assert "Agent Calibrated Trust:" in res.stdout


def test_cli_gate():
    """Verify kkt_simplex gate subcommand."""
    res = run_kkt_simplex_cli(["gate", sys.executable, "-c", "print('hello from gate')"])
    assert res.returncode == 0
    assert "Gate Return Code: 0" in res.stdout
    assert "Gate Success:     True" in res.stdout


def test_cli_certify():
    """Verify kkt_simplex certify subcommand produces valid certificate JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cert_out = os.path.join(tmpdir, "cert.json")
        res = run_kkt_simplex_cli([
            "certify",
            "--code", "def multiply(a, b): return a * b",
            "--test", "assert multiply(3, 4) == 12",
            "--critique", "Your multiplication logic is flawed!",
            "--out", cert_out,
        ])

        assert res.returncode == 0
        assert "KKTSIMPLEX EXECUTION PROOF CERTIFICATION: REJECT_HOSTILE" in res.stdout
        assert "KKT Optimality Valid: True" in res.stdout
        assert os.path.isfile(cert_out)

        with open(cert_out, "r", encoding="utf-8") as f:
            cert_data = json.load(f)

        assert cert_data["adjudication"] == "REJECT_HOSTILE"
        assert cert_data["kkt_proof"]["is_valid"] is True
        assert cert_data["execution_proof"]["is_proven"] is True


def test_cli_export_dpo():
    """Verify kkt_simplex export-dpo subcommand generates valid dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dpo_out = os.path.join(tmpdir, "dpo.jsonl")
        res = run_kkt_simplex_cli([
            "export-dpo",
            "--out", dpo_out,
            "--num-samples", "10",
        ])

        assert res.returncode == 0
        assert "Exported 10 anti-sycophancy DPO pairs" in res.stdout
        assert os.path.isfile(dpo_out)

        with open(dpo_out, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
        assert len(lines) == 10
