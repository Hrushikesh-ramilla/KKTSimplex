"""
KKTSimplex Command-Line Interface.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional
from .polytope import project_simplex_with_bounds
from .trust import evaluate_hostile_audit
from .gates import run_gate_command
from .proof import adjudicate_with_proof, generate_execution_proof
from .dpo_dataset import export_dpo_dataset_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="kkt_simplex",
        description="KKTSimplex: Verifiable Epistemic Calibration & Autonomous Execution Governance",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: audit
    audit_parser = subparsers.add_parser("audit", help="Adjudicate an adversarial audit against execution proof")
    audit_parser.add_argument("--passed", action="store_true", help="Whether deterministic tests passed")
    audit_parser.add_argument("--critique", type=str, required=True, help="Auditor critique text")
    audit_parser.add_argument("--has-counter-test", action="store_true", help="Auditor provided failing test case")


    # Command: gate
    gate_parser = subparsers.add_parser("gate", help="Run a deterministic execution gate")
    gate_parser.add_argument("cmd", nargs=argparse.REMAINDER, help="Command to execute")

    args = parser.parse_args()

    def _resolve_content(arg_val: Optional[str]) -> Optional[str]:
        if arg_val is None:
            return None
        if os.path.isfile(arg_val):
            try:
                with open(arg_val, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                return arg_val
        return arg_val

    if args.command == "audit":
        res = evaluate_hostile_audit(
            agent_verified_pass=args.passed,
            auditor_critique_text=args.critique,
            auditor_has_reproducible_test=args.has_counter_test,
        )
        print("=" * 60)
        print(f"KKTSIMPLEX AUDIT ADJUDICATION: {res.decision}")
        print("=" * 60)
        print(f"Agent Calibrated Trust:   {res.agent_trust:.4f}")
        print(f"Auditor Claim Trust:      {res.auditor_trust:.4f}")
        print(f"Verified Counter-Example: {res.verified_counter_example}")
        print(f"Rationale:                {res.rationale}")
        print("=" * 60)


    elif args.command == "gate":
        if not args.cmd:
            print("Error: No command specified to run.", file=sys.stderr)
            sys.exit(1)
        res = run_gate_command(args.cmd)
        print(f"Gate Return Code: {res.returncode}")
        print(f"Gate Success:     {res.is_success}")
        if res.stdout:
            print(f"Stdout:\n{res.stdout}")
        if res.stderr:
            print(f"Stderr:\n{res.stderr}")
        sys.exit(0 if res.is_success else 1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
