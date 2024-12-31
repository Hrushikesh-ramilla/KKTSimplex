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

    # Command: certify
    certify_parser = subparsers.add_parser("certify", help="Execute code and generate a cryptographic execution certificate")
    certify_parser.add_argument("--code", type=str, required=True, help="Python code string or path to code file")
    certify_parser.add_argument("--test", type=str, required=True, help="Python test string or path to test file")
    certify_parser.add_argument("--critique", type=str, default="", help="Auditor critique text")
    certify_parser.add_argument("--counter-test", type=str, default=None, help="Auditor alleged counter-test or path to test file")
    certify_parser.add_argument("--out", type=str, default=None, help="Output path for certificate JSON")

    # Command: export-dpo
    dpo_parser = subparsers.add_parser("export-dpo", help="Export synthetic anti-sycophancy DPO dataset pairs")
    dpo_parser.add_argument("--out", type=str, required=True, help="Output JSONL file path")
    dpo_parser.add_argument("--num-samples", type=int, default=50, help="Number of DPO pairs to generate")

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

    elif args.command == "certify":
        code_str = _resolve_content(args.code) or ""
        test_str = _resolve_content(args.test) or ""
        counter_str = _resolve_content(args.counter_test)
        critique_str = args.critique or "Adversarial critique on correctness"

        cert = adjudicate_with_proof(
            agent_code=code_str,
            test_suite=test_str,
            auditor_critique=critique_str,
            alleged_failing_test=counter_str,
        )
        print("=" * 60)
        print(f"KKTSIMPLEX EXECUTION PROOF CERTIFICATION: {cert.adjudication}")
        print("=" * 60)
        print(f"Agent Trust:          {cert.agent_trust:.4f}")
        print(f"Auditor Trust:        {cert.auditor_trust:.4f}")
        print(f"Posterior Odds Ratio: {cert.posterior_odds_ratio:.4f}")
        print(f"KKT Optimality Valid: {cert.kkt_proof.is_valid}")
        print(f"Execution Proven:     {cert.execution_proof.is_proven if cert.execution_proof else False}")
        print(f"Summary:              {cert.formal_verdict_statement}")
        print("=" * 60)

        if args.out:
            cert_dict = {
                "adjudication": cert.adjudication,
                "agent_trust": cert.agent_trust,
                "auditor_trust": cert.auditor_trust,
                "posterior_odds_ratio": cert.posterior_odds_ratio,
                "kkt_proof": {
                    "is_valid": cert.kkt_proof.is_valid,
                    "primal_feasibility_error": cert.kkt_proof.primal_feasibility_error,
                    "stationarity_error": cert.kkt_proof.stationarity_error,
                    "complementarity_error": cert.kkt_proof.complementarity_error,
                    "dual_feasibility_passed": cert.kkt_proof.dual_feasibility_passed,
                },
                "execution_proof": {
                    "is_proven": cert.execution_proof.is_proven if cert.execution_proof else False,
                    "exit_code": cert.execution_proof.exit_code if cert.execution_proof else -1,
                    "assertions_verified": cert.execution_proof.assertions_verified if cert.execution_proof else 0,
                    "alleged_counter_example_refuted": cert.execution_proof.alleged_counter_example_refuted if cert.execution_proof else False,
                } if cert.execution_proof else None,
                "formal_verdict_statement": cert.formal_verdict_statement,
            }
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(cert_dict, f, indent=2)
            print(f"Certificate saved to {args.out}")

    elif args.command == "export-dpo":
        export_dpo_dataset_jsonl(args.out, num_samples=args.num_samples)
        print(f"Exported {args.num_samples} anti-sycophancy DPO pairs to {args.out}")

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
