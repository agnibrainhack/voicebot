"""Confirm, on an installed knowlytix (or a gmsh checkout), the harness defects listed in
UPSTREAM_GMSH_ISSUES.md before they are filed.

    python tools/gmsh_upstream_repro.py                      # the installed knowlytix package
    python tools/gmsh_upstream_repro.py --root ~/src/gmsh    # a gmsh checkout (repo root or knowlytix/)

Two kinds of check:

* static  - reads the Apache-2.0 harness / benchmark source files as text (nothing is imported
            or executed) and looks for the exact construct that causes the defect; "present"
            means this installation still has it, "not found" means the code differs (perhaps
            fixed) and the issue should be re-read before filing;
* dynamic - where the open module imports cleanly, a three-line demonstration (an empty answer
            passing the exact-outcome oracle; a blocking PolicyDecision that is truthy).

Only open-source (Apache-2.0) knowlytix packages are read. knowlytix-core is never opened.
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

CHECKS = [
    # id, relative path under knowlytix/, regex that must match for the defect to be present, summary
    ("G1", "harness/testing/oracles.py", r"exact_match\s*=\s*expected in actual or actual in expected",
     "ExactOutcomeOracle: substring test in both directions - an empty answer passes"),
    ("G2", "harness/testing/multiturn.py", r"tension_energy\(\s*entity\s*,\s*entity\s*\)",
     "Cross-turn consistency compares each entity with itself - the check cannot fail"),
    ("G3", "harness/testing/gateway.py", r"allowed\s*=\s*self\.policy_engine\.check\(",
     "ToolGateway treats the PolicyDecision object as a bool - the policy engine never blocks a call"),
    ("G4", "harness/testing/policy.py", r'elif rule\.rule_type == "requires":\s*\n\s*if rule\.severity == "error":\s*\n\s*blocking_rules\.append',
     "A 'requires' rule of severity error always blocks; whether the requirement is met is never checked"),
    ("G5", "harness/testing/oracles.py", r"return self\._store\.check_holonomy\(expected, actual\)",
     "TraceOracle holonomy passes two lists where a relation is expected; the error is swallowed (always None)"),
    ("G6", "benchmark/eval/generators/base.py", r"def __init__\(self, geometric: bool = False\)",
     "Two-way (geometric) admission is off by default"),
    ("G7", "harness/testing/bridge.py", r"gens = geometric_generators\(\) \+ taxonomy_generators\(\)",
     "The default bridge never turns the geometric admission on"),
    ("G8", "harness/suite/emit.py", r"def _default_materialize\(s: Scenario\) -> str:[\s\S]{0,120}return s\.base\.query",
     "Suite default materializer is the identity: factor levels never reach the system under test"),
    ("G9", "harness/suite/evaluate.py", r"sut\(scn\.base\.query",
     "Suite evaluate.run sends the unenriched base query"),
    ("G10", "harness/graphdoe/question_rephraser.py", r"def _with_nl\([\s\S]{0,400}natural_language=new_nl[\s\S]{0,200}llm_prompt=q\.llm_prompt",
     "Template rephraser: several factors change only natural_language, while the SUT receives llm_prompt"),
    ("G11", "harness/graphdoe/question_rephraser.py", r"\*\*Ground truth:\*\* \{ground_truth\}",
     "LLM rephraser prompt contains the ground truth; the output is checked only for JSON and length"),
    ("G12", "harness/testing/cross_document.py", r"def __init__\(\s*self, stores[^)]*\):\s*\n\s*self\.additional_stores",
     "CrossDocumentGenerator.__init__ never calls super(); self.geometric is missing, every question is rejected"),
    ("G13", "harness/testing/test_generators.py", r"ground_truth = False\s*\n\s*\n\s*def _answer_fn\(s, _fake=fake_entity, _store=store\):\s*\n\s*return not _store\.entity_exists\(_fake\)",
     "NegativeGenerator: the answer function returns True while the stated ground truth is False - zero questions"),
    ("G14", "harness/core.py", r'return "Tool call skipped by policy hook\."',
     "An injected ToolGateway error reaches the agent as a policy skip and is absent from the trace"),
    ("G15", "harness/testing/campaign.py", r'n_questions=getattr\(result, "n_questions"',
     "Campaign summary reads n_questions / n_correct, which DOEBenchmarkResult does not have (total_*)"),
]


def _root(arg: str | None) -> Path:
    if arg:
        p = Path(arg).expanduser().resolve()
        return p / "knowlytix" if (p / "knowlytix").is_dir() else p
    spec = importlib.util.find_spec("knowlytix")              # locates the package without importing it
    if spec is None or not spec.submodule_search_locations:
        sys.exit("knowlytix is not installed; pass --root <gmsh checkout>")
    return Path(list(spec.submodule_search_locations)[0])


def static_checks(root: Path) -> list[dict]:
    out = []
    for cid, rel, pat, summary in CHECKS:
        f = root / rel
        if not f.exists():
            out.append({"id": cid, "file": rel, "status": "file missing", "line": None, "summary": summary})
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        m = re.search(pat, text)
        line = text[:m.start()].count("\n") + 1 if m else None
        out.append({"id": cid, "file": rel, "status": "present" if m else "not found", "line": line, "summary": summary})
    return out


def dynamic_checks() -> list[dict]:
    out = []
    try:
        from knowlytix.harness.testing.oracles import ExactOutcomeOracle
        v = ExactOutcomeOracle(None).evaluate(SimpleNamespace(ground_truth="15"), SimpleNamespace(raw_output=""))
        out.append({"id": "G1", "demo": "empty answer vs ground truth '15'", "passed": bool(getattr(v, "passed", None)),
                    "defect_present": bool(getattr(v, "passed", False))})
    except Exception as e:                                     # noqa: BLE001 - demo only
        out.append({"id": "G1", "demo": "skipped", "reason": f"{type(e).__name__}: {str(e)[:120]}"})
    try:
        from knowlytix.harness.testing.policy import PolicyDecision
        d = PolicyDecision(allowed=False)
        out.append({"id": "G3", "demo": "bool(PolicyDecision(allowed=False))", "value": bool(d),
                    "defect_present": bool(d)})
    except Exception as e:                                     # noqa: BLE001
        out.append({"id": "G3", "demo": "skipped", "reason": f"{type(e).__name__}: {str(e)[:120]}"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="gmsh checkout or knowlytix package directory")
    ap.add_argument("--no-dynamic", action="store_true")
    a = ap.parse_args()
    root = _root(a.root)
    print(f"knowlytix source: {root}\n")
    rows = static_checks(root)
    w = max(len(r["file"]) for r in rows) + 6
    for r in rows:
        loc = f"{r['file']}:{r['line']}" if r["line"] else r["file"]
        print(f"{r['id']:<4} {r['status']:<12} {loc:<{w}} {r['summary']}")
    if not a.no_dynamic and not a.root:
        print("\ndynamic demonstrations:")
        for d in dynamic_checks():
            print("  ", d)
    n = sum(r["status"] == "present" for r in rows)
    print(f"\n{n}/{len(rows)} defects present in this source; file only those marked 'present'.")


if __name__ == "__main__":
    main()
