"""Core code spells no graph names of its own.

Scoring, checking, gating and case-building code refers to the policy graph's states, acts, tools, statuses,
values, limits, rules and triggers through the registry (``N.state.PRESENT_TXN``) or by role
(``roles.RECORD``), so a renamed or removed concept fails where it is used instead of silently matching nothing.

Modules that legitimately spell names are not "core" here:
  knowledge/*        the builders that write the graph from the PDF
  sut/*              the bots under test and the mock bank (independent of the harness by design)
  enrich/*           the customer's wording (phrase tables keyed by value), proved by the invariance check
  ontology.py        Turtle export descriptions;  pipeline.py  report examples;  flow.py  the generic walker
  kb_truth.py        the charge pack's policy-question engine: its intents are that graph's kb_intent values
                     (a compiled pack answers value / limit questions from its own graph, graph/kb.py)
A few strings in core modules coincide with graph names but are the trace's own labels (TRANSFER/END as a
component outcome, TRANSFERRED as an end state); they are listed in ALLOWED.

    python tests/test_names.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gmsjourney.names import graph_names  # noqa: E402

CORE = ["trace.py", "journey.py", "project.py", "studio.py", "narrate.py", "turn_compare.py",
        "export_dataset.py", "bot_validation.py", "roles.py", "domain.py",
        "speech/contract.py", "speech/verify.py", "speech/controls.py", "speech/reader.py", "speech/session.py",
        "governance/gates.py", "governance/monitor.py", "governance/attestor.py", "governance/calibrate.py",
        "evaluation/runner.py", "evaluation/severity.py", "evaluation/mutation.py", "evaluation/resilience.py",
        "bases/journey.py", "bases/kb.py", "reuse/governance.py"]
# (module, literal): strings that are the module's own labels, not references to the graph's name
ALLOWED = {("trace.py", "TRANSFER"), ("trace.py", "END"), ("trace.py", "TRANSFERRED"), ("narrate.py", "TRANSFERRED"),
           ("roles.py", "OPTIONS"), ("roles.py", "TRANSFER"),        # roles.py: role names listed in __all__
           # optional dialog states a pack may not have, compared by name (absent: never matches)
           *{("turn_compare.py", x) for x in ("ANSWER_KB", "DISCLOSE_FEE", "SUMMARIZE", "PRESENT_TXN", "ASK_CLARIFY",
                                              "PROVIDE_CONTEXT", "OPENING", "CONFIRM_RECORDED")},
           *{("speech/verify.py", x) for x in ("ASK_CLARIFY", "ANSWER_KB")},
           *{("journey.py", x) for x in ("ASK_CLARIFY", "ANSWER_KB")},
           ("journey.py", "lookback_days"),           # an optional limit read with getattr (absent: no window)
           # trace.py: a record call's own outcome labels and the domain.yaml key naming the turn limit
           ("trace.py", "SUCCESS"), ("trace.py", "FAILED"), ("trace.py", "turn_limit")}
# names too generic to be graph references when they appear (kb intents, stage words, attribute names)
GENERIC = {"value", "limit", "fee", "days", "end", "record", "capture", "retrieve", "transfer", "finish", "opening",
           "authenticate", "procedure", "refund", "fraud", "faster", "cheaper", "mcq", "plausibility", "delivery",
           "disposition", "card_action"}


def graph_literals(path: Path, names: dict[str, str]) -> list[tuple[int, str, str]]:
    tree = ast.parse(path.read_text())
    checks = {id(n.left) for n in ast.walk(tree) if isinstance(n, ast.Compare) and len(n.ops) == 1
              and isinstance(n.ops[0], (ast.In, ast.NotIn)) and "N." in ast.unparse(n.comparators[0])}
    docs = {id(n.body[0].value) for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body
            and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    return [(n.lineno, n.value, names[n.value]) for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in names and id(n) not in docs
            and id(n) not in checks]


def test_core_modules_spell_no_graph_names():
    D = graph_names()
    names = {n: k for k, v in D.all_names().items() for n in v if len(n) > 2 and n not in GENERIC}
    bad = []
    for rel in CORE:
        p = ROOT / "gmsjourney" / rel
        for line, lit, kind in graph_literals(p, names):
            if (rel, lit) not in ALLOWED:
                bad.append(f"gmsjourney/{rel}:{line}: {lit!r} ({kind}) - use N.{kind.split()[-1]}.{lit} or a role")
    assert not bad, "graph names spelled in core code:\n" + "\n".join(bad)


def test_roles_resolve_in_the_graph():
    from gmsjourney import roles
    D = graph_names()
    for r in ("GATE", "ITEMS", "RECORD", "OPTIONS", "ACTION", "TRANSFER"):
        assert getattr(roles, r) in D.tools, r
    assert roles.RECORD in D.write_tools and roles.ACTION in D.write_tools
    assert roles.KEY in D.tools[roles.RECORD].args and roles.FEE_OPTION in D.fee_values
    assert roles.ACCOUNT_TOOLS == frozenset(D.chain) and roles.GATE not in roles.ACCOUNT_TOOLS


def test_an_undefined_name_fails_where_it_is_used():
    from gmsjourney.names import N
    try:
        N.state.PRESENT_TX
    except AttributeError as e:
        assert "did you mean PRESENT_TXN" in str(e)
    else:
        raise AssertionError("a misspelled state was accepted")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
