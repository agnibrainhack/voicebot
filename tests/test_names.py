"""Core code spells no graph names of its own.

Scoring, checking, gating and case-building code refers to the policy graph's states, acts, tools, statuses,
values, limits, rules and triggers through the registry (``N.state.PRESENT_TXN``) or by role
(``roles.RECORD``), so a renamed or removed concept fails where it is used instead of silently matching nothing.

Modules that legitimately spell names are not "core" here:
  graph/*, knowledge/build.py   the builders that write the graph from the PDF
  bots/*             the bots under test and the mock bank (independent of the harness by design)
  rewording/*        the customer's wording (phrase tables keyed by value), proved by the invariance check
  cases/meaning.py   Turtle export descriptions;  campaign/pipeline.py  report examples;  graph/flow.py  the walker
  knowledge/questions.py part 1   the charge pack's policy-question engine: its intents are that graph's
                     kb_intent values
                     (a compiled pack answers value / limit questions from its own graph, knowledge/questions.py)
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

from gmsjourney.packs.names import graph_names  # noqa: E402

# (file, part): the core code checked. A merged file has numbered parts ("# PART 2 - ..."); None: the whole file.
CORE = [("bots/tools.py", 2), ("cases/journey.py", None), ("packs/project.py", None), ("console/studio.py", None),
        ("reports/cases.py", 1), ("reports/cases.py", 2), ("campaign/outputs.py", 3), ("checks/diagnosis.py", 5),
        ("packs/names.py", 2), ("packs/loading.py", 1),
        ("checks/speech_reading.py", 1), ("checks/speech_reading.py", 2), ("checks/speech_reading.py", 3),
        ("checks/speech_checks.py", 1), ("checks/speech_policy.py", 2),
        ("checks/gates.py", 1), ("checks/gates.py", 2), ("checks/runtime.py", 2), ("checks/runtime.py", 3),
        ("checks/runtime.py", 5), ("checks/scoring.py", 2), ("checks/scoring.py", 3),
        ("checks/diagnosis.py", 3), ("checks/diagnosis.py", 4), ("cases/design.py", 2), ("knowledge/questions.py", 2)]
TRACE, NARRATE, TURNS, NAMES = ("bots/tools.py", 2), ("reports/cases.py", 1), ("reports/cases.py", 2), ("packs/names.py", 2)
JOURNEY, SPEECH = ("cases/journey.py", None), ("checks/speech_checks.py", 1)
# (part, literal): strings that are the code's own labels, not references to the graph's name
ALLOWED = {(TRACE, "TRANSFER"), (TRACE, "END"), (TRACE, "TRANSFERRED"), (NARRATE, "TRANSFERRED"),
           (NAMES, "OPTIONS"), (NAMES, "TRANSFER"),        # the roles part: role names listed in __all__
           # optional dialog states a pack may not have, compared by name (absent: never matches)
           *{(TURNS, x) for x in ("ANSWER_KB", "DISCLOSE_FEE", "SUMMARIZE", "PRESENT_TXN", "ASK_CLARIFY",
                                  "PROVIDE_CONTEXT", "OPENING", "CONFIRM_RECORDED")},
           *{(SPEECH, x) for x in ("ASK_CLARIFY", "ANSWER_KB")},
           *{(JOURNEY, x) for x in ("ASK_CLARIFY", "ANSWER_KB")},
           (JOURNEY, "lookback_days"),           # an optional limit read with getattr (absent: no window)
           # the trace part: a record call's own outcome labels and the domain.yaml key naming the turn limit
           (TRACE, "SUCCESS"), (TRACE, "FAILED"), (TRACE, "turn_limit")}
# names too generic to be graph references when they appear (kb intents, stage words, attribute names)
GENERIC = {"value", "limit", "fee", "days", "end", "record", "capture", "retrieve", "transfer", "finish", "opening",
           "authenticate", "procedure", "refund", "fraud", "faster", "cheaper", "mcq", "plausibility", "delivery",
           "disposition", "card_action"}


def part_source(path: Path, part: int | None) -> tuple[str, int]:
    """The source of one part of a merged file (and the line it starts on), or the whole file."""
    text = path.read_text()
    if part is None:
        return text, 1
    lines = text.splitlines(keepends=True)
    starts = [i for i, ln in enumerate(lines) if ln.startswith("# PART ")]
    i = starts[part - 1]
    j = starts[part] if part < len(starts) else len(lines)
    return "".join(lines[i:j]), i + 1


def graph_literals(path: Path, names: dict[str, str], part: int | None = None) -> list[tuple[int, str, str]]:
    src, first = part_source(path, part)
    tree = ast.parse(src)
    checks = {id(n.left) for n in ast.walk(tree) if isinstance(n, ast.Compare) and len(n.ops) == 1
              and isinstance(n.ops[0], (ast.In, ast.NotIn)) and "N." in ast.unparse(n.comparators[0])}
    docs = {id(n.body[0].value) for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body
            and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    return [(n.lineno + first - 1, n.value, names[n.value]) for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in names and id(n) not in docs
            and id(n) not in checks]


def test_core_modules_spell_no_graph_names():
    D = graph_names()
    names = {n: k for k, v in D.all_names().items() for n in v if len(n) > 2 and n not in GENERIC}
    bad = []
    for rel, part in CORE:
        p = ROOT / "gmsjourney" / rel
        for line, lit, kind in graph_literals(p, names, part):
            if ((rel, part), lit) not in ALLOWED:
                bad.append(f"gmsjourney/{rel}:{line}: {lit!r} ({kind}) - use N.{kind.split()[-1]}.{lit} or a role")
    assert not bad, "graph names spelled in core code:\n" + "\n".join(bad)


def test_roles_resolve_in_the_graph():
    from gmsjourney.packs import names as roles
    D = graph_names()
    for r in ("GATE", "ITEMS", "RECORD", "OPTIONS", "ACTION", "TRANSFER"):
        assert getattr(roles, r) in D.tools, r
    assert roles.RECORD in D.write_tools and roles.ACTION in D.write_tools
    assert roles.KEY in D.tools[roles.RECORD].args and roles.FEE_OPTION in D.fee_values
    assert roles.ACCOUNT_TOOLS == frozenset(D.chain) and roles.GATE not in roles.ACCOUNT_TOOLS


def test_an_undefined_name_fails_where_it_is_used():
    from gmsjourney.packs.names import N
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
