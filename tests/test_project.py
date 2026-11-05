"""The project configuration: checked against the graph, editable in the studio, and the source of the factors.

Runs on the symbolic store (no GMS training, no licence needed).
"""
from __future__ import annotations

import copy
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402

from gmsjourney.names import GraphNames  # noqa: E402
from gmsjourney.knowledge.policy_kg import PolicyKG  # noqa: E402
from gmsjourney.project import Project, check, default_path  # noqa: E402
DEFAULT_PATH = default_path()

DOM = GraphNames.from_kg(PolicyKG.load())


def _with(edit) -> list[str]:
    raw = yaml.safe_load(Path(DEFAULT_PATH).read_text())
    edit(raw)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=Path(DEFAULT_PATH).parent) as f:
        f.write(yaml.safe_dump(raw, sort_keys=False))
    try:
        return check(Project.load(f.name), DOM)
    finally:
        Path(f.name).unlink()


def test_the_shipped_project_is_valid_against_the_graph():
    assert check(Project.load(), DOM) == []


def test_names_and_values_the_graph_does_not_define_are_caught():
    def f(raw):
        lv = raw["factors"][0]["levels"]
        lv["ok"]["fixture"]["auth_results"] = ["AUTHENTICATD"]                        # not a status
        lv["human_at_auth"]["customer"][0]["say"] = ["REQUEST_HUMANS()"]               # not an act
        raw["factors"][4]["levels"]["mixed"]["customer"][0]["say"] = ["INFORM_DISPOSITION(disposition=MAYBE)"]
        raw["factors"][1]["relevant_when"] = "level('auth') not in ('ok'"               # does not parse
        raw["customer"]["answers"]["PRESENT_TXNS"] = "capture"                          # not a state
        raw["generator"]["fault_only_rules"] = ["R99"]
    issues = _with(f)
    for needle in ("AUTHENTICATD", "REQUEST_HUMANS", "MAYBE", "cannot parse", "PRESENT_TXNS", "R99"):
        assert any(needle in i for i in issues), (needle, issues)


def test_a_frame_must_be_accepted_where_its_script_answers():
    issues = _with(lambda raw: raw["factors"][8]["levels"]["switch_standard"]["customer"][0].__setitem__(
        "say", ["INFORM_DISPOSITION(disposition=RECOGNIZED)"]))
    assert any("not accepted in CONFIRM_ACTION" in i for i in issues), issues


def test_studio_preview_check_and_lossless_save():
    from gmsjourney.studio import Studio
    tmp = Path(tempfile.mkdtemp()) / "project.yaml"
    tmp.write_text(Path(DEFAULT_PATH).read_text())
    st = Studio(str(tmp))
    state = st.state()
    assert state["issues"] == [] and state["domain"]["summary"]["states"] == len(DOM.states)
    pv = st.preview(state["project"], {"n_txn": "2", "disposition": "mixed", "card_intent": "replace_expedited"},
                    ["ASK_KB(intent=value, attr=fee, entity=expedited_delivery)"])
    assert pv["valid"] and pv["signature"]["action"] == ["REPLACE_CARD", "expedited_delivery"]
    assert any(t["role"] == "bot" and any(a["act"] == "ANSWER_KB" for a in t["acts"]) for t in pv["conversation"])
    raw = copy.deepcopy(state["project"])
    import json
    raw = json.loads(json.dumps(raw))                       # what the browser sends back (integer keys become strings)
    raw["factors"][0]["about"] = "edited"
    for f in raw["factors"]:
        for lv in f["levels"].values():
            lv.setdefault("fixture", {})
            lv.setdefault("customer", [])
    r = st.save(raw)
    assert r["issues"] == []
    a, b = yaml.safe_load(Path(DEFAULT_PATH).read_text()), yaml.safe_load(tmp.read_text())
    b["factors"][0]["about"] = a["factors"][0]["about"]
    assert a == b
    assert tmp.with_suffix(".yaml.bak").exists()


def test_a_new_level_needs_no_code():
    """A level the project adds (the customer declines the card action) is played by walking the graph."""
    from gmsjourney import journey
    from gmsjourney.flow import GraphOracle
    from gmsjourney.studio import symbolic_store
    raw = yaml.safe_load(Path(DEFAULT_PATH).read_text())
    ec = next(f for f in raw["factors"] if f["name"] == "expedited_confirm")
    ec["levels"]["decline"] = {"customer": [{"script": "card", "say": ["DENY()"]}]}
    tmp = Path(tempfile.mkdtemp()) / "project.yaml"
    tmp.write_text(yaml.safe_dump(raw, sort_keys=False))
    proj = Project.load(tmp)
    assert check(proj, DOM) == []
    store = symbolic_store(proj.policy_graph)
    try:
        journey.activate(proj)
        spec = journey.JourneySpec(n_txn=2, disposition="mixed", card_intent="replace_expedited",
                                   expedited_confirm="decline")
        tr = GraphOracle(store).run(spec)
        acts = [a["act"] for e in tr.events if e["role"] == "bot" for a in e["acts"]]
        assert "ACK_DECLINED" in acts and tr.completed_action() is None and tr.end_state == "ENDED"
        assert ("CONFIRM_ACTION", "ACK_DECLINED", "DENY") in tr.flow_path
    finally:
        journey.activate(Project.load())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
