"""Opening a graph by path, its sections and core segments, the editor's versions and the GMS sphere. No knowledge
store, no gateway.

    python tests/test_graph_tools.py
"""
from __future__ import annotations

import json
import shutil
import socket
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from gmsjourney.s01_packs import loading as domain  # noqa: E402

POLICY = ROOT / "gmsjourney" / "policy"


def _outside_graph(gid: str = "pet_card_test") -> Path:
    """A compact graph outside the journey folders (a copy of the supplementary-card graph under another id)."""
    g = json.loads((POLICY / "supplementary_card" / "supplementary_card_kg.json").read_text())
    g["id"], g["title"] = gid, "Pet card (test)"
    d = Path(tempfile.mkdtemp(prefix="gms_outside_"))
    p = d / f"{gid}.json"
    p.write_text(json.dumps(g))
    return p


def _cleanup(pack: str) -> None:
    shutil.rmtree(POLICY / pack, ignore_errors=True)
    domain.set_current("charge_review")


# ----------------------------------------------------------------------------- drafted lexicon
def test_a_drafted_lexicon_passes_the_lexicon_checks_on_both_compact_graphs():
    import yaml
    from gmsjourney.s02_graph.compact import PolicyGraph
    from gmsjourney.s01_packs.drafting import draft_lexicon
    for pack, gfile in (("supplementary_card", "supplementary_card_kg.json"), ("charge_review_graph", "charge_review_kg.json")):
        d = Path(tempfile.mkdtemp(prefix="gms_lex_")) / pack
        d.mkdir()
        shutil.copy(POLICY / pack / gfile, d / "g.json")
        shutil.copy(POLICY / pack / "application.yaml", d / "application.yaml")
        lx = draft_lexicon(PolicyGraph.load(d / "g.json"))
        (d / "domain.yaml").write_text(yaml.safe_dump({"domain": {"id": pack + "_lx", "graph": "g.json", "title": "t"}, "lexicon": lx}))
        assert domain.lint(domain.load(d)) == [], pack
    domain.set_current("charge_review")


def test_shipped_journeys_need_no_drafted_wording():
    for pack in ("supplementary_card", "charge_review_graph"):
        assert domain.load(pack).lexicon_drafted == [], pack


def test_drafted_bot_lines_say_every_field_with_its_unit_and_pass_the_speech_check():
    import re
    import yaml
    from gmsjourney.s10_console import graphs as GR
    from gmsjourney.s01_packs.drafting import draft_bot_says
    for pack in ("supplementary_card", "charge_review_graph"):
        g = domain.load(pack).graph
        says = draft_bot_says(g)
        for st, s in g.states.items():
            if g.kind(st) in ("call", "each"):
                continue
            assert st in says, (pack, st)
            for f in (s.get("say") or {}):
                assert "{" + f + "}" in says[st], (pack, st, f)
        assert re.search(r"\{min_age\} years", says.get("EXPLAIN_AGE_LIMIT", "{min_age} years"))
        # the same journey with every bot line drafted: the speech check passes on every turn of every case
        src = POLICY / pack
        dst = POLICY / f"{pack}_drafted_test"
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)
        try:
            cfg = yaml.safe_load((dst / "domain.yaml").read_text())
            cfg.setdefault("domain", {})["id"] = dst.name
            cfg.get("bot", {}).pop("says", None)
            cfg.get("bot", {}).pop("cues", None)    # cue words go with a pack's own wording, not the drafted lines
            (dst / "domain.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
            d = domain.set_current(dst.name)
            assert sorted(x for x in d.lexicon_drafted if x.startswith("bot.says.")) == sorted(f"bot.says.{k}" for k in says)
            seg = next(x for x in GR.core_segments(d) if x["id"] == "speech")
            assert seg["ok"] and seg["advisory"] and seg["drafted"]
            w = GR.check_wording()
            assert w["speech_failures"] == [] and w["off_flow"] == 0 and w["turns"] > 100, (pack, w)
        finally:
            _cleanup(dst.name)


# ----------------------------------------------------------------------------- open by path
def test_an_outside_graph_becomes_a_journey_that_plays_and_versions_without_touching_the_original():
    from gmsjourney.s10_console import graphs as GR
    from gmsjourney.s10_console import bots as PG
    src = _outside_graph()
    r = GR.open_graph(str(src))
    try:
        assert r["created"] and r["pack"] == "pet_card_test" and r["problems"] == []
        w = r["wording"]                                     # drafted wording, checked by playing the project's cases
        assert w["cases"] > 10 and w["turns"] > 50 and w["speech_failures"] == [] and w["off_flow"] == 0, w
        d = domain.set_current(r["pack"])
        assert d.name == "pet_card_test" and (d.dir / "project.yaml").exists() and (d.dir / "application.yaml").exists()
        L = PG.cases("project", None)
        case = L["cases"][0]
        s = PG.get(PG.start("project", case["id"], None)["id"])
        assert all(s.say(ln["text"])["match"] for ln in case["lines"])           # the drafted wording reads back
        # edit: a new relationship value, saved as a version beside the original
        orig = (d.dir / src.name).read_text()
        E = GR.editable(d)
        assert E["format"] == "compact" and {it["section"] for it in E["items"]} >= {"customer", "bot", "both", "flow", "rules", "knowledge"}
        items = E["items"]
        rel = next(it for it in items if it["path"] == "values.relationship")
        rel["value"]["GRANDPARENT"] = {"aliases": ["my grandmother", "my grandfather"], "allowed": "yes"}
        out = GR.save_version(d, {"items": items})
        assert out["file"] == "pet_card_test.v2.json" and out["problems"] == []
        assert [c["path"] for c in out["changes"]] == ["values.relationship.GRANDPARENT"] and out["changes"][0]["change"] == "added"
        assert (d.dir / src.name).read_text() == orig                             # the original is never written
        d2 = domain.set_current(r["pack"])
        assert d2.graph_path.name == "pet_card_test.v2.json"
        assert "values.GRANDPARENT" in d2.lexicon_drafted                          # worded by the draft until written
        assert [v["file"] for v in GR.versions(d2)] == ["pet_card_test.json", "pet_card_test.v2.json"]
        try:
            GR.save_version(d2, {"items": GR.editable(d2)["items"]})
            raise AssertionError("an unchanged graph is not saved")
        except ValueError:
            pass
        back = GR.open_graph(f"gmsjourney/policy/{r['pack']}/{src.name}")      # opening the original switches back
        assert back["switched"] and not back["created"] and domain.load(r["pack"]).graph_path.name == src.name
    finally:
        _cleanup(r["pack"])


def test_opening_refuses_what_is_not_a_compact_graph_from_outside():
    from gmsjourney.s10_console import graphs as GR
    d = Path(tempfile.mkdtemp(prefix="gms_bad_"))
    shutil.copy(POLICY / "charge_review" / "charge_review_kg.json", d / "kg.json")
    (d / "notes.json").write_text('{"hello": 1}')
    for name, words in (("kg.json", "triples graph"), ("notes.json", "not a policy graph"), ("missing.json", "no file")):
        try:
            GR.open_graph(str(d / name))
            raise AssertionError(name)
        except ValueError as e:
            assert words in str(e), (name, e)
    r = GR.open_graph(GR.DEFAULT_GRAPH)                     # the default: charge review as a policy graph
    assert r["pack"] == "charge_review_graph" and not r["created"]
    r = GR.open_graph("gmsjourney/policy/charge_review/charge_review_kg.json")   # the older full graph: its own journey
    assert r["pack"] == "charge_review" and not r["created"]


def test_a_triples_graph_versions_with_its_own_hash_and_a_user_edit_source():
    from gmsjourney.s10_console import graphs as GR
    from gmsjourney.s02_graph.triples import _content_hash
    d = domain.set_current("charge_review")
    E = GR.editable(d)
    rows = E["triples"]
    rows.append({"head": "max_turns", "relation": "has_text", "tail": "Transfer after the turn limit.", "section": "knowledge"})
    orig, doc, fmt = GR.build_doc(d, {"triples": rows, "note": "test"})
    assert fmt == "triples" and doc["sha256"] == _content_hash(doc) and "user_edit" in doc["source"]["sections"]
    ch = GR.diff(orig, doc, fmt)
    assert len(ch) == 1 and ch[0]["change"] == "added" and "has_text" in ch[0]["after"]
    assert GR.check_candidate(d, doc)["loads"]


# ----------------------------------------------------------------------------- sections and core segments
def test_sections_cover_every_triple_and_core_segments_find_what_is_missing():
    from gmsjourney.s10_console import graphs as GR
    for pack in ("charge_review", "supplementary_card"):
        d = domain.set_current(pack)
        S = GR.sections(d)
        assert sum(s["count"] for s in S["sections"]) == S["triples"] == len(d.kg.triples)
        by = {s["key"]: s for s in S["sections"]}
        assert by["customer"]["relations"].get("has_alias") or by["both"]["relations"].get("has_alias")
        assert "accepts_act" in by["both"]["relations"] and "leads_to" in by["flow"]["relations"]
        assert "has_stance" in by["rules"]["relations"] and "credential" in S["shared_slots"]
        assert all(s["ok"] for s in GR.core_segments(d)), [s for s in GR.core_segments(d) if not s["ok"]]
    d = domain.set_current("supplementary_card")
    data = json.loads(d.graph_path.read_text())
    del data["limits"]["max_turns"], data["global"]["turn_limit"]
    chk = GR.check_candidate(d, data)
    miss = {s["id"] for s in chk["segments"] if not s["ok"]}
    assert "turn_limit" in miss
    seg = next(s for s in chk["segments"] if s["id"] == "turn_limit")
    assert "limits.max_turns" in seg["add"] and "global.turn_limit" in seg["add"]
    domain.set_current("charge_review")


# ----------------------------------------------------------------------------- the sphere
def test_the_sphere_puts_every_entity_on_the_unit_sphere_and_keeps_linked_ones_close():
    import math
    from gmsjourney.s10_console import graphs as sphere
    d = domain.set_current("supplementary_card")
    B = sphere.build(d, runs_root=Path(tempfile.mkdtemp()))
    assert B["placement"] == "graph" and B["note"].startswith("No trained knowledge store")
    assert all(abs(math.sqrt(n["x"] ** 2 + n["y"] ** 2 + n["z"] ** 2) - 1) < 1e-3 for n in B["nodes"])
    ids = {n["id"] for n in B["nodes"]}
    assert all(e["s"] in ids and e["t"] in ids for e in B["edges"])
    assert B["fit"]["edge_arc"] < 0.6 * B["fit"]["random_arc"]
    assert sphere.build(d, runs_root=Path(tempfile.mkdtemp()))["nodes"] == B["nodes"]     # deterministic
    import numpy as np
    X = sphere.pca_sphere(np.random.default_rng(0).normal(size=(30, 64)))
    assert X.shape == (30, 3) and np.allclose(np.linalg.norm(X, axis=1), 1)
    domain.set_current("charge_review")


# ----------------------------------------------------------------------------- texts that match what they label
def test_every_playground_case_plays_the_level_its_label_names():
    from gmsjourney.s10_console import bots as PG
    from gmsjourney.s10_console.studio import Studio
    for pack in ("supplementary_card", "charge_review_graph", "charge_review"):
        domain.set_current(pack)
        PG.reset()
        for c in PG.cases("project", None)["cases"][1:]:
            name, lv = c["id"].split("=", 1)
            assert str(c["levels"][name]) == lv, (pack, c["id"], c["levels"][name])
    domain.set_current("charge_review_graph")
    PG.reset()
    cases = {c["id"]: c for c in PG.cases("project", None)["cases"]}
    said = lambda cid, st: [ln["frame"] for ln in cases[cid]["lines"] if ln["state"] == st]   # noqa: E731
    assert said("confirm_action=deny", "CONFIRM_ACTION") == ["DENY()"]          # a confirmation answers for the item in hand
    assert said("human_request=at_confirm_action", "CONFIRM_ACTION") == ["REQUEST_HUMAN()"]
    assert cases["human_request=at_confirm_action"]["expected"]["transfer_reason"] == "customer_request"
    assert "so the call gets there" in cases["human_request=at_ask_auth_retry"]["label"]
    assert cases["policy_question=ask"]["turns"] > cases["default"]["turns"]     # the case asks its question
    st = Studio(pack="supplementary_card")
    domain.set_current("supplementary_card")
    r = st.preview(st.project.raw, {"human_request": "at_ask_auth_retry"}, [])
    assert r["with"] == "authenticate_customer = retry_authenticated" and r["signature"]["transfer_reason"] == "customer_requested_human"
    assert st.preview(st.project.raw, {"authenticate_customer": "unavailable"}, [])["with"] is None
    domain.set_current("charge_review")


def test_tool_factors_say_what_varies():
    from gmsjourney.s01_packs.drafting import tool_about
    assert tool_about("authenticate_customer", "Verifies the caller.") == "what authenticate_customer returns (verifies the caller)"
    assert tool_about("submit_card_action", "submit_card_action | After customer confirms | SUCCESS / FAILED | Claim only after SUCCESS.") \
        == "what submit_card_action returns (after customer confirms: SUCCESS / FAILED)"
    import yaml
    for pack in ("supplementary_card", "charge_review_graph"):
        P = yaml.safe_load((POLICY / pack / "project.yaml").read_text())
        for f in P["factors"]:
            if (f.get("anchor") or {}).get("tool") == f["name"]:
                assert f["about"].startswith(f"what {f['name']} returns"), (pack, f["about"])


# ----------------------------------------------------------------------------- the critique layer
def _port() -> int:
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        return so.getsockname()[1]


def test_graph_api_open_sections_edit_check_save_use_sphere():
    from gmsjourney.s10_console import server
    port = _port()
    srv = server.make_server(port, runs_dir=tempfile.mkdtemp())
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}/"

    def call(path, body=None):
        req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    src = _outside_graph("api_card_test")
    try:
        code, r = call("api/graph/open", {"path": str(src)})
        assert code == 200 and r["created"] and r["pack"] == "api_card_test", r
        code, meta = call("api/meta")
        assert any(p["name"] == "api_card_test" and p["path"].endswith("api_card_test.json") for p in meta["packs"])
        code, S = call("api/graph/api_card_test/sections")
        assert code == 200 and len(S["sections"]) == 7
        assert all(s["ok"] for s in S["segments"]), [s for s in S["segments"] if not s["ok"]]
        code, E = call("api/graph/api_card_test/edit")
        items = E["items"]
        next(it for it in items if it["path"] == "limits.max_auth_attempts")["value"]["value"] = 3
        code, C = call("api/graph/api_card_test/check", {"items": items})
        assert code == 200 and C["loads"] and [c["path"] for c in C["changes"]] == ["limits.max_auth_attempts.value"]
        code, V = call("api/graph/api_card_test/save", {"items": items})
        assert code == 200 and V["file"] == "api_card_test.v2.json"
        code, S2 = call("api/graph/api_card_test/sections")
        assert [v["file"] for v in S2["versions"] if v["active"]] == ["api_card_test.v2.json"]
        code, U = call("api/graph/api_card_test/use", {"file": "api_card_test.json"})
        assert code == 200 and U["switched"]
        code, bad = call("api/graph/api_card_test/use", {"file": "../../x.json"})
        assert code == 400
        code, B = call("api/graph/api_card_test/sphere")
        assert code == 200 and B["nodes"] and B["edges"]
        code, err = call("api/graph/open", {"path": "nowhere/graph.json"})
        assert code == 400 and "no file" in err["error"]
    finally:
        srv.shutdown()
        _cleanup("api_card_test")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
