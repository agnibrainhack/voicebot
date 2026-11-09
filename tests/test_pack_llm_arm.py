"""The LLM enrichment arm on a compiled pack (supplementary card), with a scripted stand-in for the model.

The stand-in rewrites each turn as its base wording, reads an utterance back with the rule reader, and for an
adversarial control swaps one value word for another member of the same value set. The arm must prove every
scenario separately from the template arm, reject every control, and the runner must send the proved text.

    python tests/test_pack_llm_arm.py
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gmsjourney.packs import loading as domain  # noqa: E402


class _FakeChat:
    def __init__(self, role, reader, swaps):
        self.role, self.reader, self.swaps = role, reader, swaps

    def invoke(self, messages, **_kw):
        u = json.loads(messages[-1] if isinstance(messages[-1], str) else messages[-1].content)
        if self.role == "rephrase":
            return SimpleNamespace(content=json.dumps({"utterance": u["base_wording"]}))
        if self.role == "extract":
            g = self.reader.read(u["utterance"], None)
            return SimpleNamespace(content=json.dumps(
                {"act": g.act, "slots": {k: (list(v) if isinstance(v, tuple) else v) for k, v in g.slots}}))
        text = u["original_utterance"]
        for a, b in self.swaps:
            if re.search(rf"\b{re.escape(a)}\b", text, re.I):
                return SimpleNamespace(content=json.dumps({"utterance": re.sub(rf"\b{re.escape(a)}\b", b, text,
                                                                               count=1, flags=re.I)}))
        return SimpleNamespace(content=json.dumps({"utterance": text}))


def _fixture():
    dom = domain.set_current("supplementary_card")
    from gmsjourney.rewording.catalog import Catalog, resolve_profile
    from gmsjourney.rewording.catalog import Scenario
    from gmsjourney.cases.design import compose
    from gmsjourney.llm import LLMSettings
    from gmsjourney.rewording.llm_proof import build_llm_arm
    from gmsjourney import llm as llm_client
    from gmsjourney.rewording.wording import FrameReader
    from gmsjourney.rewording.wording import Realizer
    from gmsjourney.cases.design import to_item
    from gmsjourney.rewording.proof import GraphInvarianceValidator
    from gmsjourney.cases.journey import JourneySpec
    from gmsjourney.knowledge.store import pack_oracle, pack_store
    store, oracle = pack_store(), pack_oracle()
    llm_client._messages = lambda s, u, **_kw: [s, u]
    R, rd = Realizer(), FrameReader(store)
    specs = [JourneySpec(), JourneySpec(ask_card_choice_card_product="premium_supp"),
             JourneySpec(ask_relationship_relationship="friend"), JourneySpec(confirm_application="deny")]
    bases = [to_item(s, oracle, f"J{i:03d}") for i, s in enumerate(specs)]
    ls = resolve_profile(Catalog.load(), "voice_full").level_specs()
    scns = compose(ls, bases, 8, 42, mode="embedded", assign="dispersed")
    tv = GraphInvarianceValidator(store, oracle, rd, R)
    [tv.prove(s) for s in scns]
    realizer, lv, clients = build_llm_arm(store, oracle, rd, tv, LLMSettings(cache_dir=tempfile.mkdtemp()))
    g = dom.graph
    swaps = [(a, b) for vs in g.values for a, b in zip(g.members(vs), g.members(vs)[1:] + g.members(vs)[:1])]
    lexv = dom.get("lexicon.values") or {}
    words = [((lexv.get(a) or {}).get("say") or {}).get("canonical") for a, _ in swaps]
    others = [((lexv.get(b) or {}).get("say") or {}).get("canonical") for _, b in swaps]
    swaps = [(w, o) for w, o in zip(words, others) if w and o and w != o]
    for role, c in clients.items():
        if c is not None:
            c.model = _FakeChat(role, rd, swaps)
    llm_scns = [Scenario(f"{s.sid}~llm", s.base, dict(s.factor_levels), s.mode, {"arm": "llm"}) for s in scns]
    return store, oracle, R, rd, tv, realizer, lv, llm_scns, llm_client


def test_llm_arm_runs_on_a_compiled_pack():
    from gmsjourney import llm as client
    real = client._messages
    try:
        store, oracle, R, rd, tv, realizer, lv, llm_scns, _ = _fixture()
        proofs = [lv.prove(s) for s in llm_scns]
        assert all(p.arm == "llm" for p in proofs)
        assert sum(p.accepted for p in proofs) >= len(proofs) - 1, [(p.sid, p.accepted) for p in proofs]
        ctrl = lv.negative_controls(proofs, llm_scns, per_type=10, adversarial=4)
        det = {k: v["detection_rate"] for k, v in ctrl.items() if not k.startswith("_") and v.get("n")}
        assert det and all(r == 1.0 for r in det.values()), det
        from gmsjourney.checks.scoring import run
        from gmsjourney.bots.reference import make_graph_factory
        from gmsjourney.bots.harness import JourneyRunner
        runner = JourneyRunner(store=store, realizer=R, reader=rd, bot_factory=make_graph_factory(store),
                               arm_realizers={"llm": realizer})
        rows = run(llm_scns, runner, oracle.limits, store.workflow, lv, show=False)
        assert rows and all(r["arm"] == "llm" for r in rows) and realizer.stats["pinned_hits"] > 0
    finally:
        client._messages = real
        domain.set_current("charge_review")


if __name__ == "__main__":
    test_llm_arm_runs_on_a_compiled_pack()
    print("ok test_llm_arm_runs_on_a_compiled_pack")
