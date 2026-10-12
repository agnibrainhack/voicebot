"""Checks for the 0.4 additions: hallucination probes, the LLM arm's separate proof (with a fake
LangChain-shaped model, no network), quorum, the numeric attestor, budgets, severity, operating points.

Run with `pytest -q tests/` or `python tests/test_parity_and_llm_arm.py`.
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gmsjourney.enrich.llm.client as llm_client  # noqa: E402
from gmsjourney.bases.journey import to_item  # noqa: E402
from gmsjourney.bases.kb import mine_kb_items  # noqa: E402
from gmsjourney.catalog import Catalog, resolve_profile  # noqa: E402
from gmsjourney.core import Scenario  # noqa: E402
from gmsjourney.design.doe import compose  # noqa: E402
from gmsjourney.enrich.invariance import InvarianceValidator  # noqa: E402
from gmsjourney.enrich.reader import FrameReader  # noqa: E402
from gmsjourney.enrich.realizers import Realizer  # noqa: E402
from gmsjourney.journey import JourneySpec  # noqa: E402
from gmsjourney.kb_truth import judge_kb_answer  # noqa: E402
from gmsjourney.knowledge.geode import build_store  # noqa: E402
from gmsjourney.ontology import Ontology, SemanticFrame as F  # noqa: E402
from gmsjourney.oracle import PolicyOracle  # noqa: E402

POLICY_KG = Path(__file__).resolve().parents[1] / "gmsjourney" / "policy" / "charge_review_kg.json"


@lru_cache(maxsize=1)
def env():
    store = build_store(POLICY_KG).store
    return store, PolicyOracle(store), Realizer(), FrameReader(store)


# ----------------------------------------------------------------------------- probes
def test_hallucination_probes_read_back_and_are_abstentions():
    store, _o, R, rd = env()
    ont = Ontology.from_store(store)
    items, info = mine_kb_items(store, 42)
    probes = [i for i in items if i.probe]
    assert info["hallucination_probes"]["by_kind"].get("near_miss", 0) >= 2
    assert all(i.answer == "absent" and i.admission != "rejected" for i in probes)
    for it in probes:
        for lv in ({}, {"entity_aliasing": "colloquial", "noise": "typo"}, {"style": "terse"},
                   {"question_format": "declarative", "noise": "asr"}, {"answer_anchoring": "wrong_suggestion"}):
            text = R.realize(it.frame, lv, "t", {"merchant": "Amazon", "amount": "45.67"})
            assert ont.equivalent(it.frame, rd.read(text, "PRESENT_TXN"), "PRESENT_TXN"), (lv, text)


def test_reference_bot_abstains_and_the_mutant_fabricates():
    from gmsjourney.sut.reference_bot import DEFAULT_CONFIG, KBAnswerer
    store, _o, R, _rd = env()
    items, _ = mine_kb_items(store, 42)
    sents = [s for _, s in store.kb_sentences]
    for muts, want in ((set(), True), ({"kb_fabricate_unknown"}, False)):
        bot = KBAnswerer(sents, dict(DEFAULT_CONFIG), muts)
        for it in (i for i in items if i.probe):
            assert judge_kb_answer(store, it.frame, bot.answer(R.realize(it.frame, {}, "b")))["correct"] is want


# ----------------------------------------------------------------------------- LLM arm (fake model)
class _FakeChat:
    """LangChain-shaped: invoke(messages) -> object with .content. Rephraser echoes the base wording."""

    def __init__(self, role, reader):
        self.role, self.reader = role, reader

    def invoke(self, msgs):
        _system, user = msgs
        u = json.loads(user)
        if self.role == "rephrase":
            return SimpleNamespace(content=json.dumps({"utterance": u["base_wording"]}))
        if self.role == "extract":
            g = self.reader.read(u["utterance"], None)
            return SimpleNamespace(content=json.dumps(
                {"act": g.act, "slots": {k: (list(v) if isinstance(v, tuple) else v) for k, v in g.slots}}))
        return SimpleNamespace(content=json.dumps({"utterance": re.sub(r"\bI recognize\b", "I don't recognize",
                                                                       u["original_utterance"])}))


def _llm_fixture():
    store, oracle, R, rd = env()
    llm_client._messages = lambda s, u, **_kw: [s, u]    # no langchain needed for the fake model
    from gmsjourney.enrich.llm import LLMSettings, build_llm_arm
    specs = [JourneySpec(), JourneySpec(disposition="mixed", card_intent="replace_expedited"),
             JourneySpec(auth="retry_ok", n_txn=2)]
    bases = [to_item(s, oracle, f"J{i:03d}") for i, s in enumerate(specs)]
    ls = resolve_profile(Catalog.load(), "voice_full").level_specs()
    scns = compose(ls, bases, 6, 42, mode="embedded", assign="dispersed")
    tv = InvarianceValidator(store, oracle, rd, R)
    [tv.prove(s) for s in scns]
    realizer, lv, clients = build_llm_arm(store, oracle, rd, tv, LLMSettings(cache_dir=tempfile.mkdtemp()))
    for role, c in clients.items():
        if c is not None:
            c.model = _FakeChat(role, rd)
    llm_scns = [Scenario(f"{s.sid}~llm", s.base, dict(s.factor_levels), s.mode, {"arm": "llm"}) for s in scns]
    return store, oracle, R, rd, ls, tv, realizer, lv, llm_scns


def test_llm_arm_is_proved_separately_and_controls_are_rejected():
    # _llm_fixture() rebinds llm_client._messages for the fake model; restore it afterwards so
    # the patch does not leak into whichever test runs next (it used to).
    real_messages = llm_client._messages
    try:
        _llm_arm_proof_and_controls()
    finally:
        llm_client._messages = real_messages


def _llm_arm_proof_and_controls():
    store, oracle, R, rd, ls, tv, realizer, lv, llm_scns = _llm_fixture()
    proofs = [lv.prove(s) for s in llm_scns]
    assert all(p.arm == "llm" for p in proofs)
    assert all(p.accepted for p in proofs)                         # base wording always reads back
    assert all(t["gates"]["G2_reader"] and t["gates"]["G1_facts"] for p in proofs for t in p.turns)
    ctrl = lv.negative_controls(proofs, llm_scns, per_type=20, adversarial=5)
    det = {k: v["detection_rate"] for k, v in ctrl.items() if not k.startswith("_")}
    assert det and all(r == 1.0 for r in det.values()), det
    # pinned: the runner gets exactly the proved text
    from gmsjourney.sut.adapter import JourneyRunner, make_reference_factory
    runner = JourneyRunner(store=store, realizer=R, reader=rd, bot_factory=make_reference_factory(store),
                           arm_realizers={"llm": realizer})
    from gmsjourney.evaluation.runner import run
    rows = run(llm_scns, runner, oracle.limits, store.workflow, lv)
    assert rows and all(r["arm"] == "llm" for r in rows) and realizer.stats["pinned_hits"] > 0


def test_llm_gates_reject_new_figures_leaks_and_competing_cues():
    store, _o, _R, rd = env()
    from gmsjourney.enrich.llm.gates import LLMOutputGates
    g = LLMOutputGates(store, rd)
    f = F.make("INFORM_DISPOSITION", disposition="RECOGNIZED")
    base = "Yes, I recognize that charge."
    assert g.facts(base, f, {"merchant": "Amazon", "amount": "45.67"}, {}, [base]).ok
    assert not g.facts("Yes, I recognize the $99 charge.", f, {"merchant": "Amazon", "amount": "45.67"}, {}, [base]).ok
    assert not g.facts("Yes, I recognize the Starbucks charge.", f, {"merchant": "Amazon", "amount": "45.67"}, {}, [base]).ok
    assert not g.form("Customer: yes, record_charge_response it").ok
    assert not g.one_clear_decision("I recognize it, it's not mine.", f, {}).ok
    cred = F.make("PROVIDE_CREDENTIAL", credential="4821")
    assert not g.facts("The last four digits are 4812.", cred, {}, {}, ["The last four digits of my card are 4821."]).ok


# ----------------------------------------------------------------------------- runtime additions
def test_quorum_outcomes():
    from gmsjourney.governance.gates import GateDecision as D, GateResult as G
    from gmsjourney.governance.monitor import RuntimeMonitor
    one = [G(D.ALLOW, "syntax"), G(D.DENY, "policy"), G(D.ALLOW, "gms_plausibility")]
    two = [G(D.ALLOW, "syntax"), G(D.DENY, "policy"), G(D.DENY, "gms_plausibility")]
    syn = [G(D.DENY, "syntax"), G(D.ALLOW, "policy"), G(D.ALLOW, "gms_plausibility")]
    assert RuntimeMonitor.outcomes(one) == ("denied", "review")
    assert RuntimeMonitor.outcomes(two) == ("denied", "denied")
    assert RuntimeMonitor.outcomes(syn) == ("denied", "denied")


def test_numeric_attestor():
    store, *_ = env()
    from gmsjourney.governance.attestor import NumericRangeAttestor
    a = NumericRangeAttestor(store)
    L = lambda k: store.lookup_enm("limit", k)
    ok = a.attest(f"You have {L('max_auth_attempts')} verification attempts.")
    assert ok and all(x.verdict == "PASS" for x in ok)
    assert a.attest(f"You have {L('max_transactions_per_session')} verification attempts.")[0].verdict == "REVIEW"
    assert a.attest("You have 9 verification attempts.")[0].verdict == "FAIL"


def test_budgets_and_severity_and_release_gate():
    from gmsjourney.evaluation.constraints import ConstraintBudget, check_constraints
    from gmsjourney.evaluation.severity import classify, release_gate
    c = check_constraints([{"latency_ms": 10, "words": 80}], 3, 3, ConstraintBudget(max_bot_words=60))
    assert c["within_budget"] == 0 and c["budget_violations"]
    row = {"correct": 0, "violations": ["claim_without_success"], "budget_violations": [], "clean_termination": 1}
    assert classify(row, "action_taking")[0] == "critical"
    rows = [{"correct": 1, "risk_tier": "informational", "severity": "none", "failure_codes": [], "within_budget": 1}] * 9 + \
           [{"correct": 0, "risk_tier": "informational", "severity": "critical", "failure_codes": ["HALLUCINATION_DETECTED"],
             "within_budget": 1}]
    g = release_gate(rows)
    assert g["passed"] is False and g["tiers"]["informational"]["reasons"]


def test_operating_points_p50_holdout_bootstrap():
    from gmsjourney.governance.calibrate import bootstrap_thetas, holdout_eval, logistic_p50
    scores = [(0.2 + 0.01 * i, "admit") for i in range(20)] + [(1.0 + 0.02 * i, "block") for i in range(40)]
    p = logistic_p50(scores)
    assert 0.39 < p["theta"] < 1.0 and p["false_admits"] == 0
    h = holdout_eval(scores)
    assert h["regulated"]["heldout_accuracy"] == 1.0
    b = bootstrap_thetas(scores, B=50)
    assert b["regulated"]["ci95"][0] <= b["regulated"]["ci95"][1]


# ----------------------------------------------------------------------------- LLM providers
# (stdlib mock rather than pytest fixtures: this file also runs as a plain script)
def test_provider_defaults_and_fallback_to_deepseek():
    """The gateway is the default; when it cannot be built the fallback takes over and says so."""
    C = llm_client
    s = C.LLMSettings()
    assert s.provider == "aexp" and s.fallback_provider == "deepseek"
    assert s.model == "gpt-41" and "gaiapis" in s.base_url      # the provider's own defaults

    with mock.patch.dict(os.environ):
        os.environ.pop("AEXP_ACCESS_TOKEN", None)
        os.environ["DEEPSEEK_API_KEY"] = "test-key-never-used"
        s = C.LLMSettings()
        with mock.patch.object(C, "_token", side_effect=C.LLMUnavailable("no gcloud")):
            model = C.make_chat_model(s)
        assert s.resolved_provider == "deepseek" and s.effective_provider == "deepseek"
        assert s.model == "deepseek-chat" and "deepseek" in s.base_url  # re-pointed, not left on gpt-41
        assert model is not None
        deepseek_key = C.CachedLLM(s, model=model, tag="t").key("sys", "user", "salt")
        # a fallback to DeepSeek must still find the cache written before the gateway existed
        old_style = C.LLMSettings(provider="deepseek", model="deepseek-chat")
        assert deepseek_key == C.CachedLLM(old_style, tag="t").key("sys", "user", "salt")

    # the cache key follows the transport that actually served the text
    with mock.patch.dict(os.environ, {"AEXP_ACCESS_TOKEN": "token-abc"}):
        s2 = C.LLMSettings()
        assert s2.effective_provider == "aexp"
        assert C.CachedLLM(s2, tag="t").key("sys", "user", "salt") != deepseek_key

    # with neither transport available the error names every reason tried
    with mock.patch.dict(os.environ):
        os.environ.pop("DEEPSEEK_API_KEY", None)
        os.environ.pop("AEXP_ACCESS_TOKEN", None)
        with mock.patch.object(C, "_token", side_effect=C.LLMUnavailable("no gcloud")):
            try:
                C.make_chat_model(C.LLMSettings())
                raise AssertionError("expected LLMUnavailable")
            except C.LLMUnavailable as e:
                assert "aexp" in str(e) and "deepseek" in str(e), str(e)


def test_gateway_transport_builds_the_documented_request():
    """The gateway request matches the deployment's shape: URL, auth header, use-case header, roles."""
    import requests
    C = llm_client
    seen = {}

    class _Resp:
        status_code = 200
        text = ""

        def json(self):
            return {"choices": [{"message": {"content": '{"utterance": "hi"}'}}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3}}

    def _post(url, headers=None, json=None, timeout=None):
        seen.update(url=url, headers=headers, payload=json)
        return _Resp()

    with mock.patch.dict(os.environ, {"AEXP_ACCESS_TOKEN": "token-123", "AEXP_USE_CASE_ID": "825"}):
        model = C.GatewayChat(C.LLMSettings())
        with mock.patch.object(requests, "post", _post):
            out = model.invoke(C._messages("SYS", "USER"))
    assert out.content == '{"utterance": "hi"}'
    assert out.usage_metadata == {"input_tokens": 7, "output_tokens": 3}
    assert seen["url"] == "https://gaiapis.aexp.com/gcpapi/v1/microsoft/models/gpt-41/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer token-123"
    assert seen["headers"]["x-aip-usecase-id"] == "825"
    assert seen["payload"]["messages"] == [{"role": "system", "content": "SYS"},
                                           {"role": "user", "content": "USER"}]
    assert seen["payload"]["max_tokens"] == 220


def test_failover_switches_provider_on_a_call_time_failure():
    """A transport that dies mid-run hands over to the fallback once, and the switch is visible."""
    C = llm_client
    s = C.LLMSettings()
    built = []

    def _fake_build(provider, settings):
        built.append(provider)
        settings.resolved_provider = provider
        settings.model = C.PROVIDER_DEFAULTS[provider]["model"]
        return SimpleNamespace(invoke=lambda msgs: SimpleNamespace(content="from " + provider))

    class _Down:
        def invoke(self, msgs):
            raise RuntimeError("gateway down")

    with mock.patch.object(C, "_build", _fake_build):
        fo = C.FailoverChat(_Down(), s, "deepseek")
        out = fo.invoke([SimpleNamespace(type="user", content="x")])
    assert out.content == "from deepseek" and fo.switched_to == "deepseek"
    assert built == ["deepseek"] and s.effective_provider == "deepseek"
    assert "gateway down" in (fo.reason or "")


def test_gateway_path_is_langchain_free():
    """The gateway transport must not touch LangChain: it works with langchain_core unimportable,
    through the same CachedLLM path the campaign uses."""
    import requests
    C = llm_client

    class _Resp:
        status_code = 200
        text = ""

        def json(self):
            return {"choices": [{"message": {"content": "gateway answer"}}], "usage": {}}

    with mock.patch.dict(os.environ, {"AEXP_ACCESS_TOKEN": "token"}):
        settings = C.LLMSettings(cache_dir=tempfile.mkdtemp())
        blocked = {"langchain_core": None, "langchain_core.messages": None}
        with mock.patch.dict(sys.modules, blocked):          # None in sys.modules = ImportError
            with mock.patch.object(requests, "post", lambda *a, **k: _Resp()):
                client = C.CachedLLM(settings, tag="langchain-free")
                assert client.invoke("sys", "user", salt="s") == "gateway answer"
                assert client.stats.errors == 0 and client.stats.provider == "aexp"


def test_first_call_after_a_fallback_is_filed_under_the_resolved_provider():
    """The cache path is computed before the provider resolves, so it must be recomputed after —
    otherwise the first answer of every client lands in the requested provider's directory under
    a key nobody looks up again."""
    C = llm_client
    answers = []

    def _stub(settings):
        return SimpleNamespace(invoke=lambda msgs: (answers.append(1),
                                                    SimpleNamespace(content="from deepseek"))[1])

    with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "k"}):
        cache = Path(tempfile.mkdtemp())
        settings = C.LLMSettings(provider="aexp", cache_dir=str(cache))
        settings.model = "gpt-41"                     # as CachedLLM.__post_init__ resolves it
        with mock.patch.object(C, "_token", side_effect=C.LLMUnavailable("no gcloud")), \
             mock.patch.object(C, "_deepseek_model", _stub):
            client = C.CachedLLM(settings, tag="t")
            assert client.invoke("s", "u", salt="x") == "from deepseek"
            assert settings.effective_provider == "deepseek"

            # filed under the transport that answered, not the one that was requested
            resolved = client._path(client.key("s", "u", "x"))
            assert "deepseek-chat" in str(resolved) and resolved.exists()
            assert not any(p.suffix == ".json" for p in (cache / "gpt-41").rglob("*.json")) \
                if (cache / "gpt-41").exists() else True

            # and a second client finds it instead of calling again
            again = C.CachedLLM(settings, tag="t")
            assert again.invoke("s", "u", salt="x") == "from deepseek"
            assert again.stats.cache_hits == 1 and len(answers) == 1


def test_cached_llm_records_a_failover_in_its_stats():
    """The switch reaches the report: a campaign that fell back must not look like one that did not."""
    C = llm_client

    class _Switched:
        """A model that has already handed over to the fallback, as FailoverChat leaves it."""
        switched_to = "deepseek"
        reason = "RuntimeError: boom"

        def invoke(self, msgs):
            return SimpleNamespace(content="answer")

    s = C.LLMSettings(cache_dir=tempfile.mkdtemp())
    client = C.CachedLLM(s, tag="failover-test", model=_Switched())
    assert client.invoke("sys", "user", salt="s") == "answer"
    assert client.stats.switched_to == "deepseek"
    assert "boom" in (client.stats.failover_reason or "")
    assert client.report()["switched_to"] == "deepseek"
    assert client.report()["provider"] == s.effective_provider


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
