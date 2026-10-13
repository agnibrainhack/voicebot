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


# ----------------------------------------------------------------------------- the Microsoft gateway
# (stdlib mock rather than pytest fixtures: this file also runs as a plain script)
def test_gateway_is_the_only_transport():
    """One transport, no fallback: the defaults name the gateway and nothing else can be selected."""
    C = llm_client
    s = C.LLMSettings()
    assert s.provider == "aexp" and s.model == "gpt-41" and "gaiapis" in s.base_url
    assert not hasattr(C, "FailoverChat") and not hasattr(C, "_deepseek_model")
    with mock.patch.dict(os.environ, {"AEXP_ACCESS_TOKEN": ""}), \
            mock.patch.object(C, "_token", side_effect=C.LLMUnavailable("no gcloud")):
        try:
            C.make_chat_model(C.LLMSettings())
            raise AssertionError("expected LLMUnavailable")
        except C.LLMUnavailable as e:
            assert "gcloud" in str(e)


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

    def _post(url, headers=None, json=None, timeout=None, **kw):
        seen.update(url=url, headers=headers, payload=json, verify=kw.get("verify"))
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


def test_gateway_url_override_ca_bundle_and_400_adaptation():
    """GMSJOURNEY_LLM_URL is used verbatim ({model} filled in), the CA bundle is passed to requests,
    a 401 refreshes the token once, and a 400 on sampling parameters drops them for the rest of the run."""
    import requests
    C = llm_client
    calls = []

    class _R:
        def __init__(self, code, body):
            self.status_code, self._b, self.text, self.headers = code, body, json.dumps(body), {}

        def json(self):
            return self._b

    def _post(url, headers=None, json=None, timeout=None, verify=None):
        calls.append({"url": url, "keys": sorted(json), "verify": verify, "auth": headers["Authorization"]})
        if len(calls) == 1:
            return _R(401, {"error": "expired"})
        if "temperature" in json:
            return _R(400, {"error": {"message": "Unsupported parameter: 'temperature'"}})
        return _R(200, {"choices": [{"message": {"content": "ok"}}], "usage": {}})

    bundle = tempfile.NamedTemporaryFile(suffix=".crt", delete=False).name
    env = {"AEXP_ACCESS_TOKEN": "tok", "GMSJOURNEY_LLM_URL": "gaiapis.example/gcpapi/x/{model}/completions",
           "GMSJOURNEY_CA_BUNDLE": bundle}
    with mock.patch.dict(os.environ, env), mock.patch.object(requests, "post", _post), \
            mock.patch.object(C.time, "sleep", lambda *_: None):
        model = C.GatewayChat(C.LLMSettings())
        assert model.invoke(C._messages("S", "U")).content == "ok"
    assert calls[0]["url"] == "https://gaiapis.example/gcpapi/x/gpt-41/completions"
    assert all(c["verify"] == bundle for c in calls)
    assert len(calls) == 3 and calls[-1]["keys"] == ["messages", "model"]
    assert model.adaptations == ["sampling parameters dropped (model + messages only)"]


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


def test_prefetch_is_concurrent_and_changes_nothing():
    """Prefetching issues the first-attempt calls on a thread pool; the proof then finds them in the
    cache, so the result is identical to a sequential run and only retries go out afterwards."""
    import threading
    import time as _t
    store, oracle, R, rd, ls, tv, realizer, lv, llm_scns = _llm_fixture()
    seq = [(p.sid, p.accepted, p.enriched_signature) for p in (lv.prove(s) for s in llm_scns)]
    store, oracle, R, rd, ls, tv, realizer, lv, llm_scns = _llm_fixture()      # fresh caches
    active, peak = [0], [0]
    lock = threading.Lock()
    for c in (lv.llm_realizer.llm, lv.extractor.llm):
        inner = c.model

        class _Slow:
            def __init__(self, m):
                self.m = m

            def invoke(self, msgs):
                with lock:
                    active[0] += 1
                    peak[0] = max(peak[0], active[0])
                _t.sleep(0.02)
                with lock:
                    active[0] -= 1
                return self.m.invoke(msgs)
        c.model = _Slow(inner)
    info = lv.prefetch(llm_scns, workers=6, log=None)
    assert info["rephrase_ok"] > 0 and info["rephrase_failed"] == 0 and peak[0] > 1
    misses = lv.llm_realizer.llm.stats.cache_misses
    par = [(p.sid, p.accepted, p.enriched_signature) for p in (lv.prove(s) for s in llm_scns)]
    assert par == seq
    assert lv.llm_realizer.llm.stats.cache_misses == misses        # the proof made no new rephrase calls


def test_parallel_prove_all_matches_sequential():
    """prove_all(workers>1) overlaps the gateway waits of different scenarios (retries included) and
    returns exactly the sequential proofs, in input order."""
    import hashlib
    import threading
    import time as _t

    class _FirstTryOff:
        """Every other first attempt comes back off-topic, so the proof has retries to overlap."""
        def __init__(self, m):
            self.m = m

        def invoke(self, msgs):
            u = str(msgs[-1])
            if "previous_attempt_rejected_because" not in u and int(hashlib.sha256(u.encode()).hexdigest(), 16) % 2:
                return '{"utterance": "what is the weather like today"}'
            return self.m.invoke(msgs)

    real_messages = llm_client._messages
    store, oracle, R, rd, ls, tv, realizer, lv, llm_scns = _llm_fixture()
    lv.llm_realizer.llm.model = _FirstTryOff(lv.llm_realizer.llm.model)
    seq = [(p.sid, p.accepted, p.enriched_signature, p.retries) for p in lv.prove_all(llm_scns, show=False)]
    assert sum(r for *_, r in seq) > 0                           # retries really happen
    store, oracle, R, rd, ls, tv, realizer, lv, llm_scns = _llm_fixture()      # fresh caches
    lv.llm_realizer.llm.model = _FirstTryOff(lv.llm_realizer.llm.model)
    active, peak = [0], [0]
    lock = threading.Lock()
    for c in (lv.llm_realizer.llm, lv.extractor.llm):
        inner = c.model

        class _Slow:
            def __init__(self, m):
                self.m = m

            def invoke(self, msgs):
                with lock:
                    active[0] += 1
                    peak[0] = max(peak[0], active[0])
                _t.sleep(0.02)
                with lock:
                    active[0] -= 1
                return self.m.invoke(msgs)
        c.model = _Slow(inner)
    par = [(p.sid, p.accepted, p.enriched_signature, p.retries) for p in lv.prove_all(llm_scns, workers=6, show=False)]
    assert par == seq and peak[0] > 1
    assert lv.llm_realizer.llm.io_release is None                # the hook is removed afterwards
    llm_client._messages = real_messages


def test_live_responses_are_printed_cache_hits_are_not():
    import contextlib
    import io
    import tempfile as _tf
    from gmsjourney.enrich.llm import CachedLLM, LLMSettings

    class _Echo:
        def invoke(self, msgs):
            return '{"utterance": "yes that one is mine"}'
    c = CachedLLM(LLMSettings(cache_dir=_tf.mkdtemp(), log_responses=True), model=_Echo(), tag="rephrase")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        c.invoke("sys", "user", salt="J1|0")
        c.invoke("sys", "user", salt="J1|0")                   # cache hit: silent
    out = buf.getvalue()
    assert out.count("[LLM rephrase") == 1 and "J1|0" in out and "yes that one is mine" in out


def test_circuit_breaker_stops_after_consecutive_failures():
    import contextlib
    import io
    import tempfile as _tf
    from gmsjourney.enrich.llm import CachedLLM, LLMSettings, LLMUnavailable as LU
    calls = [0]

    class _Down:
        def invoke(self, msgs):
            calls[0] += 1
            raise TimeoutError("read timed out")
    c = CachedLLM(LLMSettings(cache_dir=_tf.mkdtemp(), max_consecutive_failures=3), model=_Down(), tag="rephrase")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        for i in range(10):
            try:
                c.invoke("s", f"u{i}")
            except LU:
                pass
    assert calls[0] == 3                                    # no network calls after the breaker opened
    assert not c.available and "3 failed calls in a row" in c.unavailable_reason()
    assert "FAILED" in buf.getvalue() and "read timed out" in buf.getvalue()


def test_pick_llm_rows_caps_and_spreads_over_bases():
    from types import SimpleNamespace as NS
    from gmsjourney.pipeline import pick_llm_rows
    rows = [NS(sid=f"B{b}~{k}", base=NS(qid=f"B{b}")) for b in range(10) for k in range(6)]
    assert pick_llm_rows(rows, 1.0, None, 1) == rows
    got = pick_llm_rows(rows, 1.0, 12, 1)
    assert len(got) == 12 and len({s.base.qid for s in got}) == 10          # every base before any repeats
    assert got == sorted(got, key=rows.index)                               # design order kept
    assert pick_llm_rows(rows, 1.0, 12, 1) == got                           # deterministic


def test_partial_network_failures_are_retried_and_never_counted_as_meaning_failures():
    """Flaky gateway: transient failures are recovered by the retry pass (same proofs as a clean run);
    a call that keeps failing leaves its scenario 'not generated' - never a gate failure, never run."""
    import hashlib
    real_messages = llm_client._messages
    store, oracle, R, rd, ls, tv, realizer, lv, llm_scns = _llm_fixture()
    clean = [(p.sid, p.accepted, p.enriched_signature) for p in lv.prove_all(llm_scns, show=False, log=None)]
    clean_summary = lv.summarize_llm(lv.prove_all(llm_scns, show=False, log=None), [])

    def h(u):
        return int(hashlib.sha256(u.encode()).hexdigest(), 16)

    class _Flaky:                                   # every 3rd distinct prompt fails once, then works
        def __init__(self, m):
            self.m, self.seen = m, set()

        def invoke(self, msgs):
            u = str(msgs[-1])
            if h(u) % 3 == 0 and u not in self.seen:
                self.seen.add(u)
                raise ConnectionError("connection reset")
            return self.m.invoke(msgs)

    store, oracle, R, rd, ls, tv, realizer, lv, llm_scns = _llm_fixture()
    for c in (lv.llm_realizer.llm, lv.extractor.llm):
        c.settings.max_consecutive_failures = 0
        c.model = _Flaky(c.model)
    got = [(p.sid, p.accepted, p.enriched_signature) for p in lv.prove_all(llm_scns, show=False, log=None,
                                                                             retry_unavailable=2)]
    assert got == clean                                                    # fully recovered

    class _Dead:                                    # the independent reader is down for good
        def invoke(self, msgs):
            raise ConnectionError("down")
    store, oracle, R, rd, ls, tv, realizer, lv, llm_scns = _llm_fixture()
    lv.extractor.llm.settings.max_consecutive_failures = 0
    lv.extractor.llm.model = _Dead()
    proofs = lv.prove_all(llm_scns, show=False, log=None)
    sm = lv.summarize_llm(proofs, [])
    assert sm["accepted"] == 0 and sm["not_generated_network"] == len(llm_scns)
    assert sm["rejected_meaning_changed"] == 0 and not sm["gate_failures_by_attempt"].get("G2_dual_read")
    assert all(len(t["attempts"]) <= 1 for p in proofs for t in p.turns)    # no rephrase retries wasted
    assert clean_summary["not_generated_network"] == 0
    llm_client._messages = real_messages


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
