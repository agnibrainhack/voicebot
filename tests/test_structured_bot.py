"""The LLM bot under test (gmsjourney/s06_bots/llm_bot.py): its model reads and words, the journey's graph decides.

No knowledge store build, no gateway: the Playground's bench store and scripted models.

* a reading is used only when the state accepts it: an intent, a slot value or a code the state does not take, or a
  code that is not in the customer's words, is read as unclear (counted, with why);
* small talk changes nothing and is not a clarification; a policy claim is answered from the graph ("Actually, ..."
  when the customer's figure is wrong), never agreed to as said; a policy question gets exactly the fact asked;
* a worded line that adds or drops a fact, a figure, an option or a promise is not said: its template is;
* the wording model is chosen apart from the reading model (the same, a gateway deployment, a local model, or the
  templates), and the oracle treats small talk as no answer (asked again, no clarification).

    python tests/test_structured_bot.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gmsjourney.s01_packs import loading as domain  # noqa: E402
from gmsjourney.llm import CachedLLM, LLMSettings  # noqa: E402


class Script:
    """A chat model that answers reading requests with ``reads`` (in order; then the stand-in's own reading) and
    wording requests with ``word(payload)`` (default: the templates)."""

    def __init__(self, reads=(), word=None):
        from fake_sut_model import HappyPath
        self.reads, self.word, self.base = list(reads), word, HappyPath()
        self.calls = self.readings = self.wordings = 0

    def invoke(self, msgs):
        self.calls += 1
        p = json.loads(msgs[-1].content)
        if "menu" in p:
            self.readings += 1
            out = self.reads.pop(0) if self.reads else self.base.read(p)
        else:
            self.wordings += 1
            out = self.word(p) if self.word else {"lead": "Thanks for that." if p.get("customer_turn") == "social" else "",
                                                  "lines": [x["template"] for x in p["lines"]]}
        return SimpleNamespace(content=json.dumps(out), usage_metadata={})


def _bot(model, wording=True, pack="charge_review_graph"):
    import tempfile
    from gmsjourney.s03_knowledge import build as backend
    from gmsjourney.s06_bots.llm_bot import make_structured_factory
    from gmsjourney.s06_bots.tools import ToolEnvironment
    from gmsjourney.s10_console import bots as PG
    domain.set_current(pack)
    backend.CACHE = Path(tempfile.mkdtemp(prefix="gms_llm_cache_"))    # every call reaches the scripted model
    PG.cases("project", None)
    case = PG.find_case("project", "default")
    r = CachedLLM(LLMSettings(temperature=0, max_tokens=200, json_mode=True), model=model, tag="sut")
    w = CachedLLM(LLMSettings(temperature=0, max_tokens=400, json_mode=True), model=model, tag="sut-word") if wording else None
    fac = make_structured_factory(PG.bench().store, reader_llm=r, wording_llm=w)
    env = ToolEnvironment(json.loads(json.dumps(case["fixture"])))
    return fac, fac(env), env, case


def _acts(turn):
    return [a["act"] for a in turn.acts]


def _restore(keep):
    from gmsjourney.s03_knowledge import build as backend
    backend.CACHE = keep
    domain.set_current("charge_review")


def _keep():
    from gmsjourney.s03_knowledge import build as backend
    return backend.CACHE


# ------------------------------------------------------------------ reading
def test_a_reading_is_used_only_when_the_state_accepts_it():
    keep = _keep()
    try:
        m = Script(reads=[{"kind": "actionable", "intent": "INFORM_DISPOSITION", "slots": {"disposition": "RECOGNIZED"}},  # not here
                          {"kind": "actionable", "intent": "PROVIDE_CREDENTIAL", "slots": {"credential": "4821"}}])
        fac, bot, env, case = _bot(m)
        bot.start()
        t = bot.respond("I'd like to start, please.")
        assert _acts(t) == ["ASK_CLARIFY", "ASK_AUTH"] and "not one this state takes" in bot.reader.last["why"]
        bot.respond("It's four eight two one.")                         # spoken digits are the code
        assert [c["tool"] for c in env.log][:1] == ["authenticate_customer"] and env.log[0]["args"]["credential"] == "4821"
        m = Script(reads=[{"kind": "actionable", "intent": "PROVIDE_CREDENTIAL", "slots": {"credential": "9999"}}])
        fac2, bot, env, case = _bot(m)                                   # a code the customer did not say
        bot.start()
        t = bot.respond("It's four eight two one.")
        assert "not in the customer's words" in bot.reader.last["why"] and not env.log and _acts(t)[0] == "ASK_CLARIFY"
        st = fac.stats.as_dict()
        assert st["rejected"] == 1 and st["readings"] == 2 and len(st["notes"]) == 1 and fac2.stats.rejected == 1
    finally:
        _restore(keep)


def test_values_are_mapped_to_the_graphs_members_and_a_plain_yes_reads_as_the_state_says():
    keep = _keep()
    try:
        from gmsjourney.s06_bots.llm_bot import LLMReader, StructuredStats
        domain.set_current("charge_review_graph")
        g = domain.current().graph
        r = LLMReader(g, None, StructuredStats())
        menu = r.menu("PRESENT_TXN")
        f, kind, why = r.frame({"kind": "actionable", "intent": "INFORM_DISPOSITION", "slots": {"disposition": "recognized"}}, "", "PRESENT_TXN", menu)
        assert (f.act, f.get("disposition"), why) == ("INFORM_DISPOSITION", "RECOGNIZED", None)
        f, kind, why = r.frame({"kind": "actionable", "intent": "yes"}, "yes", "PRESENT_TXN", menu)
        assert (f.act, f.get("disposition")) == ("INFORM_DISPOSITION", "RECOGNIZED"), f
        f, kind, why = r.frame({"kind": "actionable", "intent": "INFORM_DISPOSITION", "slots": {"disposition": "MAYBE_LATER"}}, "", "PRESENT_TXN", menu)
        assert f.act == "UNCLEAR" and "MAYBE_LATER" in why
        f, kind, why = r.frame({"kind": "question", "about": "value:fee_usd:expedited_delivery"}, "", "PRESENT_TXN", menu)
        assert (f.act, f.get("intent"), f.get("attr"), f.get("entity")) == ("ASK_KB", "value", "fee_usd", "expedited_delivery")
        f, kind, why = r.frame({"kind": "question", "about": "limit:max_turns"}, "", "PRESENT_TXN", menu)   # internal
        assert f.get("intent") == "unstated_fact" and "no such fact" in why
        f, kind, why = r.frame({"kind": "social"}, "thanks", "PRESENT_TXN", menu)
        assert f.act == "CHITCHAT" and kind == "social" and why is None
    finally:
        _restore(keep)


# ------------------------------------------------------------------ social talk, claims, questions
def test_small_talk_is_not_an_answer_and_a_claim_is_answered_from_the_graph():
    keep = _keep()
    try:
        fac, bot, env, case = _bot(Script())
        bot.start()
        t = bot.respond("Thanks, hope you're having a good day!")
        assert _acts(t) == ["ASK_AUTH"] and bot.reader.last["kind"] == "social"     # asked again, no clarification
        assert t.text.startswith("Thanks for that.")                                 # the acknowledgement lead
        bot.respond("The last four digits of my card are 4821.")
        t = bot.respond("The expedited delivery fee is $5, right?")
        assert _acts(t) == ["ANSWER_KB", "PRESENT_TXN"] and t.acts[0]["text"].startswith("Actually, ") and "$15" in t.text
        assert "$5" not in t.text                                                    # the customer's figure is not repeated
        t = bot.respond("The expedited delivery fee is $15, right?")
        assert t.acts[0]["text"].startswith("Yes, that's right.")
        t = bot.respond("How much is expedited delivery?")
        assert _acts(t)[0] == "ANSWER_KB" and "standard" not in t.acts[0]["text"].lower()   # only the fact asked
        assert fac.stats.kinds == {"social": 1, "actionable": 1, "claim": 2, "question": 1}
    finally:
        _restore(keep)


def test_the_oracle_asks_again_after_small_talk_without_a_clarification():
    keep = _keep()
    try:
        from gmsjourney.s04_cases.meaning import SemanticFrame as F
        from gmsjourney.s06_bots.reference import GraphBot
        from gmsjourney.s06_bots.tools import ToolEnvironment
        from gmsjourney.s10_console import bots as PG
        domain.set_current("charge_review_graph")
        PG.cases("project", None)
        case = PG.find_case("project", "default")
        b = GraphBot(ToolEnvironment(json.loads(json.dumps(case["fixture"]))), PG.bench().store)
        b.start()
        st = b._state
        assert b.react(st, F.make("CHITCHAT")) is None and not b._pending and b._unclear == {}
        assert b.react(st, F.make("UNCLEAR")) is None and b._pending[-1]["act"] == "ASK_CLARIFY"
    finally:
        _restore(keep)


# ------------------------------------------------------------------ wording
def test_a_worded_line_that_changes_a_fact_is_not_said():
    keep = _keep()
    try:
        def word(p):                                   # adds a figure to the first line; says the others as given
            lines = [x["template"] for x in p["lines"]]
            lines[0] = lines[0].rstrip(".") + ", and it's only $99 today."
            return {"lead": "Sure, expedited delivery is great.", "lines": lines}
        fac, bot, env, case = _bot(Script(word=word))
        t = bot.start()
        assert "$99" not in t.text and "expedited" not in t.text.lower()            # the template and no lead
        st = fac.stats.as_dict()
        assert st["wording_rejected"] == 1 and st["templated"] == 1 and st["worded"] == 1 and st["leads"] == 0
        assert any("figures" in n["why"] for n in st["notes"])          # (no customer turn yet: no lead is said)
        fac, bot, env, case = _bot(Script(word=lambda p: {"lines": ["just one line"]}))   # not one line per act
        t = bot.start()
        assert fac.stats.wording_rejected == 2 and "four digits" in t.text
        m = Script()
        fac, bot, env, case = _bot(m, wording=False)                                 # templates: no wording calls
        bot.start()
        bot.respond("The last four digits of my card are 4821.")
        assert m.wordings == 0 and m.readings == 1 and fac.stats.templated >= 3
    finally:
        _restore(keep)


def test_the_wording_model_is_chosen_apart_from_the_reading_model():
    keep = _keep()
    old = os.environ.get("GMSJOURNEY_SUT_FAKE_MODEL")
    os.environ["GMSJOURNEY_SUT_FAKE_MODEL"] = "fake_sut_model:HappyPath"
    try:
        import tempfile
        from gmsjourney.s10_console import bots as B
        domain.set_current("charge_review_graph")
        base = {"kind": "llm", "model": "gpt-41"}
        fac, name, _ = B.make_factory(base)
        assert "gpt-41 reads" in name and "the same model words" in name and fac.wording_llm is not None
        assert fac.wording_llm.settings.max_tokens == 400 and fac.llm.settings.max_tokens == 200
        fac, name, _ = B.make_factory({**base, "wording": "gateway", "wording_model": "gpt-4.1-mini"})
        assert fac.wording_llm.settings.model == "gpt-4.1-mini" and "gpt-4.1-mini words" in name
        fac, name, _ = B.make_factory({**base, "wording": "template"})
        assert fac.wording_llm is None and "templates" in name
        folder = Path(tempfile.mkdtemp()) / "gpt-oss-20b"
        folder.mkdir()
        (folder / "config.json").write_text("{}")
        fac, name, _ = B.make_factory({**base, "wording": "local", "wording_local_path": str(folder),
                                       "wording_vllm_url": "http://127.0.0.1:9/v1"})
        assert "(local, vLLM) words" in name and fac.wording_llm is not fac.llm
        try:
            B.make_factory({**base, "wording": "local", "wording_local_path": str(folder) + "-missing"})
            raise AssertionError("an unusable local wording model must be refused")
        except RuntimeError as e:
            assert "wording local model" in str(e)
        try:
            B.start("project", "default", {**base, "wording": "local", "wording_local_path": str(folder)})
            raise AssertionError("the Playground needs a vLLM server for a local wording model")
        except ValueError as e:
            assert "wording's local model needs a vLLM server" in str(e)
        assert B.wording_where({}) == "same" and B.wording_where({"wording": "nonsense"}) == "same"
    finally:
        if old is None:
            os.environ.pop("GMSJOURNEY_SUT_FAKE_MODEL", None)
        else:
            os.environ["GMSJOURNEY_SUT_FAKE_MODEL"] = old
        _restore(keep)


def test_a_factory_without_its_store_says_so_and_the_menu_is_small():
    keep = _keep()
    try:
        from gmsjourney.s06_bots.llm_bot import READ_SYSTEM, LLMReader, StructuredStats, make_structured_factory
        domain.set_current("charge_review_graph")
        fac = make_structured_factory(None, reader_llm=CachedLLM(LLMSettings(), model=Script(), tag="sut"))
        try:
            fac(None)
            raise AssertionError("a bot without the knowledge store must say so")
        except RuntimeError as e:
            assert "knowledge store" in str(e)
        g = domain.current().graph
        r = LLMReader(g, None, StructuredStats())
        biggest = max(len(json.dumps(r.menu(s))) for s in g.states if g.kind(s) == "ask")
        assert len(READ_SYSTEM) + biggest < 4000, biggest                    # a reading call: well under 1k tokens
    finally:
        _restore(keep)


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
