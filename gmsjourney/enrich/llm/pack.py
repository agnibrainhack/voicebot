"""The LLM enrichment arm for any pack compiled from a compact graph: prompts, schema and checks read from the pack.

The charge-review arm names its own wordings (the bot's questions, the reader's words for each value, the policy
question intents). For a compiled pack each comes from the pack:

    the agent's question     the reference bot's wording of the prompt (domain.yaml -> bot.says)
    understood names         the lexicon's wordings of each value (values.<V>.say + cues)
    what the customer means  the act and each slot's value in words
    policy questions         the graph's figures: value (attr, entity) and limit (key)
    the reader's schema      the graph's acts, their slots and value sets
"""
from __future__ import annotations

from typing import Any

from ...ontology import SemanticFrame as F

KB_INTENTS = {"value": ("attr", "entity"), "limit": ("key",)}


def active():
    """The active pack when it is compiled from a compact graph, else None (charge review's arm applies)."""
    from ...domain import current
    dom = current()
    return dom if dom.graph is not None else None


def _words(name: str) -> str:
    return str(name).lower().replace("_", " ")


def label(dom, value: Any) -> str:
    """A value, a limit or an attribute in words."""
    v = str(value)
    say = ((dom.get("lexicon.values") or {}).get(v) or {}).get("say") or {}
    if say.get("canonical"):
        return say["canonical"]
    g = dom.graph
    if v in g.limits:
        return g.limits[v].get("text") or _words(v)
    al = g.aliases(v) if g.member_set(v) else []
    return al[0] if al else _words(v)


def reader_lexicon(dom) -> dict[str, list[str]]:
    """value -> the words the rule reader binds to it (canonical first)."""
    out = {}
    lex = dom.get("lexicon.values") or {}
    g = dom.graph
    for vs in g.values:
        for m in g.members(vs):
            ent = lex.get(m) or {}
            say = ent.get("say") or {}
            words = [say[k] for k in ("canonical", "synonym", "colloquial") if say.get(k)]
            words += [c for c in ent.get("cues") or [] if "\\" not in c and "(" not in c]
            if words:
                out[m] = list(dict.fromkeys(words))
    acts = dom.get("lexicon.acts") or {}
    for a in ("REQUEST_HUMAN", "END"):
        ph = (acts.get(a) or {}).get("phrase") or {}
        words = [ph[k] for k in ("canonical", "synonym", "colloquial") if ph.get(k)]
        if words:
            out[a] = words
    return out


def state_question(dom, state: str | None, ctx: dict[str, Any] | None) -> str:
    """The agent's line the customer answers: the reference bot's wording of the prompt."""
    if not state:
        return "The agent is waiting for your reply."
    from ...graph.bot import _say
    return _say({"act": state, **{k: v for k, v in (ctx or {}).items() if k != "state"}})


def describe_meaning(dom, f: F, spoken: bool = False) -> str | None:
    """The meaning in plain words for a pack act (None: a global act the charge arm already describes)."""
    if f.act in ("CONFIRM", "DENY", "REQUEST_HUMAN", "END", "UNCLEAR", "CHITCHAT", "PROVIDE_CREDENTIAL"):
        return None
    if f.act == "ASK_KB":
        return "asks a question: " + describe_kb(dom, f)
    parts = [f"{_words(k)}: {label(dom, v)}" for k, v in f.slots if v is not None]
    return f"{_words(f.act)}" + (f" ({'; '.join(parts)})" if parts else "")


def describe_kb(dom, f: F) -> str:
    i = f.get("intent")
    if i == "value":
        return f"what the {_words(f.get('attr'))} of {label(dom, f.get('entity'))} is"
    if i == "limit":
        return label(dom, f.get("key"))
    return _words(i)


def slot_values(f: F) -> list[str]:
    return [str(v) for _, v in f.slots if isinstance(v, str)]


def act_schema(dom) -> dict[str, dict[str, tuple[str, bool]]]:
    """act -> slot -> (value set, required), the global acts and the policy question included."""
    g = dom.graph
    out: dict[str, dict[str, tuple[str, bool]]] = {}
    for a in (g.data.get("acts") or {}):
        out[a] = {sl: (sp["set"], not sp.get("optional")) for sl, sp in g.slot_specs(a).items()}
    for a in ("CONFIRM", "DENY", "END", "UNCLEAR", "REQUEST_HUMAN"):
        out.setdefault(a, {})
    if (g.glob.get("question") or {}).get("act"):
        out["ASK_KB"] = {"intent": ("kb_intent", True), "attr": ("kb_attribute", False), "entity": ("kb_entity", False),
                         "key": ("limit", False)}
    return out


def extraction_schema(dom) -> dict[str, Any]:
    g = dom.graph
    vs = {k: g.members(k) for k in g.values}
    vs["kb_intent"] = list(KB_INTENTS)
    vs["kb_attribute"] = list(g.numeric_attributes())
    vs["kb_entity"] = sorted({m for sp in g.numeric_attributes().values() for m in sp["members"]})
    vs["limit"] = [k for k, lv in g.limits.items() if not lv.get("internal")]
    return {a: {s: {"values": vs.get(v) or v, "required": req} for s, (v, req) in slots.items()}
            for a, slots in act_schema(dom).items()}


def kb_intents_text(dom) -> dict[str, str]:
    g = dom.graph
    attrs = ", ".join(g.numeric_attributes())
    return {"value": f"a figure of a named option (attr one of {attrs}; entity) | slots: attr, entity",
            "limit": "a limit of the policy (key) | slots: key"}


def gates_one_decision(reader, text: str, frame: F, levels: dict[str, str]) -> list[str]:
    """One clear decision: per slot, at most one value cued (two with a self-correction); no slot cue the meaning
    does not carry."""
    from ..realizers import BASE_LEVELS
    cues = reader.cue_counts(text)
    lv = {**BASE_LEVELS, **(levels or {})}
    allowed = 2 if lv.get("self_correction", "none") != "none" or frame.get("corrected_from") else 1
    r = []
    meant = {k for k, v in frame.slots if v is not None}
    for slot in reader.lx.slots:
        if slot in meant:
            if cues.get(f"{slot}_values", 0) > allowed:
                r.append(f"{cues[f'{slot}_values']} competing {slot} cues (allowed {allowed})")
        elif cues.get(slot) and frame.act not in ("ASK_KB",):
            r.append(f"carries a {slot} cue the meaning does not have")
    return r
