"""Grading an agent that gives only tool calls and words: the oracle's SAYS, held in the agent's words.

A real agent under test makes tool calls (each with its result) and speaks; it does not say which dialog acts its
words are. So a turn is graded on what can be observed, against the graph's own run of the same journey (the
oracle):

* the **tool calls** with their key arguments, exactly as before (s06_bots/turns.compare_turn);
* the **oracle's SAYS** for that turn (the acts the graph says there, each with its values): every one must be
  **held** by the agent's words.

An expected act is held when the words express it and say no other value than the oracle's:

    held            heard (a cue word of the act, s07_checks/speech_reading.bot_cues; or every field the act must
                    say, read as claims; OPENING: the opening statement; a policy answer: a figure about what was
                    asked) and every value the words give for its fields is the oracle's
    contradicted    heard, but the words give another value (noted T1 as RECOGNIZED where the oracle has
                    UNRECOGNIZED; a policy answer that is not the policy's)
    not_heard       the act can be heard from words, and these words do not express it
    not_checkable   the act has no cue word and nothing it must say: the words cannot confirm it (it is taken as
                    said when the turn's tool calls match; reported as such, never hidden)

Each sentence of the agent's words is given to one expected act (the first one it expresses that has no sentence
yet; else the one it expresses nearest after the last; a sentence that expresses none goes with the act before it),
so the speech check reads each claim under the act it belongs to (a summary is held to the confirmed writes, a
choice offered at a prompt is a choice). A cue heard in a sentence that belongs to no expected act is reported as
**said but not expected**, and counts as an act the agent took.

    hold(verifier, expected, text, facts, question=None, store=None)
        -> {"sentences": [{text, act, of, heard}], "says": [{act, label, status, how, sentences, said, detail}],
            "extra": [{act, sentence, how}], "acts": [...the acts the turn is graded as having said...],
            "from": [the expected act each of those came from],
            "segments": [(label or None, words)], "heard": [...every act the words express...]}

The truth of what the agent says still comes only from the tools' results and the customer's meaning cards; the
oracle's SAYS decides which act a sentence belongs to, never what is true.
"""
from __future__ import annotations

from typing import Any

from ..s07_checks.speech_reading import _sentences, words_to_digits


def _need(c, label: dict) -> list[str]:
    """The claim fields an act must say and has a value for (OPENING: the opening statement)."""
    spec = c.acts.get(label.get("act"))
    if spec is None:
        return []
    need = [c.claim_field(f) for f in spec.says if f.required and label.get(f.field) is not None]
    return [x for x in need if x]


def _num(v: Any) -> Any:
    try:
        f = float(v)
        return int(f) if f == int(f) else f
    except (TypeError, ValueError):
        return v


def hold(verifier, expected: list[dict] | None, text: str, facts, question: Any = None, store: Any = None,
         answer_act: str | None = None) -> dict[str, Any]:
    """The oracle's acts for one turn, each checked against the agent's words (see the module docstring).
    ``expected`` None (no oracle for this turn: an adaptive customer, injected faults): the acts are the ones the
    words express."""
    c, reader = verifier.c, verifier.reader
    g = getattr(getattr(verifier, "generic", None), "m", None)
    g = getattr(g, "g", None)
    sents = _sentences(text or "")
    reads = [reader.read(s, facts, question=question) for s in sents]
    lows = [words_to_digits(s).lower() for s in sents]
    whole = reader.read(text or "", facts, question=question)
    heard_all = list(whole.heard)

    def cue_in(act: str, low: str) -> str | None:
        spec = c.acts.get(act)
        return next((q for q in (spec.cues if spec else ()) if q and q in low), None)

    kb = None
    if answer_act and question is not None and store is not None and (text or "").strip():
        from ..s03_knowledge.questions import anchored_figures, judge_kb_answer
        kb = judge_kb_answer(store, question, text)
        kb_at = [k for k, s in enumerate(sents) if anchored_figures(store, question, s)]
    if expected is None:                                 # no oracle for this turn: what the words express, in order
        covers, order = _covers(c), []
        for k, r in enumerate(reads):
            cued = sorted((lows[k].find(q), a) for a in r.heard if (q := cue_in(a, lows[k])))
            xs = [a for _, a in cued] or list(r.heard)      # a cue says which act; fields alone only without one
            xs = [a for a in xs if not any(a in covers.get(b, ()) and b not in covers.get(a, ()) for b in xs if b != a)]
            order += [a for a in xs if a not in order]
        acts = [{"act": a} for a in order]
        return {"sentences": [{"text": s, "act": None, "heard": [{"act": a, "how": _how(c, a, r, lows[k], cue_in)}
                                                                for a in r.heard]} for k, (s, r) in enumerate(zip(sents, reads))],
                "says": [], "extra": [], "acts": acts, "from": [], "segments": [(None, text or "")],
                "heard": heard_all, "oracle": False}

    exp = [dict(a) for a in expected]

    def heard_in(i: int, k: int) -> str | None:
        """How sentence k expresses expected act i (None: it does not)."""
        lab = exp[i]
        act = lab.get("act")
        if act == answer_act:
            if kb is not None and k in kb_at:
                return "a figure about what the customer asked"
            return None
        q = cue_in(act, lows[k])
        if q:
            return f"the cue “{q}”"
        need = _need(c, lab)
        said = {x.field for x in reads[k].claims}
        if need and all(x in said for x in need):
            return "the opening statement" if need == ["opening"] else "it says its " + ", ".join(need)
        return None

    names = {e.get("act") for e in exp}
    covers = _covers(c)
    assigned: list[int | None] = []          # the expected act each sentence belongs to (None: none)
    extra_of: dict[int, list[tuple[str, str]]] = {}   # sentence -> acts it says that the oracle does not expect
    used: set[int] = set()
    last: int | None = None                  # the expected act given a sentence last
    cur: int | None = None                   # what the sentence before belonged to
    for k in range(len(sents)):
        cands = [i for i in range(len(exp)) if heard_in(i, k)]
        if cands and not any(i not in used for i in cands):   # only acts already said: unless it says another
            mine = {exp[i].get("act") for i in cands}
            unexpected = [(a, q) for a in reads[k].heard if a not in names and (q := cue_in(a, lows[k]))
                          and not any(a in covers.get(m, ()) for m in mine)]
            if unexpected:
                extra_of[k] = unexpected
                cur = None
                assigned.append(None)
                continue
        if cands:
            fresh = [i for i in cands if i not in used]
            after = [i for i in cands if last is None or i >= last]
            pick = min(fresh) if fresh else (min(after) if after else max(cands))
            used.add(pick)
            last = cur = pick
            assigned.append(pick)
            continue
        own = exp[cur].get("act") if cur is not None else None
        unexpected = [(a, q) for a in reads[k].heard if a not in names and (q := cue_in(a, lows[k]))
                      and a not in covers.get(own, ())]
        if unexpected:                       # it says something the oracle does not: an act of the agent's own
            extra_of[k] = unexpected
            cur = None
            assigned.append(None)
        else:                                # it goes on with what the sentence before said
            assigned.append(cur)

    says = []
    for i, lab in enumerate(exp):
        act = lab.get("act")
        spec = c.acts.get(act)
        mine = [k for k, a in enumerate(assigned) if a == i]
        hows = [h for h in (heard_in(i, k) for k in range(len(sents))) if h]
        need = _need(c, lab)
        if act == answer_act:
            checkable = question is not None and store is not None
        else:
            checkable = bool(spec and (spec.cues or need))
        row = {"act": act, "label": {k: v for k, v in lab.items() if k != "text" or act == "OPENING"},
               "sentences": mine, "how": hows[0] if hows else None, "said": {}, "detail": ""}
        if not checkable:
            row["status"] = "not_checkable"
            row["detail"] = ("no policy question to answer" if act == answer_act else
                             f"{act} has no cue word and nothing it must say: the words cannot confirm it")
        elif not hows and not (act != answer_act and need and all(x in {cl.field for cl in whole.claims} for x in need)):
            row["status"] = "not_heard"
            row["detail"] = "no " + (" or ".join(filter(None, [
                "cue word (" + ", ".join(f"“{q}”" for q in spec.cues[:4]) + (", ..." if len(spec.cues) > 4 else "") + ")"
                if spec and spec.cues else None, ("claim for " + ", ".join(need)) if need else None,
                "figure about what the customer asked" if act == answer_act else None])) or "evidence") + " in the words"
        elif act == answer_act:
            row["status"] = "held" if kb.get("correct") else "contradicted"
            row["detail"] = ("the answer is the policy's" if kb.get("correct") else "the answer is not the policy's") + \
                (f" ({kb.get('detail')})" if kb.get("detail") else "")
            row["kb"] = kb
        else:
            claims = [cl for k in (mine or range(len(sents))) for cl in reads[k].claims]
            bad = []
            for fld, want in lab.items():
                if fld in ("act", "text") or want in (None, "") or isinstance(want, list):
                    continue
                if isinstance(want, dict):              # a record (a summary): {charge: value}
                    vals = {str(v) for v in want.values() if isinstance(v, str)}
                    sets = {g.member_set(v) for v in vals} - {None} if g is not None else set()
                    got = sorted({str(cl.value) for cl in claims if cl.field in sets})
                    if got:
                        row["said"][fld] = got
                        if not set(got) <= vals:
                            bad.append(f"{fld}: the oracle says " + ", ".join(f"{k} = {v}" for k, v in want.items())
                                       + f", the words say {', '.join(got)}")
                    continue
                if isinstance(want, str) and str(want) in (getattr(facts, "items", None) or {}):   # a returned charge
                    got = sorted({str(cl.value) for cl in claims if cl.field == "item"})
                    if got:
                        row["said"][fld] = got
                        if str(want) not in got:
                            bad.append(f"{fld}: the oracle says {want}, the words name {', '.join(got)}")
                    continue
                vs = g.member_set(str(want)) if (g is not None and isinstance(want, str)) else None
                if vs:
                    got = sorted({str(cl.value) for cl in claims if cl.field == vs})
                    if got:
                        row["said"][fld] = got
                        if str(want) not in got:
                            bad.append(f"{fld}: the oracle says {want}, the words say {', '.join(got)}")
                elif isinstance(want, (int, float)) and not isinstance(want, bool):
                    fs = next((f for f in (spec.says if spec else ()) if f.field == fld), None)
                    cf = c.claim_field(fs) if fs is not None else None
                    got = sorted({_num(cl.value) for cl in claims if cf and cl.field == cf
                                  and isinstance(cl.value, (int, float))}, key=str)
                    if got:
                        row["said"][fld] = got
                        if _num(want) not in got:
                            bad.append(f"{fld}: the oracle says {want}, the words say {', '.join(map(str, got))}")
            row["status"] = "contradicted" if bad else "held"
            row["detail"] = "; ".join(bad) if bad else "heard by " + (hows[0] if hows else "its fields")
        says.append(row)

    def narrow(xs: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """A summary repeats "I've noted ...": an act whose cue is part of another act said in the same sentence
        (by that act's own line) is not one of its own."""
        names_x = {a for a, _ in xs}
        return [(a, q) for a, q in xs if not any(a in covers.get(b, ()) and b not in covers.get(a, ())
                                                 for b in names_x if b != a)]
    extra = [{"act": a, "sentence": k, "how": f"the cue \u201c{q}\u201d"} for k, xs in extra_of.items()
             for a, q in narrow(xs)]

    keyed, prev_key = [], -1.0               # the acts in the order the words say them
    for i, (row, lab) in enumerate(zip(says, exp)):
        if row["status"] == "not_heard":
            continue
        a = {k: v for k, v in lab.items() if k != "text" or lab.get("act") == "OPENING"}   # the agent's words, not the oracle's
        if row["status"] == "contradicted":
            for fld, got in row["said"].items():        # graded as said: the words' values, not the oracle's
                if not got:
                    continue
                if isinstance(a.get(fld), dict):
                    a[fld] = {k: (got[0] if len(got) == 1 else v) for k, v in a[fld].items()}
                else:
                    a[fld] = got[0]
        key = float(min(row["sentences"])) if row["sentences"] else prev_key + 0.01
        prev_key = key
        keyed.append((key, i, a))
    for x in extra:
        if all(x["act"] != a.get("act") for _, _, a in keyed):
            keyed.append((x["sentence"] + 0.5, None, {"act": x["act"]}))
    keyed.sort(key=lambda t: t[0])
    acts, src = [a for _, _, a in keyed], [i for _, i, _ in keyed]

    segments, seen = [], set()
    for k, a in enumerate(assigned):
        if a is None:
            if segments and segments[-1][0] is None:
                segments[-1] = (None, segments[-1][1] + " " + sents[k])
            else:
                segments.append((None, sents[k]))
        elif a not in seen:
            seen.add(a)
            segments.append((exp[a], " ".join(sents[j] for j, b in enumerate(assigned) if b == a)))
    return {"sentences": [{"text": s, "act": (exp[assigned[k]].get("act") if assigned[k] is not None else None),
                           "of": assigned[k], "heard": [{"act": a, "how": _how(c, a, reads[k], lows[k], cue_in)}
                                                        for a in reads[k].heard]}
                          for k, s in enumerate(sents)],
            "says": says, "extra": extra, "acts": acts, "from": src, "segments": segments, "heard": heard_all,
            "oracle": True}


_COVERS: dict[int, dict[str, set[str]]] = {}


def _covers(c) -> dict[str, set[str]]:
    """act -> the other acts whose cue words its own line in the pack (``bot.says``) contains: a retry asks for the
    card digits again, so ASK_AUTH's cue in an ASK_AUTH_RETRY line is not an act of its own."""
    if id(c) in _COVERS:
        return _COVERS[id(c)]
    import re
    from ..s01_packs.loading import current
    lines = (current().section("bot") or {}).get("says") or {}
    out: dict[str, set[str]] = {}
    for act, line in lines.items():
        low = words_to_digits(re.sub(r"\[[^\]]*\]|\{[^}]*\}", " ", str(line))).lower()
        out[act] = {o for o, spec in c.acts.items() if o != act and any(q and q in low for q in spec.cues)}
    _COVERS[id(c)] = out
    return out


def _how(c, act: str, read, low: str, cue_in) -> str:
    q = cue_in(act, low)
    if q:
        return f"the cue “{q}”"
    return "its fields"
