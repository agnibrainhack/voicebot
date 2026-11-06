"""The speech check for any pack compiled from a compact graph: what the bot says, read and checked against the
graph, the GMS and the session, with nothing named in code.

The charge-review reader knows its own fields (merchants, amounts, dispositions, card actions, delivery fees).
For a compiled pack every field comes from the graph:

* **values**: every member of every value set a customer slot or a tool result takes, worded as the graph's
  aliases and the pack lexicon's bot phrasings say; a mention is a claim on that value set, *offered* (in a question
  or an offer), *done* (said as completed) or *stated*;
* **figures**: a money figure is a value of the pack's money attribute (``*_usd``), a span of business days a value
  of its days attribute, each about the member named nearest (else the one the customer chose); a number next to a
  limit's unit ("18 years", "4 cards") is that limit;
* **the opening statement**.

Each claim is checked where its truth lives:

    a value said to be done          a successful write of a tool that takes that value (the graph's write tools)
    a value a tool returned          that tool's latest result (an ineligibility reason)
    a value the customer gave        what the customer said
    a figure                         the exact register (``lookup_enm``), the GMS geometry reported alongside
    a limit                          the exact register's limit

Completeness reads each act's required fields by where their truth lives (``says`` bindings: ``caller.<slot>``,
``result.<tool>.<field>``, ``policy.<attr>:caller.<slot>``, ``policy.limit:<key>``, ``text.opening``).
"""
from __future__ import annotations

import re
from typing import Any

from . import reader as _r
from .reader import (ANY, BUSINESS_DAYS, MONEY, Claim, ReadUtterance, UtteranceReader, _dedupe, _nearest, _num,
                     _sentences, _squash, fallback, words_to_digits)


def _kind(attr: str) -> str:
    if attr.endswith("_usd"):
        return "money"
    if "days" in attr:
        return "days"
    return "count"


class PackModel:
    """What the generic check needs to know about a compiled pack, read from its compact graph."""

    def __init__(self, g, lexicon: dict[str, Any] | None = None):
        self.g = g
        self.numeric = g.numeric_attributes()                         # attr -> {set, members}
        self.money = [a for a in self.numeric if _kind(a) == "money"]
        self.days = [a for a in self.numeric if _kind(a) == "days"]
        slot_sets = {sp["set"] for a in (g.data.get("acts") or {}) for sp in g.slot_specs(a).values()
                     if sp["set"] in g.values}
        self.returned = {}                                             # value set -> (tool, field)
        for t, sp in g.tools.items():
            for f, vs in (sp.get("returns") or {}).items():
                if isinstance(vs, str) and vs in g.values:
                    self.returned[vs] = (t, f)
        self.value_sets = [vs for vs in g.values if vs in slot_sets or vs in self.returned]
        self.slot_of_set = {}
        for a in (g.data.get("acts") or {}):
            for sl, sp in g.slot_specs(a).items():
                self.slot_of_set.setdefault(sp["set"], sl)
        self.write_args = {}                                           # value set -> [(tool, arg)]
        for t, sp in g.tools.items():
            if sp.get("role") == "write":
                for arg, vs in (sp.get("args") or {}).items():
                    self.write_args.setdefault(vs, []).append((t, arg))
        self.units = {k: str(lv.get("unit") or "") for k, lv in g.limits.items() if not lv.get("internal")}
        lex = lexicon or {}
        self.phrases: dict[str, list[str]] = {}
        for vs in self.value_sets:
            for m in g.members(vs):
                ws = list(g.aliases(m)) + [m.lower().replace("_", " ")]
                ent = (lex.get("values") or {}).get(m) or {}
                ws += list((ent.get("say") or {}).values()) + list(ent.get("cues") or [])
                self.phrases[m] = sorted({w.lower() for w in ws if w and "{" not in w}, key=len, reverse=True)


class GenericReader(UtteranceReader):
    """Reads a compiled pack's bot turns into claims on its value sets, figures and limits."""

    def __init__(self, contract, model: PackModel):
        super().__init__(contract)
        self.m = model
        self._value_phrases = {vs: sorted({(p, v) for v in model.g.members(vs) for p in model.phrases.get(v, [])},
                                          key=lambda x: -len(x[0])) for vs in model.value_sets}
        self._limit_res = {}
        for k, unit in model.units.items():
            if unit:
                stem = re.escape(unit.rstrip("s"))
                self._limit_res[k] = [re.compile(rf"\b(\d+)\s+(?:\w+\s+)?{stem}s?\b")]

    def read(self, text: str, facts, about_item: str | None = None, question: Any = None) -> ReadUtterance:
        out = ReadUtterance(text=text or "", sentences=_sentences(text or ""))
        if self._opening and self._opening in _squash(text or ""):
            out.claims.append(Claim("opening", True, self.c.opening, text))
        seen: set = set()                                       # (field, about) of the figures said so far
        for sent in out.sentences:
            new = self._sentence(sent, facts, question, seen)
            seen.update((c.field, c.about) for c in new)
            out.claims += new
        out.heard = self._heard(out)
        return out

    def _sentence(self, sent: str, facts, question: Any = None, seen: set | None = None) -> list[Claim]:
        low = words_to_digits(sent).lower()
        asks = bool(_r.QUESTION_OR_CONDITION.search(sent.strip()))
        mode = "offered" if asks or _r.OFFER.search(low) else "done" if _r.COMPLETION.search(low) else "states"
        claims: list[Claim] = []
        named: dict[str, list] = {vs: self._values_named(low, vs) for vs in self.m.value_sets}
        for vs, ms in named.items():
            for mt in ms:
                claims.append(Claim(vs, mt.value, mt.words, sent, None, mode))
        asked = question.get("entity") if question is not None else None

        def about(attr: str, pos: int) -> str | None:
            vs = self.m.numeric[attr]["set"]
            hit = _nearest(named.get(vs) or [], pos)
            if hit or asked:
                return hit or fallback(asked, attr, seen if seen is not None else set())
            return facts.caller.get(self.m.slot_of_set.get(vs, ""))

        days_spans = [(m.start(), m.end()) for m in BUSINESS_DAYS.finditer(low)]
        for mt in MONEY.finditer(low):
            v = _num(mt.group(1) or mt.group(2))
            for attr in self.m.money[:1]:
                claims.append(Claim(attr, v, mt.group(0), sent, about(attr, mt.start())))
        fm = _r.FREE.search(low)
        if fm and self.m.money:
            attr = self.m.money[0]
            if named.get(self.m.numeric[attr]["set"]):
                claims.append(Claim(attr, 0, fm.group(0), sent, about(attr, fm.start())))
        for mt in BUSINESS_DAYS.finditer(low):
            lo, hi = int(mt.group(1)), int(mt.group(2) or mt.group(1))
            for attr in self.m.days[:1]:
                claims.append(Claim(attr, (lo, hi), mt.group(0), sent, about(attr, mt.start())))
        for key, regs in self._limit_res.items():
            for rx in regs:
                for mt in rx.finditer(low):
                    if any(s <= mt.start() < e for s, e in days_spans):
                        continue
                    claims.append(Claim("limit", int(mt.group(1)), mt.group(0), sent, key))
        return _dedupe(claims)

    def _heard(self, r: ReadUtterance) -> list[str]:
        low = words_to_digits(r.text).lower()
        said = {c.field for c in r.claims}
        heard = []
        for act, spec in self.c.acts.items():
            need = [self.c.claim_field(f) for f in spec.says if f.required]
            need = [x for x in need if x]
            if any(q.lower() in low for q in spec.cues) or (need and all(x in said for x in need)):
                heard.append(act)
        return heard


class GenericChecks:
    """Claim checks and completeness for a compiled pack (used by UtteranceVerifier when the contract is generic)."""

    def __init__(self, verifier, model: PackModel):
        self.v, self.m = verifier, model

    def _asks_for(self, state: str) -> set[str]:
        """The value sets a prompt state asks the customer to choose from (its accepted acts' slots)."""
        from ..names import graph_names
        return {sp["set"] for a in graph_names().accepts.get(state, ()) for sp in self.m.g.slot_specs(a).values()
                if sp.get("set")}

    def verify_claim(self, cl: Claim, facts, label: dict | None = None):
        from .verify import ClaimVerdict, _num_eq
        f, m = cl.field, self.m
        store = self.v.store
        if f == "opening":
            return ClaimVerdict(cl, "supported", self.v.c.opening, "required opening statement")
        if f == "limit":
            truth = store.lookup_enm("limit", cl.about)
            ok = _num_eq(cl.value, truth)
            return ClaimVerdict(cl, "supported" if ok else "contradicted", truth, f"exact register limit/{cl.about}",
                                "" if ok else f"{cl.about} is {truth}", self.v._geometry(cl.about, "has_value", cl.value))
        if f in m.numeric:
            members = list(m.numeric[f]["members"])
            truth_of = lambda x: store.lookup_enm(f, x)                       # noqa: E731
            fits = (lambda t: _num_eq(cl.value, t)) if not isinstance(cl.value, tuple) else \
                (lambda t: cl.value[0] <= float(t) and float(t) <= cl.value[1] or _num_eq(cl.value[1], t))
            who = None if cl.about == ANY else cl.about or (label or {}).get(m.slot_of_set.get(m.numeric[f]["set"], ""))
            if who in members:
                truth = truth_of(who)
                ok = fits(truth)
                return ClaimVerdict(cl, "supported" if ok else "contradicted", truth, f"exact register {f}/{who}",
                                    "" if ok else f"{who}: {f} is {truth}",
                                    self.v._geometry(who, f"has_{f}", cl.value if not isinstance(cl.value, tuple)
                                                     else cl.value[1]))
            hits = [x for x in members if fits(truth_of(x))]
            if hits:
                return ClaimVerdict(cl, "supported", cl.value, f"exact register {f}/{hits[0]}")
            return ClaimVerdict(cl, "unsupported", None, f"exact register {f}", f"no member has this {f}")
        if f in m.value_sets:
            summary = [v for k, v in (label or {}).items() if k not in ("act", "text") and isinstance(v, dict)]
            if summary and f in m.write_args:           # a summary of what was written: each value must be one
                done = {str((w.get("args") or {}).get(a)) for t, a in m.write_args[f] for w in facts.writes.get(t, [])}
                ok = str(cl.value) in done
                tools = ", ".join(t for t, _ in m.write_args[f])
                return ClaimVerdict(cl, "supported" if ok else "contradicted", sorted(done), f"successful {tools}",
                                    "" if ok else f"no successful {tools} wrote {cl.value}")
            if cl.mode == "done" and f in m.write_args:
                done = [(t, a, w) for t, a in m.write_args[f] for w in facts.writes.get(t, [])]
                if not done:
                    tools = ", ".join(t for t, _ in m.write_args[f])
                    return ClaimVerdict(cl, "unsupported", None, tools, f"said to be done, but no {tools} succeeded")
                vals = {(w.get("args") or {}).get(a) for _, a, w in done}
                ok = cl.value in vals
                return ClaimVerdict(cl, "supported" if ok else "contradicted", sorted(map(str, vals)),
                                    f"successful {done[-1][0]}", "" if ok else f"the write was for {', '.join(map(str, vals))}")
            if f in m.returned:
                tool, fld = m.returned[f]
                truth = facts.results.get(tool, {}).get(fld)
                if truth is None:
                    return ClaimVerdict(cl, "unsupported" if cl.mode == "states" else "unverifiable", None, tool,
                                        f"{tool} returned no {fld}")
                return self.v._compare(cl, truth, f"{tool} result {fld}")
            slot = m.slot_of_set.get(f)
            truth = facts.caller.get(slot) if slot else None
            asked = (label or {}).get("act")
            states_one = any(isinstance(v, str) and m.g.member_set(v) == f for k, v in (label or {}).items()
                             if k not in ("act", "text"))       # a confirmation names its value: not a choice offered
            if (cl.mode == "offered" and asked and not states_one and f in self._asks_for(asked)
                    and cl.value in m.g.members(f)):
                return ClaimVerdict(cl, "supported", cl.value, f"{asked} asks for a {f}",
                                    "a choice the prompt offers (the graph: it accepts this value set)")
            if truth is None or cl.mode == "offered":
                return ClaimVerdict(cl, "unverifiable", None, "", "mentioned, not asserted" if truth else
                                    "the customer has not said it yet")
            return self.v._compare(cl, truth, "what the customer said")
        return ClaimVerdict(cl, "unverifiable", None, "", f"no check for {f}")

    def completeness(self, act: str, label: dict, read: ReadUtterance, facts) -> list:
        from .verify import Finding
        spec = self.v.c.acts.get(act)
        if spec is None:
            return []
        out = []
        for fs in spec.says:
            if not fs.required:
                continue
            want = self.v.c.claim_field(fs)
            if want is None:
                continue
            src = fs.source
            if src.startswith("caller.") and facts.caller.get(src.split(".", 1)[1]) is None and not label.get(fs.field):
                continue
            if fs.field in label and label[fs.field] in (None, ""):
                continue                              # the act has no value for it (a card action with no delivery)
            if src.startswith("result."):
                tool, _, fld = src[7:].partition(".")
                if facts.results.get(tool, {}).get(fld) is None:
                    continue
            if not read.of(want):
                out.append(Finding("missing", act, f"{act} must say the {fs.field} ({src})", spec.source))
        return out


def claim_field(model: PackModel, binding: str) -> str | None:
    """The claim field a ``says`` binding is read as (None: free text the reader does not read)."""
    kind, _, rest = binding.partition(".")
    if kind == "text" and rest == "opening":
        return "opening"
    if kind == "policy":
        what, _, _ = rest.partition(":")
        return "limit" if what.startswith("limit") else what
    if kind == "caller":
        vs = model.g.slot_set(rest)
        return vs if vs in model.value_sets else None
    if kind == "result":
        tool, _, fld = rest.partition(".")
        vs = (model.g.tools.get(tool, {}).get("returns") or {}).get(fld)
        return vs if isinstance(vs, str) and vs in model.value_sets else None
    if kind == "attribute":
        return None
    return None


def _faithfulness(self, label: dict, read: ReadUtterance) -> list:
    """Do the words say what the label claims? A labelled value of a value set must be the one the words say
    (when they name one); a labelled act must be heard."""
    from .verify import Finding, _num
    act = label["act"]
    spec = self.v.c.acts.get(act)
    out = []
    # an act is judged only when the words could show it: a cue, or a field it must say and has a value for
    # (TRANSFER's reason is optional; a card action with no delivery has no delivery to say)
    need = [self.v.c.claim_field(f) for f in (spec.says if spec else []) if f.required and label.get(f.field) is not None]
    need = [x for x in need if x]
    judgeable = spec and (spec.cues or need)
    said = {c.field for c in read.claims}
    heard = act in read.heard or (need and all(x in said for x in need))
    if judgeable and not heard and read.text.strip():
        out.append(Finding("label_not_said", act, f"labelled {act} but the words do not express it: {read.text!r}"))
    for fld, want in label.items():
        if fld in ("act", "text") or want in (None, ""):
            continue
        vs = self.m.g.member_set(str(want)) if isinstance(want, str) else None
        if vs in self.m.value_sets:
            said = {str(c.value) for c in read.of(vs)}
        elif isinstance(want, (int, float)) and self.m.money:
            said = {str(_num(c.value)) for c in read.of(self.m.money[0])}
            want = _num(want)
        else:
            continue
        if said and str(want) not in said:
            out.append(Finding("text_contradicts_label", act, f"label {fld}={want}, the words say {', '.join(sorted(said))}"))
    return out


GenericChecks.faithfulness = _faithfulness
