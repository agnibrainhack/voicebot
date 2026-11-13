"""Policy in what the bot says, and proof that the speech check catches errors.

Parts of this file:
  1. ``PolicyChecker``: every clause no reader claimed is checked against the graph in order: a figure with a
     unit, a forbidden stance, a promised action, a policy statement; otherwise it is "unread" (advisory). An
     optional LLM judge re-reads policy statements.
  2. Negative controls: edit only the words of one act in a real conversation (wrong merchant, opposite answer,
     wrong fee, invented policy ...) and require the check to flag every edited turn.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable


# =====================================================================================================================
# PART 1 - Policy in what the bot says: every clause is read, checked against the graph, or flagged.
#
# The readers (reader.py, generic.py) read the facts a journey's graph names: values, figures about named things,
# limits in their listed wordings, the stances a rule has phrasings for. What they do not read used to pass unchecked:
# a limit in other words ("up to 120 days"), a policy that does not exist ("we always refund ..."), a promise of an
# action the policy does not have ("I'll waive your fee"). This module reads what is left, clause by clause, against
# what the graph states (``PolicyCorpus``):
#
#   statements  every text the graph carries: rules, limits, tools, the opening, and a compact graph's ``knowledge``
#   figures     every number it states: each limit (with its unit) and each member's numeric attributes
#   actions     what the bot can do: its tools, its acts, and the values its write tools take
#   stances     rules that forbid asserting a fact (R27), with their asserting / denying wordings
#
# Each clause gets one outcome (``PolicyChecker.check``):
#
#   read        a reader made a claim from it, or it carries an act's cue: checked by the readers' checks
#   figure      a number with a unit the graph uses: the graph's figure for the same thing must be that number
#                 supported | contradicted (the graph says another value) | unsupported (no such figure)
#   promise     "I'll / we will / let me <do something>": the action must be one the graph has
#                 supported | unsupported (an action the policy does not provide)
#   statement   a policy-sounding clause ("always", "never", "entitled", "our policy" ...): it must restate one of
#               the graph's statements, and must not assert what a must_not stance forbids
#                 supported (the statement it restates is cited) | unsupported (invented) | contradicted
#   unread      a statement with content none of these read ("before I review charges, I need to authenticate you"):
#               listed and counted, advisory by default; questions and near-empty clauses are not listed
#
# and, in an answer to a policy question:
#
#   volunteered  any fact besides the one asked: a figure about another subject (asked the standard fee, told the
#                expedited fee too), another figure of the asked subject (asked its fee, told its delivery days),
#                a figure naming nothing that is not the asked one, or a policy statement; a policy answer gives
#                exactly the fact asked, so this fails the turn unless the project lists it as advisory
#                (project.yaml ``speech.policy.advisory``)
#   disclosed    anything the graph marks internal, or the project lists under ``speech.policy.never_volunteer``:
#                a failure whether asked or not
#
# A clause is matched to a statement on shared word stems weighted by how rare they are among the statements, plus
# equal figures. That decides which statement a clause is about; it does not prove a paraphrase keeps its meaning.
# ``speech.policy.judge: llm`` adds an LLM judge for policy-sounding clauses: given the candidate statements it says
# which one the clause restates or contradicts, or that it states policy the graph does not have, and must cite a
# statement id for a supported or contradicted verdict (its answers are cached, so a campaign replays exactly).
#
# Nothing here names a journey: the plain-English markers have defaults below (``speech.policy.markers`` overrides
# them) and everything else comes from the graph.
# =====================================================================================================================

from .speech_reading import ANY, Claim, ReadUtterance, words_to_digits

# ------------------------------------------------------------------ plain-English markers (not policy)
MARKERS = {
    # a clause that states a rule as a rule
    "policy": ["always", "never", "guarantee", "guaranteed", "guarantees", "entitled", "our policy", "the policy",
               "policy is", "by policy", "allowed", "not allowed", "permitted", "eligible", "ineligible",
               "every time", "no matter", "automatically", "at no cost", "for free", "no charge", "waived",
               "refundable", "non-refundable", "is proof", "means fraud", "counts as", "required by law"],
    # a clause in which the bot commits to doing something
    "commitment": ["i'll", "i will", "we'll", "we will", "i can", "we can", "let me", "i'm going to", "i am going to",
                   "we're going to", "we are going to", "you'll get", "you will get", "you'll receive",
                   "you will receive", "you'll be", "you will be", "i've gone ahead and", "i have gone ahead and"],
    # what a bot does in any conversation: never a policy promise
    "procedural": ["help", "assist", "check", "look", "see", "find", "pull", "verify", "confirm", "ask", "tell",
                   "say", "share", "explain", "walk", "review", "read", "repeat", "note", "start", "begin",
                   "continue", "proceed", "move", "get", "take", "handle", "sort", "make", "try", "wait", "hold",
                   "give", "put", "connect", "transfer", "pass", "end", "close", "stay", "keep", "go", "come", "let"],
    # words between the commitment and its verb ("I'll [also just go ahead and] ...")
    "filler": ["also", "just", "now", "then", "first", "next", "quickly", "gladly", "certainly", "definitely",
               "right", "away", "ahead", "and", "be", "happy", "glad", "able", "sure", "to", "go", "need", "have",
               "please", "really", "still", "again", "back", "here", "there", "today", "immediately"],
    # a count that is not a policy figure ("you have 1 more attempt")
    "remaining": ["more", "left", "remaining", "another", "other", "next", "last", "previous", "first", "second"],
}
STOP = set("""a an the and or but so to of for in on at by with from as is are was were be been being it its it's this
that these those there here i you we they he she me my your our us them his her their will would can could shall should
may might must do does did done have has had not no yes if then than also just only any all each every one ones which
who whom what when where why how about into over under up down out off again further once more most other some such
own same too very s t can't cannot won't don't doesn't isn't aren't wasn't weren't hasn't haven't hadn't i'm i've i'll
you're you've you'll we're we've we'll it'll that's there's let get got please thanks thank okay ok sure""".split())
NEGATION = re.compile(r"\b(not|no|never|isn't|aren't|wasn't|weren't|doesn't|don't|didn't|can't|cannot|won't|"
                      r"without|nothing|none|neither|nor)\b|n't\b")


def markers() -> dict[str, list[str]]:
    """The marker lists: the defaults, with any the project sets (project.yaml speech.policy.markers)."""
    out = {k: list(v) for k, v in MARKERS.items()}
    try:
        from ..s01_packs.project import project
        over = ((project().raw.get("speech") or {}).get("policy") or {}).get("markers") or {}
    except Exception:                                   # noqa: BLE001 - a pack without a project: the defaults
        over = {}
    for k, v in over.items():
        out[k] = list(v)
    return out


def settings() -> dict[str, Any]:
    """project.yaml speech.policy: advisory (finding kinds that never fail a turn), never_volunteer, judge."""
    try:
        from ..s01_packs.project import project
        sp = ((project().raw.get("speech") or {}).get("policy") or {})
    except Exception:                                   # noqa: BLE001
        sp = {}
    return {"advisory": list(sp.get("advisory", ["unread"])),
            "never_volunteer": list(sp.get("never_volunteer") or []),
            "judge": str(sp.get("judge") or "graph"), "llm_model": sp.get("llm_model")}


def _re(words: list[str]) -> re.Pattern:
    return re.compile(r"(?<![a-z'])(" + "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True)) + r")(?![a-z'])",
                      re.I)


# ------------------------------------------------------------------ words
_SUFFIXES = ("ations", "ation", "ingly", "ings", "ing", "ulent", "ously", "ous", "ness", "ments", "ment", "edly", "ies",
             "ied", "ed", "es", "ly", "s")


def stem(w: str) -> str:
    """A crude stem shared by a word's common forms: 'transactions'/'transaction' -> 'transa', 'charges' -> 'charg',
    'eligible'/'eligibility' -> 'eligib' (prefixes of six letters after stripping a common ending)."""
    w = w.lower().strip("'")
    for pre in ("non", "un"):                       # 'unrecognized' and 'recognizing' are about the same thing
        if w.startswith(pre) and len(w) > len(pre) + 5:
            w = w[len(pre):]
            break
    for suf in _SUFFIXES:
        if len(w) > len(suf) + 3 and w.endswith(suf):
            w = w[: -len(suf)]
            break
    w = w.rstrip("e") if len(w) > 4 else w
    return w[:6]


def terms(text: str) -> list[str]:
    """The content stems of a text (no stop words, nothing shorter than three letters)."""
    return [stem(w) for w in re.findall(r"[a-z][a-z']+", (text or "").lower()) if w not in STOP and len(w) > 2]


def clauses(sentence: str) -> list[str]:
    """A sentence cut where a new statement starts: at ';', and at ', and' / ', but' / ', so' / ', plus' / ', while'."""
    parts = re.split(r";\s*|,\s*(?:and|but|so|plus|while|although|though|whereas)\s+|\s+(?:and also|but also)\s+",
                     sentence)
    return [p.strip(" ,.") for p in parts if p and p.strip(" ,.")]


NUMBER = re.compile(r"(?<![\d.$])(\d+(?:\.\d+)?)(?:\s*(?:-|–|to)\s*(\d+(?:\.\d+)?))?\s+((?:[a-z]+\s+){0,2}?[a-z]+)",
                    re.I)


# ------------------------------------------------------------------ what the graph states
@dataclass(frozen=True)
class Statement:
    """A text the graph carries: a rule, a limit, a tool's description, the opening or a knowledge line."""
    sid: str
    text: str
    kind: str                       # rule | limit | tool | knowledge | opening
    stems: frozenset
    subjects: frozenset             # the members and limits it names
    stance: str | None = None
    internal: bool = False


@dataclass(frozen=True)
class Figure:
    """A number the graph states: a limit (with its unit), or a member's numeric attribute."""
    subject: str
    attr: str
    value: Any                      # a number, or (low, high) for a range of days
    kind: str                       # money | days | count
    units: tuple                    # unit word stems this figure is said in
    internal: bool = False
    unit: str = ""                  # the unit as the graph writes it ("calendar_days")


@dataclass
class PolicyCorpus:
    """Everything the policy check compares a clause with, read from the graph: statements, figures with
    their units, how subjects are worded, action verbs, forbidden stances, and word rarity (idf) for matching.
    """
    statements: list[Statement]
    figures: list[Figure]
    subjects: dict[str, list[str]]  # member / limit -> how it is worded (lower case)
    kinds: dict[str, str]           # member / limit -> its kind (value set name, or "limit")
    action_stems: set
    stances: list[dict[str, Any]]   # {rule, assert: regex | None, asserted: [...], denied: [...]}
    unit_stems: set
    idf: dict[str, float] = field(default_factory=dict)
    action_phrases: list = field(default_factory=list)   # how the write tools' values are worded

    @classmethod
    def from_contract(cls, contract, kg, graph=None) -> "PolicyCorpus":
        """Everything from the graph: ``kg`` (its triples), ``contract`` (wordings, stances) and, for a compact graph,
        ``graph`` (its knowledge lines and which limits are internal)."""
        triples = list(getattr(kg, "triples", []) or [])
        tails = lambda h, r: [t["tail"] for t in triples if t["head"] == h and t["relation"] == r]  # noqa: E731
        limits = [t["head"] for t in triples if t["relation"] == "is_a" and t["tail"] == "limit"]
        internal = {k for k, lv in ((graph.limits if graph is not None else {}) or {}).items() if lv.get("internal")}
        # how members and limits are worded
        subjects: dict[str, list[str]] = {}
        kinds: dict[str, str] = {}
        for vs, vals in (contract.value_sets or {}).items():
            for v in vals:
                ws = {p.lower() for p in contract.phrases.get(v, []) if p}
                ws.add(str(v).lower().replace("_", " "))
                subjects[v] = sorted(ws, key=len, reverse=True)
                kinds[v] = vs
        for vs in (graph.values if graph is not None else {}):            # every value set, not only spoken ones
            for v in graph.members(vs):
                if v not in subjects:
                    subjects[v] = sorted({str(v).lower().replace("_", " "), *[a.lower() for a in graph.aliases(v)]},
                                         key=len, reverse=True)
                    kinds[v] = vs
        if graph is None:                                                 # a PDF graph: its member types
            for t in triples:
                if t["relation"] == "is_a" and t["tail"] not in ("limit", "rule", "tool", "dialog_state", "dialog_act",
                                                                 "escalation_trigger", "value_set", "tool_step", "policy"):
                    if t["head"] not in subjects:
                        subjects[t["head"]] = [str(t["head"]).lower().replace("_", " ")]
                        kinds[t["head"]] = str(t["tail"])
        for k in limits:
            kinds[k] = "limit"
        # figures
        figures: list[Figure] = []
        unit_stems: set = {stem("day"), stem("days")}
        for k in limits:
            try:
                val = kg.limit(k)
            except Exception:                                             # noqa: BLE001
                val = (tails(k, "has_value") or [None])[0]
            if val is None:
                continue
            unit = str((tails(k, "has_unit") or [""])[0] or "")
            words = [w for w in re.split(r"[_\s]+", unit.lower()) if w]
            us = tuple(sorted({stem(w) for w in words}))
            unit_stems.update(us)
            kind = "days" if any("day" in w for w in words) else "count"
            figures.append(Figure(k, "limit", val, kind, us, k in internal, unit))
        ranges: dict[tuple, dict] = {}
        for t in triples:
            r, h, v = t["relation"], t["head"], t["tail"]
            if not r.startswith("has_") or r == "has_value" or h in limits or r[4:] in limits:
                continue
            try:
                num = float(v)
            except (TypeError, ValueError):
                continue
            if isinstance(v, bool):
                continue
            attr = r[4:]
            num = int(num) if num.is_integer() else num
            if attr.endswith(("_min", "_max")):
                ranges.setdefault((h, attr[:-4]), {})[attr[-3:]] = num
                continue
            kind = "money" if attr.endswith("_usd") else "days" if "days" in attr else "count"
            figures.append(Figure(h, attr, num, kind, (stem("days"),) if kind == "days" else ()))
        for (h, attr), mm in ranges.items():
            lo, hi = mm.get("min", mm.get("max")), mm.get("max", mm.get("min"))
            kind = "money" if attr.endswith("_usd") else "days" if "days" in attr else "count"
            figures.append(Figure(h, attr, (lo, hi), kind, (stem("days"),) if kind == "days" else ()))
        # statements
        stmts: list[Statement] = []
        rule_stance = {t["head"]: str(t["tail"]) for t in triples if t["relation"] == "has_stance"}

        def named(text: str) -> frozenset:
            low = text.lower()
            return frozenset(s for s, ws in subjects.items() if any(re.search(r"\b" + re.escape(w) + r"\b", low) for w in ws))
        for t in triples:
            if t["relation"] != "has_text" or not str(t["tail"]).strip():
                continue
            h, text = t["head"], str(t["tail"])
            kind = "limit" if h in limits else "rule" if h in rule_stance else "opening" if "opening" in h else "tool"
            stmts.append(Statement(h, text, kind, frozenset(terms(text)), named(text) | ({h} if h in limits else set()),
                                   rule_stance.get(h), h in internal))
        for i, text in enumerate((graph.data.get("knowledge") or []) if graph is not None else []):
            stmts.append(Statement(f"K{i + 1}", str(text), "knowledge", frozenset(terms(text)), named(str(text))))
        # actions: the verbs of the tools, and the values the write tools take (how each is worded)
        action_stems: set = set()
        action_phrases: list[str] = []
        tools = [t["head"] for t in triples if t["relation"] == "is_a" and t["tail"] == "tool"]
        for name in tools:
            first = re.split(r"[_\s]+", str(name).lower())[0]
            if len(first) > 2:
                action_stems.add(stem(first))
        write_sets = set()
        if getattr(contract, "model", None) is not None:
            write_sets = set(contract.model.write_args)
        else:                                             # a PDF graph: the value sets the bot speaks of
            write_sets = set(contract.value_sets)
        for vs in write_sets:                             # a value by its own name ("replace card"), not by how
            for v in contract.value_sets.get(vs, ()) or (graph.members(vs) if graph is not None else ()):
                w = str(v).lower().replace("_", " ")      # customers say it ("that's mine")
                action_stems.update(terms(w))
                action_phrases.append(w)
        # stances that may not be asserted
        stances = []
        for rid, sp in (contract.stances or {}).items():
            stances.append({"rule": rid, "assert": None, "asserted": [p.lower() for p in sp.asserted],
                            "denied": [p.lower() for p in sp.denied]})
        for rid, spec in ((graph.rules if graph is not None else {}) or {}).items():
            meta = spec.get("check") if isinstance(spec, dict) else (spec[2] if len(spec) > 2 else None)
            for m in (meta if isinstance(meta, list) else [meta]):
                if isinstance(m, dict) and m.get("pattern") and m.get("forbidden_stance") \
                        and not any(s["rule"] == rid for s in stances):
                    stances.append({"rule": rid, "assert": re.compile(m["pattern"], re.I), "asserted": [], "denied": []})
        c = cls(stmts, figures, subjects, kinds, action_stems, stances, unit_stems)
        c.action_phrases = sorted(set(action_phrases), key=len, reverse=True)
        n = max(1, len(stmts))
        df: dict[str, int] = {}
        for s in stmts:
            for x in s.stems:
                df[x] = df.get(x, 0) + 1
        c.idf = {x: math.log(1 + n / d) for x, d in df.items()}
        return c

    # ---------------------------------------------------------------- lookups
    def named_in(self, text: str) -> list[str]:
        """The members and limits a text names."""
        low = text.lower()
        return [s for s, ws in self.subjects.items() if any(re.search(r"\b" + re.escape(w) + r"\b", low) for w in ws)]

    def match(self, text: str, numbers: list[float] = ()) -> tuple[Statement | None, float]:
        """The statement a clause restates: the most shared rare stems (and equal figures). (None, 0) when no
        statement shares at least two pieces of evidence, or one when the clause has nothing else to say."""
        st = set(terms(text))
        if not st and not numbers:
            return None, 0.0
        best, score = None, 0.0
        for s in self.statements:
            shared = st & s.stems
            figs = [n for n in numbers if any(_eq(n, f.value) for f in self.figures if f.subject in s.subjects
                                              or str(n) in s.text or f"{int(n) if float(n).is_integer() else n}" in s.text)]
            evidence = len(shared) + len(figs)
            if evidence == 0:
                return_ok = False
            else:
                rest = len(st - s.stems)
                return_ok = evidence >= 2 or (evidence == 1 and rest <= 1)
            if not return_ok:
                continue
            w = sum(self.idf.get(x, 1.0) for x in shared) + 2.0 * len(figs)
            if w > score:
                best, score = s, w
        return best, score


def _eq(a: Any, b: Any) -> bool:
    try:
        if isinstance(b, (tuple, list)):
            return float(b[0]) <= float(a) <= float(b[1])
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return False


# ------------------------------------------------------------------ the checker
@dataclass
class ClauseVerdict:
    """What one clause was found to say about policy."""
    clause: str
    sentence: str
    outcome: str                    # figure | promise | statement | unread
    status: str                     # supported | contradicted | unsupported | unverifiable
    cites: str | None = None        # the statement / figure it was checked against
    truth: Any = None
    why: str = ""
    subjects: tuple = ()
    internal: bool = False
    judge: str = "graph"            # graph | llm

    def failed_for_disclosure(self, asked: set) -> bool:
        """True when this verdict cannot be a disclosure: a failed check, or about what the customer asked."""
        return self.status not in ("supported", "contradicted") or (not self.internal and bool(set(self.subjects) & asked))

    def as_claim(self) -> Claim:
        """The verdict as a speech claim, so it is reported and counted with the readers' claims."""
        fld = {"figure": "policy_figure", "promise": "promise", "statement": "policy"}.get(self.outcome, "unread")
        return Claim(fld, self.cites or self.clause, self.clause, self.sentence,
                     (self.subjects[0] if self.subjects else None), "states")


class PolicyChecker:
    """Reads the clauses the readers left, against the graph (see the module docstring)."""

    def __init__(self, contract, kg, graph=None, judge=None):
        self.c = contract
        self.corpus = PolicyCorpus.from_contract(contract, kg, graph)
        self.cfg = settings()
        mk = markers()
        self._policy = _re(mk["policy"])
        self._commit = _re(mk["commitment"])
        self._procedural = {stem(w) for w in mk["procedural"]} | {w for w in mk["procedural"]}
        self._filler = set(mk["filler"])
        self._remaining = _re(mk["remaining"])
        self._cues = [q.lower() for a in contract.acts.values() for q in a.cues if q]
        self.judge = judge

    # ---------------------------------------------------------------- one utterance
    def check(self, read: ReadUtterance, question: Any = None, asked: set | None = None) -> list[ClauseVerdict]:
        """Every clause of the utterance that the readers did not read: its policy outcome."""
        out: list[ClauseVerdict] = []
        for sent in read.sentences:
            mine = [c for c in read.claims if c.sentence == sent]
            for cl in clauses(sent):
                low = words_to_digits(cl).lower()
                covered = [c for c in mine if str(c.words).lower() and str(c.words).lower() in low]
                v = self._clause(cl, sent, low, covered)
                if v is not None:
                    out.append(v)
        return out

    def _clause(self, clause: str, sent: str, low: str, covered: list[Claim]) -> ClauseVerdict | None:
        # 1. stray figures: a number with a unit the graph uses, not already read as a claim
        claimed = " ".join(str(c.words).lower() for c in covered)
        for m in NUMBER.finditer(low):
            if m.group(0).lower() in claimed or any(m.group(1) in str(c.words) for c in covered):
                continue
            unit_words = m.group(3).split()
            ustem = [stem(w) for w in unit_words]
            if not any(u in self.corpus.unit_stems for u in ustem):
                continue
            ctx = low[max(0, m.start() - 24): m.end() + 24]
            if self._remaining.search(ctx):
                continue                                    # "1 more attempt", "the last 2 days": not a policy figure
            lo = float(m.group(1))
            hi = float(m.group(2)) if m.group(2) else lo
            return self._figure(clause, sent, low, (lo, hi), ustem)
        if covered:
            # read by a reader; a forbidden stance can still be asserted around it (unless a reader read the stance)
            return None if any(c.field == "stance" for c in covered) else self._stance(clause, sent, low)
        st = self._stance(clause, sent, low)
        if st is not None:
            return st
        # 2. promises
        cm = self._commit.search(low)
        if cm:
            v = self._promise(clause, sent, low[cm.end():])
            if v is not None:
                return v
        # 3. policy-sounding statements
        if self._policy.search(low):
            return self._statement(clause, sent, low)
        if any(q in low for q in self._cues):
            return None                                     # an act's cue ("anything else"): read
        if sent.rstrip().endswith("?"):
            return None                                     # a question asserts nothing
        content = [t for t in terms(clause) if t not in self._procedural]
        if len(content) < 2:
            return None                                     # "That didn't match", "Thanks": nothing to read
        return ClauseVerdict(clause, sent, "unread", "unverifiable", why="no policy content the checks read")

    # ---------------------------------------------------------------- outcomes
    def _figure(self, clause, sent, low, value, ustem) -> ClauseVerdict:
        kind_days = any("day" in u for u in ustem)
        cands = [f for f in self.corpus.figures if (kind_days and f.kind == "days")
                 or (not kind_days and f.units and set(f.units) & set(ustem))]
        named = set(self.corpus.named_in(clause))
        if named & {f.subject for f in cands}:
            cands = [f for f in cands if f.subject in named]
        st, _ = self.corpus.match(clause, [value[0]])
        if st is not None and any(f.subject in st.subjects for f in cands):
            topical = [f for f in cands if f.subject in st.subjects]
        else:
            topical = cands
        said = value[0] if value[0] == value[1] else value

        def fits(f: Figure) -> bool:
            if isinstance(f.value, (tuple, list)):
                return float(f.value[0]) <= value[0] and value[1] <= float(f.value[1])
            return value[0] == value[1] and abs(float(f.value) - value[0]) < 1e-9
        hit = next((f for f in topical if fits(f)), None) or next((f for f in cands if fits(f)), None)
        words = " ".join(ustem)
        if hit is not None:
            return ClauseVerdict(clause, sent, "figure", "supported", f"{hit.subject}.{hit.attr}", hit.value,
                                 f"the graph's {hit.attr} for {hit.subject}", (hit.subject,), hit.internal)
        if topical:
            f = topical[0]
            return ClauseVerdict(clause, sent, "figure", "contradicted", f"{f.subject}.{f.attr}", f.value,
                                 f"said {_fmt(said)} {words}; the graph's {f.attr} for {f.subject} is {_fmt(f.value)}",
                                 (f.subject,), f.internal)
        return ClauseVerdict(clause, sent, "figure", "unsupported", None, None,
                             f"said {_fmt(said)} {words}; the graph states no such figure")

    def _promise(self, clause, sent, rest) -> ClauseVerdict | None:
        """The main verb after the commitment ("I'll [also] waive ...") must be an action the graph has: a tool's
        verb or a value its write tools take; or the clause names such a value ("a replacement card")."""
        words = [w for w in re.findall(r"[a-z][a-z']+", rest) if w not in self._filler and w not in STOP]
        main = words[0] if words else None
        if main is None or main in self._procedural or stem(main) in self._procedural:
            return None                                     # "I'll check that", "let me transfer you": procedural
        low = rest.lower()
        if stem(main) in self.corpus.action_stems:
            return ClauseVerdict(clause, sent, "promise", "supported", None, None, f"'{main}' is an action the graph has")
        phrase = next((p for p in self.corpus.action_phrases if re.search(r"\b" + re.escape(p) + r"\b", low)), None)
        if phrase:
            return ClauseVerdict(clause, sent, "promise", "supported", None, None,
                                 f"names '{phrase}', a value the graph's write tools take")
        return ClauseVerdict(clause, sent, "promise", "unsupported", None, None,
                             f"commits to '{main}', which is no action the graph provides (its tools' verbs and the "
                             f"values its write tools take)", tuple(self.corpus.named_in(clause)))

    def _statement(self, clause, sent, low) -> ClauseVerdict:
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", low)]
        st, _ = self.corpus.match(clause, nums)
        if st is None:
            v = ClauseVerdict(clause, sent, "statement", "unsupported", None, None,
                              "states policy that none of the graph's rules, limits or knowledge lines states",
                              tuple(self.corpus.named_in(clause)))
        else:
            v = ClauseVerdict(clause, sent, "statement", "supported", st.sid, st.text,
                              f"restates {st.sid}: {st.text}", tuple(sorted(st.subjects)), st.internal)
        return self.judge.review(v, self.corpus) if self.judge is not None else v

    def _stance(self, clause, sent, low) -> ClauseVerdict | None:
        """A rule's fact asserted when the rule forbids asserting it (R27: 'this means fraud')."""
        for s in self.corpus.stances:
            if any(p in low for p in s["denied"]):
                return ClauseVerdict(clause, sent, "statement", "supported", s["rule"], None,
                                     f"denies what {s['rule']} forbids asserting")
            hit = (s["assert"].search(low) if s["assert"] is not None else None) or \
                next((p for p in s["asserted"] if p in low), None)
            if hit:
                before = low[: (hit.start() if hasattr(hit, "start") else low.find(hit))]
                if NEGATION.search(before[-30:]) or "?" in clause:
                    continue
                return ClauseVerdict(clause, sent, "statement", "contradicted", s["rule"], None,
                                     f"asserts what {s['rule']} forbids asserting")
        return None

    # ---------------------------------------------------------------- relevance
    def asked_subjects(self, question: Any) -> set[str]:
        """What a policy question asks about: its entity or limit key."""
        if question is None:
            return set()
        return {x for x in (question.get("entity"), question.get("key")) if x}

    def volunteered(self, claim, question: Any) -> bool:
        """In an answer to a policy question, a fact besides the one asked: another subject, another figure of the
        asked subject, a figure naming nothing that is not the asked one, or a policy statement (not a promise)."""
        if question is None or claim.field in ("promise", "opening", "item", "amount"):   # a charge is not policy
            return False
        ent, attr, key = question.get("entity"), question.get("attr"), question.get("key")
        asked = {x for x in (ent, key) if x}
        if not asked or str(claim.value) in asked:       # naming what was asked is not telling anything else
            return False
        want = [f for f in self.corpus.figures if (f.subject == ent and attr and f.attr == attr)
                or (key and f.subject == key)]
        same = lambda a, b: _num(a) == _num(b)                     # noqa: E731
        is_asked = any(same(claim.value, f.value) for f in want)
        about = claim.about
        if about in asked:
            if claim.field == "policy" or is_asked or not want:
                return False
            # another figure of the asked subject: one the graph states for it under another attribute
            return any(f.subject == about and f.attr != attr and same(claim.value, f.value) for f in self.corpus.figures)
        if not about or about == ANY:
            return not is_asked
        return True

    def never(self, subjects: tuple | list, internal: bool) -> str | None:
        """Why saying this is a disclosure: an internal limit, or a subject the project never volunteers."""
        if internal:
            return "an internal figure the graph does not let the bot tell the customer"
        lst = set(self.cfg["never_volunteer"])
        hit = next((s for s in subjects if s in lst), None)
        return f"{hit} is on the project's never_volunteer list" if hit else None


def _num(v: Any) -> Any:
    """A figure's value for comparing (ranges as pairs of floats, numbers as floats)."""
    if isinstance(v, (tuple, list)):
        return tuple(_num(x) for x in v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def _fmt(v: Any) -> str:
    if isinstance(v, (tuple, list)):
        a, b = (_fmt(v[0]), _fmt(v[1]))
        return a if a == b else f"{a}-{b}"
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else str(f)
    except (TypeError, ValueError):
        return str(v)


# ------------------------------------------------------------------ the optional LLM judge
JUDGE_SYSTEM = """You check one clause a customer-service bot said against the policy statements of its journey.
Answer with JSON only: {"verdict": "restates" | "contradicts" | "invented" | "not_policy", "statement": "<id or null>",
"why": "<one sentence>"}.
restates: the clause says what one statement says (same meaning; other words are fine). Give its id.
contradicts: the clause says the opposite of a statement, or a different figure. Give its id.
invented: the clause states a policy, rule, entitlement or figure that no statement supports.
not_policy: the clause states no policy (small talk, a procedural step)."""


class LLMPolicyJudge:
    """Re-reads policy-sounding clauses with an LLM given the graph's closest statements (project.yaml
    ``speech.policy.judge: llm``). A restates / contradicts verdict must cite one of the statements offered, or it
    is ignored; answers are cached (enrich/llm/client.CachedLLM), so a campaign replays them exactly. When the LLM
    is unavailable the graph's verdict stands and the reason is kept in ``errors``."""

    def __init__(self, model: str | None = None, llm: Any = None, k: int = 6):
        if llm is None:
            from ..llm import CachedLLM, LLMSettings
            llm = CachedLLM(settings=LLMSettings(model=model or "", temperature=0.0, role="testing"), tag="policy-judge")
        self.llm, self.k = llm, k
        self.errors: list[str] = []
        self.calls = 0

    def candidates(self, clause: str, corpus: PolicyCorpus) -> list[Statement]:
        st = set(terms(clause))
        scored = sorted(corpus.statements, key=lambda s: -sum(corpus.idf.get(x, 1.0) for x in st & s.stems))
        return scored[: self.k]

    def review(self, v: ClauseVerdict, corpus: PolicyCorpus) -> ClauseVerdict:
        cands = self.candidates(v.clause, corpus)
        user = "Statements:\n" + "\n".join(f"{s.sid}: {s.text}" for s in cands) + f"\n\nClause: {v.clause}"
        self.calls += 1
        try:
            ans = self.llm.invoke_json(JUDGE_SYSTEM, user, salt="policy-judge/1") or {}
        except Exception as e:                       # noqa: BLE001 - the graph's verdict stands
            self.errors.append(f"{type(e).__name__}: {str(e)[:160]}")
            return v
        verdict, sid, why = str(ans.get("verdict") or ""), ans.get("statement"), str(ans.get("why") or "")
        by_id = {s.sid: s for s in cands}
        if verdict in ("restates", "contradicts") and sid in by_id:
            s = by_id[sid]
            return ClauseVerdict(v.clause, v.sentence, "statement", "supported" if verdict == "restates" else "contradicted",
                                 s.sid, s.text, f"LLM judge: {why} ({s.sid}: {s.text})", tuple(sorted(s.subjects)),
                                 s.internal, "llm")
        if verdict == "invented":
            return ClauseVerdict(v.clause, v.sentence, "statement", "unsupported", None, None, f"LLM judge: {why}",
                                 v.subjects, False, "llm")
        if verdict == "not_policy":
            return ClauseVerdict(v.clause, v.sentence, "unread", "unverifiable", None, None, f"LLM judge: {why}",
                                 v.subjects, False, "llm")
        return v


# =====================================================================================================================
# PART 2 - Checking the checker: negative controls on what the bot says.
#
# A check that never fires proves nothing. Each control below changes ONLY the words of one act in a
# real conversation (the act labels, the tool calls and the outcome stay exactly as they were) in a
# way that makes the words false, incomplete or unfaithful. The speech check must flag the turn where
# the words changed. The detection rate per control is how much the check can be trusted; the
# unchanged conversations (the false-alarm baseline) must stay clean.
#
# Input   a runner factory (``bot_factory -> JourneyRunner``), the reference bot factory and scenarios.
# Output  {control: {applied, detected, detection_rate, missed: [examples]}} and the false-alarm count.
#
#     from gmsjourney.s07_checks.speech_policy import run_controls
#     res = run_controls(lambda f: JourneyRunner(bot_factory=f, **kw), make_reference_factory(store), scenarios)
#     res["controls"]["flip_disposition"]["detection_rate"]   # 1.0
#
# The controls (what the customer would hear, against what is true):
#   swap_merchant      a charge presented with another returned charge's merchant
#   change_amount      a charge's amount one dollar off
#   shift_date         a charge's date one day off
#   drop_date          a charge presented without its date (R6: merchant, amount and date)
#   flip_disposition   a recorded answer said as its opposite
#   wrong_fee          a delivery fee ten dollars off
#   wrong_days         a delivery time of 9 business days
#   wrong_limit        the transaction limit one too many
#   claim_done_early   "your card has been replaced" said before any card action succeeded
#   assert_fraud       "this means fraud" (R27 says it is not, by itself, proof of fraud)
#   drop_summary_item  a summary that leaves out a recorded response (R19)
#
# Policy controls (every pack, built from its graph; s07_checks/speech_policy.py must flag them):
#   invented_policy        "Our policy is to always refund the full amount on request." (no such statement)
#   wrong_policy_figure    a customer-facing limit restated in other words with the wrong number
#   unauthorised_promise   "I'll also waive any fees ..." (a verb no tool of the graph performs)
#   internal_disclosure    an internal limit (graph) or a never_volunteer one (project) told to the customer
# =====================================================================================================================

from ..s01_packs.names import N
from ..s01_packs import names as roles

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]


# ------------------------------------------------------------------ the edits
# Each edit takes (act, its text, ctx) and returns the new text, or None when it does not apply to this act.
# ctx: {"merchants": {transaction_id: merchant} returned so far, "submitted": a card action has succeeded}
def _swap_merchant(act: dict, text: str, ctx: dict) -> str | None:
    """PRESENT_TXN: name another returned charge's merchant (the amount and date stay this charge's)."""
    if act.get("act") != N.state.PRESENT_TXN or len(ctx["merchants"]) < 2:
        return None
    mine = ctx["merchants"].get(act.get("txn"))
    other = next((m for t, m in ctx["merchants"].items() if t != act.get("txn")), None)
    return text.replace(mine, other) if mine and other and mine in text else None


def _change_amount(act: dict, text: str, ctx: dict) -> str | None:
    """PRESENT_TXN / PROVIDE_CONTEXT: the charge's amount one dollar higher."""
    if act.get("act") not in (N.state.PRESENT_TXN, N.state.PROVIDE_CONTEXT):
        return None
    m = re.search(r"\$(\d+\.\d{2})", text)
    return text[:m.start()] + f"${float(m.group(1)) + 1:.2f}" + text[m.end():] if m else None


def _shift_date(act: dict, text: str, ctx: dict) -> str | None:
    """PRESENT_TXN: the charge's date one day off."""
    m = re.search(r"\b(" + "|".join(MONTHS) + r") (\d{1,2})\b", text)
    if act.get("act") != N.state.PRESENT_TXN or not m:
        return None
    d = int(m.group(2))
    return text[:m.start()] + f"{m.group(1)} {d + 1 if d < 28 else d - 1}" + text[m.end():]


def _drop_date(act: dict, text: str, ctx: dict) -> str | None:
    """PRESENT_TXN: leave the date out (pdf.2B requires merchant, amount and date)."""
    m = re.search(r" on (" + "|".join(MONTHS) + r") \d{1,2}", text)
    return text[:m.start()] + text[m.end():] if act.get("act") == N.state.PRESENT_TXN and m else None


def _flip_disposition(act: dict, text: str, ctx: dict) -> str | None:
    """CONFIRM_RECORDED: say the opposite of the answer that was recorded."""
    if act.get("act") != N.state.CONFIRM_RECORDED or act.get("disposition") not in (N.value.RECOGNIZED, N.value.UNRECOGNIZED):
        return None
    said = "you don't recognize it" if act["disposition"] == N.value.RECOGNIZED else "you recognize it"
    return f"Thanks, I've recorded that {said}."


def _wrong_fee(act: dict, text: str, ctx: dict) -> str | None:
    """DISCLOSE_FEE: the fee ten dollars higher."""
    m = re.search(r"\$(\d+)", text)
    return text[:m.start()] + f"${int(m.group(1)) + 10}" + text[m.end():] if act.get("act") == N.state.DISCLOSE_FEE and m else None


def _wrong_days(act: dict, text: str, ctx: dict) -> str | None:
    """DISCLOSE_FEE: the delivery time as 9 business days."""
    if act.get("act") != N.state.DISCLOSE_FEE:
        return None
    new = re.sub(r"\b\d+(?:-\d+)? business days", "9 business days", text)
    return new if new != text else None


def _wrong_limit(act: dict, text: str, ctx: dict) -> str | None:
    """EXPLAIN_LIMIT: the transaction limit one higher."""
    m = re.search(r"up to (\d+)", text)
    return text[:m.start()] + f"up to {int(m.group(1)) + 1}" + text[m.end():] if act.get("act") == N.state.EXPLAIN_LIMIT and m else None


def _claim_done_early(act: dict, text: str, ctx: dict) -> str | None:
    """CONFIRM_ACTION, before any card action succeeded: claim the card was already replaced (R17)."""
    if act.get("act") != N.state.CONFIRM_ACTION or ctx["submitted"]:
        return None
    return "Your card has been replaced. " + text


def _assert_fraud(act: dict, text: str, ctx: dict) -> str | None:
    """CONFIRM_RECORDED of an unrecognized charge: add "This means fraud." (R27 says it is not proof)."""
    return text + " This means fraud." if act.get("act") == N.state.CONFIRM_RECORDED and act.get("disposition") == N.value.UNRECOGNIZED else None


def _drop_summary_item(act: dict, text: str, ctx: dict) -> str | None:
    """SUMMARIZE with two or more records: leave one recorded answer out (pdf.2F)."""
    if act.get("act") != N.state.SUMMARIZE or len(act.get("recorded") or {}) < 2:
        return None
    m = re.search(r" and the [A-Za-z' ]+ charge as [a-z ]+(?=[.,])", text) or re.search(r", the [A-Za-z' ]+ charge as [a-z ]+", text)
    return text[:m.start()] + text[m.end():] if m else None


CONTROLS: dict[str, Callable[[dict, str, dict], str | None]] = {
    "swap_merchant": _swap_merchant, "change_amount": _change_amount, "shift_date": _shift_date,
    "drop_date": _drop_date, "flip_disposition": _flip_disposition, "wrong_fee": _wrong_fee,
    "wrong_days": _wrong_days, "wrong_limit": _wrong_limit, "claim_done_early": _claim_done_early,
    "assert_fraud": _assert_fraud, "drop_summary_item": _drop_summary_item,
}


# ------------------------------------------------------------------ controls for a pack compiled from a compact graph
def _pack():
    from ..s01_packs.loading import current
    dom = current()
    return dom if dom.graph is not None else None


def _wording(dom, v: str) -> list[str]:
    say = ((dom.get("lexicon.values") or {}).get(v) or {}).get("say") or {}
    return [w for w in (say.get("canonical"), *dom.graph.aliases(v)) if w]


def _value_fields(dom, act: dict) -> list[tuple[str, str, str]]:
    """(field, value, value set) for every field of the act whose value is a member of a value set."""
    g = dom.graph
    return [(k, v, g.member_set(v)) for k, v in act.items()
            if k not in ("act", "text") and isinstance(v, str) and g.member_set(v)]


def _p_swap_value(act, text, ctx):
    """A value the act says, worded as another member of its value set."""
    dom = _pack()
    for _, v, vs in _value_fields(dom, act):
        for w in _wording(dom, v):
            if w.lower() in text.lower():
                other = next((m for m in dom.graph.members(vs) if m != v and _wording(dom, m)), None)
                if other:
                    i = text.lower().index(w.lower())
                    return text[:i] + _wording(dom, other)[0] + text[i + len(w):]
    return None


def _p_wrong_money(act, text, ctx):
    """A money figure ten dollars off."""
    m = re.search(r"\$(\d+)", text)
    return text[:m.start()] + f"${int(m.group(1)) + 10}" + text[m.end():] if m else None


def _p_wrong_limit(act, text, ctx):
    """A limit one off (an act that says a limit)."""
    dom = _pack()
    units = [str(lv.get("unit") or "").rstrip("s") for lv in dom.graph.limits.values() if lv.get("unit")]
    for u in units:
        m = re.search(rf"\b(\d+)(\s+(?:\w+\s+)?{re.escape(u)}s?)\b", text)
        if m:
            return text[:m.start()] + str(int(m.group(1)) + 1) + m.group(2) + text[m.end():]
    return None


def _p_claim_done_early(act, text, ctx):
    """A write said to be done before any write succeeded."""
    if ctx["submitted"]:
        return None
    dom = _pack()
    g = dom.graph
    for _, v, vs in _value_fields(dom, act):
        if any(vs == avs for t, sp in g.tools.items() if sp.get("role") == "write" for avs in (sp.get("args") or {}).values()):
            w = _wording(dom, v)
            if w:
                return text + f" {w[0][:1].upper() + w[0][1:]} has been submitted."
    return None


def _p_drop_required(act, text, ctx):
    """A required value left out of the words."""
    dom = _pack()
    for _, v, _vs in _value_fields(dom, act):
        for w in _wording(dom, v):
            i = text.lower().find(w.lower())
            if i >= 0:
                return " ".join((text[:i] + text[i + len(w):]).split())
    return None


# ------------------------------------------------------------------ policy controls (any pack: built from its graph)
_CORPUS: dict[str, Any] = {}


def _corpus():
    """The active pack's policy corpus (s07_checks/speech_policy.py), built once per pack."""
    from ..s01_packs.loading import current
    dom = current()
    if dom.name not in _CORPUS:
        from .speech_reading import SpeechContract
        kg = dom.kg
        _CORPUS[dom.name] = PolicyCorpus.from_contract(SpeechContract.from_graph(kg), kg, getattr(kg, "graph", None))
    return _CORPUS[dom.name]


def _speaks(act: dict, text: str) -> bool:
    """An act with words to add to (the first one in the conversation the control reaches)."""
    return bool(text.strip())


def _invented_policy(act, text, ctx):
    """A policy the graph does not have, said as policy."""
    return text + " Our policy is to always refund the full amount on request." if _speaks(act, text) else None


def _wrong_policy_figure(act, text, ctx):
    """A customer-facing limit restated in other words with the wrong number."""
    if not _speaks(act, text):
        return None
    f = next((f for f in _corpus().figures if f.attr == "limit" and not f.internal and f.unit), None)
    if f is None:
        return None
    return text + f" Just so you know, the limit is {int(float(f.value)) * 3 + 7} {f.unit.replace('_', ' ')}."


def _unauthorised_promise(act, text, ctx):
    """A promise of an action the graph has no tool for."""
    if not _speaks(act, text):
        return None
    verb = next((v for v in ("waive", "refund", "credit", "reimburse") if stem(v) not in _corpus().action_stems), None)
    return text + f" I'll also {verb} any fees on your account this year." if verb else None


def _internal_disclosure(act, text, ctx):
    """An internal figure (marked internal in the graph, or on the project's never_volunteer list) told to the customer."""
    if not _speaks(act, text):
        return None
    never = set(settings()["never_volunteer"])
    f = next((f for f in _corpus().figures if f.attr == "limit" and (f.internal or f.subject in never) and f.unit), None)
    if f is None:
        return None
    return text + f" I can only go up to {int(float(f.value))} {f.unit.replace('_', ' ')} with you."


POLICY_CONTROLS: dict[str, Callable[[dict, str, dict], str | None]] = {
    "invented_policy": _invented_policy, "wrong_policy_figure": _wrong_policy_figure,
    "unauthorised_promise": _unauthorised_promise, "internal_disclosure": _internal_disclosure,
}
CONTROLS.update(POLICY_CONTROLS)

PACK_CONTROLS: dict[str, Callable[[dict, str, dict], str | None]] = {
    **POLICY_CONTROLS,
    "swap_value": _p_swap_value, "wrong_money": _p_wrong_money, "wrong_limit": _p_wrong_limit,
    "claim_done_early": _p_claim_done_early, "drop_required_value": _p_drop_required,
}


def controls() -> dict[str, Callable]:
    """The controls for the active pack."""
    return PACK_CONTROLS if _pack() is not None else CONTROLS


# ------------------------------------------------------------------ a bot whose words are edited once
@dataclass
class _EditedBot:
    """Wraps a bot: the first act the control applies to gets its words edited (labels untouched)."""
    inner: Any
    tools: Any
    control: str
    turn: int = -1                      # the bot turn whose words were edited (-1: never applied)
    edit: dict = field(default_factory=dict)
    _n: int = 0

    def start(self):
        """The opening turn (edited if the control applies to it)."""
        return self._maybe(self.inner.start())

    def respond(self, text):
        """The bot's reply to ``text`` (edited if the control applies and has not been applied yet)."""
        return self._maybe(self.inner.respond(text))

    def _maybe(self, turn):
        """Edit the first act the control applies to, once per conversation; remember which turn it was."""
        from ..s06_bots.reference import BotTurn
        n, self._n = self._n, self._n + 1
        if self.turn >= 0:
            return turn
        log = getattr(self.tools, "log", None) or getattr(getattr(self.tools, "env", None), "log", []) or []
        ctx = {"merchants": {t["transaction_id"]: t["merchant"] for c in log if roles.ITEMS and c["tool"] == roles.ITEMS
                             for t in (c.get("result") or {}).get("transactions", [])},
               "submitted": any(c["tool"] in roles.WRITE_TOOLS and c.get("ok") for c in log) if _pack() is not None
               else any(c["tool"] == roles.ACTION and c.get("ok") for c in log)}
        acts = [dict(a) for a in turn.acts]
        for a in acts:
            new = controls()[self.control](a, a.get("text", ""), ctx)
            if new is not None and new != a.get("text"):
                self.edit = {"act": a.get("act"), "before": a.get("text"), "after": new}
                a["text"] = new
                self.turn = n
                return BotTurn(" ".join(x.get("text", x["act"]) for x in acts), acts, turn.ended)
        return turn


def run_controls(make_runner: Callable[[Callable], Any], bot_factory: Callable, scenarios: list,
                 controls_: list[str] | None = None, max_examples: int = 3) -> dict[str, Any]:
    """Apply every control to every scenario where it can apply; count the edited turns the speech check flags."""
    out: dict[str, Any] = {"controls": {}}
    for name in controls_ or list(controls()):
        applied = detected = 0
        missed = []
        holder: dict[str, _EditedBot] = {}

        def factory(tools, _name=name):                # the runner makes a fresh bot per conversation
            b = _EditedBot(bot_factory(tools), tools, _name)
            holder["bot"] = b
            return b
        runner = make_runner(factory)
        for scn in scenarios:
            res = runner.run(scn)
            b = holder.get("bot")
            if b is None or b.turn < 0:
                continue
            applied += 1
            turns = (res.extras.get("speech") or {}).get("turns") or []
            rep = turns[b.turn] if b.turn < len(turns) else None
            if rep is not None and not rep["ok"]:
                detected += 1
            elif len(missed) < max_examples:
                missed.append({"sid": scn.sid, **b.edit, "report": rep})
        out["controls"][name] = {"applied": applied, "detected": detected,
                                 "detection_rate": detected / applied if applied else None, "missed": missed}
    rates = [v["detection_rate"] for v in out["controls"].values() if v["detection_rate"] is not None]
    out["overall"] = {"applied": sum(v["applied"] for v in out["controls"].values()),
                      "detected": sum(v["detected"] for v in out["controls"].values()),
                      "min_rate": min(rates) if rates else None}
    return out
