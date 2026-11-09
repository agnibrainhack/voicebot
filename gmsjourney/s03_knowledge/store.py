"""The symbolic knowledge store: the policy's facts, figures, rules and workflow, in memory.

Every step that needs a policy fact reads it here: an exact figure by key (``lookup_enm('limit',
'max_auth_attempts')``, never parsed from text), the rules with their stances, the workflow edges, the value sets
and their aliases. The trained (geometric) store of ``s03_knowledge/build.py`` extends this class; the console and the
playground use this symbolic one alone, which compiles in about a second and needs no licence.

Parts of this file:
  1. Ingest: the policy markdown -> candidate facts and exact figures, each with its source span.
  2. ``GMSStore``: the compiled store (Exact Numeric Memory with integrity hashes, typed triples, rules,
     workflow, value sets, synonyms, knowledge-base sentences).
  3. ``export_canonicals``: the canonical vocabulary as the trained store holds it (gms_canonicals.json).
  4. The GMS view: the policy rendered as the tables knowlytix's ingest reads (``render_gms_markdown``).
  5. ``symbolic_store`` / ``pack_store`` / ``pack_oracle``: the active pack's store and oracle, cached per pack.
"""
from __future__ import annotations

import hashlib
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


# =====================================================================================================================
# PART 1 - Ingest: policy markdown -> candidate facts with provenance (book Ch. 4.1).
#
# Two extraction paths run over the same document, mirroring the book's
# "hybrid" ingest:
#
# * a deterministic table/rule path that recovers exact figures byte-for-byte
#   (the authoritative backbone), and
# * a naive prose path that pairs each number in a knowledge-base sentence with
#   the nearest entity mention. The prose path is intentionally simple, so the
#   self-correction loop (geode.py) has real work to do: it must corroborate,
#   merge, or reject every prose candidate against the table backbone.
#
# Nothing here decides truth. Ingest only proposes; GEODE diagnoses and repairs.
# =====================================================================================================================

# --------------------------------------------------------------------------
# Candidate records
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Source:
    """Where a fact came from: section, row/line id and the verbatim span."""
    section: str
    ref: str
    span: str
    path: str = "table"          # table | rule | workflow | prose


@dataclass
class CandidateTriple:
    """A fact read from the policy document (head, relation, tail), with where it was read and any attributes;
    not yet admitted into the store.
    """
    head: str
    relation: str
    tail: str
    source: Source
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.head, self.relation, self.tail)


@dataclass
class CandidateENM:
    """A candidate for the exact numeric register (category, key) -> value."""
    category: str
    key: str
    value: str           # canonical decimal string
    unit: str
    source: Source


@dataclass
class Rule:
    """One policy rule: its id, stance (must / must_not / may), what it says as subject-predicate-object, its text
    and source.
    """
    rule_id: str
    stance: str          # must | must_not | may
    subject: str
    predicate: str
    obj: str
    text: str
    source: Source


@dataclass
class WorkflowEdge:
    """One step of the policy's workflow: from one state to another under a condition."""
    wid: str
    src: str
    dst: str
    condition: str
    source: Source


@dataclass
class IngestResult:
    """Everything read from the policy document: candidate facts and exact values, rules, workflow, value sets,
    aliases, entity types, knowledge-base sentences, required language and the raw sections.
    """
    title: str
    triples: list[CandidateTriple]
    enm: list[CandidateENM]
    rules: list[Rule]
    workflow: list[WorkflowEdge]
    value_sets: dict[str, list[str]]
    aliases: dict[str, list[str]]
    entity_types: dict[str, str]
    kb_sentences: list[tuple[str, str]]
    required_language: dict[str, str]
    sections: dict[str, str]


# --------------------------------------------------------------------------
# Markdown helpers
# --------------------------------------------------------------------------
_SECTION_RE = re.compile(r"^##\s+(\d+)\.\s+(.*)$")


def _split_sections(md: str) -> tuple[str, dict[str, str]]:
    title = ""
    sections: dict[str, list[str]] = {}
    current = None
    for line in md.splitlines():
        if line.startswith("# ") and not title:
            title = line[2:].strip()
            continue
        m = _SECTION_RE.match(line)
        if m:
            current = m.group(2).strip().lower()
            sections[current] = []
            continue
        if current is not None:
            sections[current].append(line)
    return title, {k: "\n".join(v).strip() for k, v in sections.items()}


def _parse_table(block: str) -> list[dict[str, str]]:
    rows = [ln.strip() for ln in block.splitlines() if ln.strip().startswith("|")]
    if len(rows) < 2:
        return []
    header = [c.strip() for c in rows[0].strip("|").split("|")]
    out = []
    for ln in rows[2:]:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if len(cells) != len(header):
            continue
        out.append(dict(zip(header, cells)))
    return out


def canonical_number(raw: str) -> str:
    """Canonical decimal string used as the ENM value ("15", "0", "2.5")."""
    s = raw.replace(",", "").replace("$", "").strip()
    if re.fullmatch(r"-?\d+", s):
        return str(int(s))
    if re.fullmatch(r"-?\d+\.\d+", s):
        return s.rstrip("0").rstrip(".") if "." in s else s
    raise ValueError(f"not a number: {raw!r}")


# --------------------------------------------------------------------------
# Ingest
# --------------------------------------------------------------------------
POLICY_ENTITY = "charge_review_policy"


def ingest_markdown(path: str | Path) -> IngestResult:
    """Read a policy document (markdown) into candidate facts: limits and fees become exact values and triples,
    options, rules, workflow and the knowledge-base sentences are read section by section.
    """
    md = Path(path).read_text()
    title, sec = _split_sections(md)

    triples: list[CandidateTriple] = []
    enm: list[CandidateENM] = []
    etypes: dict[str, str] = {POLICY_ENTITY: "policy"}

    def add(h: str, r: str, t: str, src: Source, **attrs: Any) -> None:
        triples.append(CandidateTriple(h, r, t, src, attrs))

    # ---- limits -> ENM + triples ------------------------------------------
    for row in _parse_table(sec.get("limits", "")):
        src = Source("limits", row["key"], row["text"])
        val = canonical_number(row["value"])
        enm.append(CandidateENM("limit", row["key"], val, row["unit"], src))
        etypes[row["key"]] = "limit"
        add(POLICY_ENTITY, f"has_{row['key']}", val, src, origin=row["source"])

    # ---- delivery options -> ENM + triples --------------------------------
    for row in _parse_table(sec.get("replacement delivery options", "")):
        opt = row["option"]
        etypes[opt] = "delivery_option"
        span = " | ".join(f"{k}={v}" for k, v in row.items())
        src = Source("replacement delivery options", opt, span)
        add(opt, "is_a", "delivery_option", src)
        add("replace_card", "delivered_by", opt, src)
        for col, rel, cat, unit in (
            ("fee_usd", "has_fee_usd", "delivery_fee_usd", "usd"),
            ("business_days_min", "has_business_days_min", "delivery_days_min", "business_days"),
            ("business_days_max", "has_business_days_max", "delivery_days_max", "business_days"),
        ):
            val = canonical_number(row[col])
            enm.append(CandidateENM(cat, opt, val, unit, src))
            add(opt, rel, val, src)
        add(opt, "requires_fee_confirmation", row["requires_fee_confirmation"], src)

    # ---- tools ---------------------------------------------------------------
    for row in _parse_table(sec.get("tools", "")):
        tool = row["tool"]
        etypes[tool] = "tool"
        src = Source("tools", tool, row["when_used"])
        add(tool, "is_a", "tool", src)
        add(tool, "has_stage", row["stage"], src)
        add(tool, "success_value", row["success_value"], src)
        for fv in row["failure_values"].split(";"):
            add(tool, "failure_value", fv.strip(), src)
        add(tool, "on_failure", row["on_failure"], src)
        etypes.setdefault(row["stage"], "stage")

    # ---- workflow ----------------------------------------------------------
    workflow: list[WorkflowEdge] = []
    wf_re = re.compile(r"^-\s*(W\d+):\s*(\S+)\s*->\s*(\S+)\s*:\s*(.*)$")
    for line in sec.get("workflow", "").splitlines():
        m = wf_re.match(line.strip())
        if not m:
            continue
        wid, a, b, cond = m.groups()
        src = Source("workflow", wid, line.strip(), "workflow")
        workflow.append(WorkflowEdge(wid, a, b, cond.strip(), src))
        add(a, "enables", b, src, condition=cond.strip(), wid=wid)
        for node in (a, b):
            etypes.setdefault(node, "tool" if node.endswith(("_customer", "_transactions",
                                                              "_response", "_options",
                                                              "_action", "_human")) else "state")

    # ---- rules -------------------------------------------------------------
    rules: list[Rule] = []
    for row in _parse_table(sec.get("rules", "")):
        src = Source("rules", row["rule_id"], row["text"], "rule")
        r = Rule(row["rule_id"], row["stance"], row["subject"], row["predicate"],
                 row["object"], row["text"], src)
        rules.append(r)
        etypes[r.rule_id] = "rule"
        add(r.rule_id, "has_stance", r.stance, src)
        add(r.rule_id, "constrains", r.subject, src)
        add(r.subject, r.predicate, r.obj, src, stance=r.stance, rule=r.rule_id)

    # ---- value sets + aliases ----------------------------------------------
    value_sets: dict[str, list[str]] = {}
    aliases: dict[str, list[str]] = {}
    for row in _parse_table(sec.get("dispositions and other value sets", "")):
        vs, val = row["value_set"], row["value"]
        value_sets.setdefault(vs, []).append(val)
        etypes.setdefault(val, vs)
        src = Source("value sets", f"{vs}:{val}", row["aliases"])
        add(val, "member_of", vs, src)
        als = [a.strip() for a in row["aliases"].split(";") if a.strip()]
        aliases[val] = als
        for a in als:
            add(val, "has_alias", a, src)

    # ---- domain model ------------------------------------------------------
    for row in _parse_table(sec.get("domain model", "")):
        src = Source("domain model", f"{row['head']}.{row['relation']}", row["source"])
        add(row["head"], row["relation"], row["tail"], src)
        for n in (row["head"], row["tail"]):
            etypes.setdefault(n, "concept")

    # ---- escalation triggers -----------------------------------------------
    for row in _parse_table(sec.get("escalation triggers", "")):
        src = Source("escalation triggers", row["trigger"], row["condition"])
        etypes[row["trigger"]] = "escalation_trigger"
        add(row["trigger"], "is_a", "escalation_trigger", src)
        add(row["trigger"], "justified_by", row["rule"], src)

    # ---- knowledge base prose ----------------------------------------------
    kb = []
    for i, line in enumerate(l.strip() for l in sec.get("knowledge base", "").splitlines()):
        if line:
            kb.append((f"KB{len(kb) + 1}", line))

    required = {}
    m = re.search(r'Required opening:\s*"([^"]+)"', sec.get("required language", ""))
    if m:
        required["opening"] = m.group(1)

    result = IngestResult(
        title=title, triples=triples, enm=enm, rules=rules, workflow=workflow,
        value_sets=value_sets, aliases=aliases, entity_types=etypes,
        kb_sentences=kb, required_language=required, sections=sec,
    )
    # Naive prose candidates are appended last; GEODE must reconcile them.
    result.triples.extend(extract_prose_candidates(result))
    return result


# --------------------------------------------------------------------------
# Naive prose extraction (the noisy half of the hybrid ingest)
# --------------------------------------------------------------------------
_PROSE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\$(\d+(?:\.\d+)?)\s+fee"), "has_fee_usd"),
    (re.compile(r"fee is \$(\d+(?:\.\d+)?)"), "has_fee_usd"),
    (re.compile(r"(\d+)\s*[-–]\s*(\d+)\s+business days"), "has_business_days_range"),
    (re.compile(r"(?<![-–\d])(\d+)\s+business days"), "has_business_days_point"),
    (re.compile(r"up to (\d+) (?:recent )?transactions"), "has_max_transactions_per_session"),
    (re.compile(r"at most (\d+) transactions"), "has_max_transactions_per_session"),
    (re.compile(r"previous (\d+) calendar days"), "has_lookback_days"),
    (re.compile(r"[Aa]t most (\d+) incorrect authentication attempts"), "has_max_auth_attempts"),
]


def _nearest_option(sentence: str, pos: int, aliases: dict[str, list[str]]) -> str | None:
    """The delivery option mentioned closest *before* `pos` (naive binding)."""
    best, best_pos = None, -1
    low = sentence.lower()
    for opt in ("standard_delivery", "expedited_delivery"):
        for a in [opt.split("_")[0]] + aliases.get(opt, []):
            for m in re.finditer(re.escape(a.lower()), low):
                if m.start() < pos and m.start() > best_pos:
                    best, best_pos = opt, m.start()
    return best


def extract_prose_candidates(ing: IngestResult) -> list[CandidateTriple]:
    """Second reading of the same figures from free text (knowledge-base sentences, rule and limit texts), each bound
    to the nearest option it mentions; used to corroborate or contradict the structured reading.
    """
    texts: list[tuple[str, str, str]] = [("knowledge base", sid, s) for sid, s in ing.kb_sentences]
    # rule and limit texts are secondary evidence for the same figures
    texts += [("rules", r.rule_id, r.text) for r in ing.rules]
    out: list[CandidateTriple] = []
    for section, ref, sent in texts:
        for pat, rel in _PROSE_PATTERNS:
            for m in pat.finditer(sent):
                src = Source(section, ref, sent, "prose")
                if rel in ("has_fee_usd",):
                    subj = _nearest_option(sent, m.start(), ing.aliases) or "unbound"
                    out.append(CandidateTriple(subj, rel, canonical_number(m.group(1)), src))
                elif rel == "has_business_days_range":
                    subj = _nearest_option(sent, m.start(), ing.aliases) or "unbound"
                    out.append(CandidateTriple(subj, "has_business_days_min",
                                               canonical_number(m.group(1)), src))
                    out.append(CandidateTriple(subj, "has_business_days_max",
                                               canonical_number(m.group(2)), src))
                elif rel == "has_business_days_point":
                    subj = _nearest_option(sent, m.start(), ing.aliases) or "unbound"
                    v = canonical_number(m.group(1))
                    out.append(CandidateTriple(subj, "has_business_days_min", v, src))
                    out.append(CandidateTriple(subj, "has_business_days_max", v, src))
                else:
                    out.append(CandidateTriple(POLICY_ENTITY, rel,
                                               canonical_number(m.group(1)), src))
    return out


# =====================================================================================================================
# PART 2 - The compiled policy store: the symbolic layer every step reads (book Ch. 2).
#
# ``GMSStore`` holds what the compiler (s03_knowledge/build.py: build_store) reads from the policy:
#
# 1. the asserted knowledge graph (typed triples with provenance),
# 2. the Exact Numeric Memory register (ENM): every authoritative number keyed by
#    (category, key) and protected by a SHA-256 integrity hash, so a figure is
#    *read*, never parsed out of prose,
# 3. the rules, the tool workflow, value sets, customer synonyms, entity types,
#    knowledge-base sentences and the required language.
#
# It carries no geometry. The trained GMS is knowlytix's (s03_knowledge/build.py:
# ``KXPolicyStore`` subclasses this class and answers every geometric read -
# score_triple, cap radii, tension, holonomy, the tuned encoders - from the
# knowlytix store). Helpers shared by the calibration code live here:
# ``negatives_for`` (counterfactual tails) and ``_operating_point`` (a threshold under a
# false-accept ceiling).
# =====================================================================================================================


FALSE_ACCEPT_CEILING = 0.05
_NUMERIC = re.compile(r"^-?\d+(?:\.\d+)?$")


class IntegrityError(RuntimeError):
    """Raised when an ENM value no longer matches its integrity hash."""


@dataclass
class ENMEntry:
    """One exact value (category, key, value, unit) with its SHA-256 digest and the sources it was read from: the
    store's tamper-evident record of a figure.
    """
    category: str
    key: str
    value: str
    unit: str
    sha256: str
    sources: list[dict] = field(default_factory=list)

    @staticmethod
    def digest(category: str, key: str, value: str, unit: str) -> str:
        return hashlib.sha256(f"{category}|{key}|{value}|{unit}".encode()).hexdigest()

    def verify(self) -> bool:
        return self.sha256 == self.digest(self.category, self.key, self.value, self.unit)

    def as_number(self) -> int | float:
        return int(self.value) if re.fullmatch(r"-?\d+", self.value) else float(self.value)


class GMSStore:
    """The symbolic policy store: admitted triples with provenance, exact values, rules, workflow, value sets and
    aliases, with the lookups the cases and checks use (value of a limit, fee of an option, rules about a thing ...).
    """
    # ------------------------------------------------------------------ init
    def __init__(self, *, title: str, triples: Iterable[tuple[str, str, str]],
                 provenance: dict[tuple[str, str, str], list[dict]],
                 enm: dict[tuple[str, str], ENMEntry], rules: dict[str, Rule],
                 workflow: list[WorkflowEdge], value_sets: dict[str, list[str]],
                 aliases: dict[str, list[str]], entity_types: dict[str, str],
                 kb_sentences: list[tuple[str, str]], required_language: dict[str, str]):
        self.title = title
        self.triples: set[tuple[str, str, str]] = set(triples)
        self.provenance = provenance
        self.enm = enm
        self.rules = rules
        self.workflow = workflow
        self.value_sets = value_sets
        self.aliases = aliases
        self.entity_types = entity_types
        self.kb_sentences = kb_sentences
        self.required_language = required_language
        self.kg = None                                     # the policy graph (s02_graph/triples.py)
        self.calibration: dict[str, Any] = {}
        self._reindex()

    def _reindex(self) -> None:
        self.by_h: dict[str, set] = {}
        self.by_r: dict[str, set] = {}
        self.by_t: dict[str, set] = {}
        for tr in self.triples:
            h, r, t = tr
            self.by_h.setdefault(h, set()).add(tr)
            self.by_r.setdefault(r, set()).add(tr)
            self.by_t.setdefault(t, set()).add(tr)

    # ------------------------------------------------------------- exact reads
    def lookup_enm(self, category: str, key: str) -> int | float:
        """Byte-exact numeric read from the ENM register (integrity-checked)."""
        e = self.enm[(category, key)]
        if not e.verify():
            raise IntegrityError(f"ENM integrity failure for {category}/{key}")
        return e.as_number()

    def query_triples(self, head: str | None = None, relation: str | None = None,
                      tail: str | None = None) -> list[tuple[str, str, str]]:
        pool = self.triples
        if head is not None:
            pool = self.by_h.get(head, set())
        if relation is not None:
            pool = pool & self.by_r.get(relation, set()) if head is not None else self.by_r.get(relation, set())
        out = [tr for tr in pool
               if (head is None or tr[0] == head) and (relation is None or tr[1] == relation)
               and (tail is None or tr[2] == tail)]
        return sorted(out)

    def tails(self, head: str, relation: str) -> list[str]:
        return [t for _, _, t in self.query_triples(head, relation)]

    def relation_range(self, relation: str) -> list[str]:
        return sorted({t for _, _, t in self.by_r.get(relation, set())})

    # ----------------------------------------------------------- calibration
    @staticmethod
    def _operating_point(pos: list[float], neg: list[float], ceiling: float = FALSE_ACCEPT_CEILING,
                         accept_low: bool = True) -> dict[str, float]:
        """Threshold maximising accuracy subject to a false-accept ceiling.

        accept_low=True means a score <= tau is accepted (distances)."""
        if not pos:
            return {"tau": float("nan"), "tar": 0.0, "far": 0.0, "n_pos": 0, "n_neg": len(neg)}
        s = (lambda x: x) if accept_low else (lambda x: -x)
        P, N = [s(x) for x in pos], [s(x) for x in neg]
        cands = sorted(set(P + N + [max(P) + 0.05]))
        best = None
        for c in cands:
            tar = sum(p <= c for p in P) / len(P)
            far = (sum(n <= c for n in N) / len(N)) if N else 0.0
            if far > ceiling:
                continue
            acc = (tar * len(P) + (1 - far) * len(N)) / (len(P) + len(N))
            if best is None or acc > best[0] + 1e-12 or (abs(acc - best[0]) < 1e-12 and c < best[1]):
                best = (acc, c, tar, far)
        if best is None:                       # no threshold meets the ceiling
            c = min(P)
            best = (0.0, c, sum(p <= c for p in P) / len(P),
                    (sum(n <= c for n in N) / len(N)) if N else 0.0)
        # place the threshold midway into the gap above the chosen point when possible
        acc, c, tar, far = best
        above = [n for n in N if n > c]
        if above and tar == 1.0:
            c = (c + min(above)) / 2 if min(above) > max(P) else c
        tau = c if accept_low else -c
        return {"tau": float(tau), "tar": float(tar), "far": float(far),
                "n_pos": len(P), "n_neg": len(N)}

    def negatives_for(self, head: str, relation: str, tail: str, rng: random.Random,
                      n: int = 6) -> list[str]:
        """Counterfactual tails: same-range tails the head does not assert, plus numeric perturbations."""
        asserted = set(self.tails(head, relation))
        pool = [t for t in self.relation_range(relation) if t not in asserted]
        if _NUMERIC.match(tail):
            x = float(tail)
            pert = {x + 1, x - 1, x * 2, x + 10, x + 5, round(x / 2)}
            pool += [f"{p:g}" for p in pert if p >= 0 and f"{p:g}" not in asserted]
        else:
            same_type = [e for e, ty in self.entity_types.items()
                         if ty == self.entity_types.get(tail) and e not in asserted]
            pool += same_type
        pool = sorted(set(pool) - {tail})
        rng.shuffle(pool)
        return pool[:n]

    # ---------------------------------------------------------------- summary
    def summary(self) -> dict[str, Any]:
        return {"entities": len(self.entity_types), "triples": len(self.triples),
                "relations": len(self.by_r), "enm_entries": len(self.enm),
                "rules": len(self.rules), "workflow_edges": len(self.workflow)}


# =====================================================================================================================
# PART 3 - The canonical vocabulary as the *trained* GMS holds it (gms_canonicals.json).
#
# For every canonical individual the export records what the trained store says about it,
# next to what the policy graph (policy/charge_review/charge_review_kg.json) states:
#
# * value sets  - each canonical value, its aliases, and how the trained GMS binds every
#                 alias (``fuzzy_match_entity(alias, mode="frame")``: the GEODE-tuned v encoder)
#                 - agreement with the
#                 graph is reported per alias; the value's membership score and the tension to
#                 its disjoint siblings (u-space);
# * numbers     - every exact figure (ENM) with its unit, integrity digest and the value the
#                 trained store returns;
# * states      - each dialog state with the meaning cards it accepts, how it reads a bare
#                 yes/no, its rules (id, stance, text), limits (with values), next tools,
#                 transitions and escalation triggers;
# * acts        - the slots of each meaning card and the value set each slot draws from;
# * facts       - every triple the GMS is trained on with its geodesic score, the relation's
#                 calibrated cut, and whether the fact sits inside it;
# * relations   - per relation: the cut and the share of its facts inside it.
#
# Nothing here is hand-written: it is read off the trained store and the graph.
# =====================================================================================================================

def _safe(fn, default=None):
    try:
        return fn()
    except Exception:                                     # noqa: BLE001 - a relation may be uncalibrated
        return default


def export_canonicals(store, kg, backend_info: dict | None = None) -> dict[str, Any]:
    """The store's canonical answers and scores as a JSON-ready dict (each fact's score and cut-off, each figure,
    each rule), tied to the graph's SHA-256.
    """
    meta = {"schema": "gmsjourney.gms_canonicals/1", "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "backend": "knowlytix", "policy_graph_sha256": kg.sha256,
            "policy_document": kg.data["source"]["document"],
            "encoders": getattr(store, "encoders", None),
            "versions": (backend_info or {}).get("versions")}

    def score(h, r, t):
        return _safe(lambda: float(store.score_triple(h, r, str(t))))

    def cut(r):
        return _safe(lambda: float(store.tau(r)))

    # ---------------------------------------------------------------- value sets
    value_sets: dict[str, Any] = {}
    for vs, vals in store.value_sets.items():
        members = []
        for v in vals:
            aliases = []
            for a in store.aliases.get(v, []):
                bound, conf = _safe(lambda a=a: store.fuzzy_match_entity(a, mode="frame"), (None, None))
                aliases.append({"alias": a, "gms_binds_to": bound,
                                "confidence": None if conf is None else round(float(conf), 4),
                                "agrees": bound is not None and str(bound).lower() == v.lower()})
            s, c = score(v, "member_of", vs), cut("member_of")
            members.append({
                "canonical": v, "label": v.replace("_", " ").lower(), "aliases": aliases,
                "alias_agreement": (sum(a["agrees"] for a in aliases) / len(aliases)) if aliases else None,
                "member_of_score": s, "member_of_cut": c,
                "member_of_inside": (s is not None and c is not None and s <= c),
                "tension_to_siblings": {o: _safe(lambda o=o: round(float(store.tension_energy(v, o)), 4))
                                        for o in vals if o != v},
                "accepted_in_states": sorted({t["head"] for t in kg.where("accepts_act")
                                              if any(q.get("values_from") == vs
                                                     for q in kg.slot_schema(t["tail"]).values())}),
            })
        value_sets[vs] = {"members": members, "disjoint": True}
    from ..s02_graph.triples import EXTRA_VALUE_SETS
    for vs, vals in EXTRA_VALUE_SETS.items():
        value_sets[vs] = {"members": [{"canonical": v, "label": v.replace("_", " ")} for v in vals],
                          "disjoint": True, "note": "policy-question vocabulary (not trained geometry)"}

    # ---------------------------------------------------------------- numbers
    numbers = []
    for (cat, key), e in sorted(store.enm.items()):
        got = _safe(lambda: store.lookup_enm(cat, key))
        numbers.append({"category": cat, "key": key, "value": e.as_number(), "unit": e.unit,
                        "sha256": e.sha256, "trained_store_value": got,
                        "agree": got is not None and float(got) == float(e.as_number())})

    # ---------------------------------------------------------------- states + acts
    states = {}
    for sid in kg.states:
        y, n = kg.interprets(sid)
        rows = [t for t in kg.state_triples(sid) if t.get("gms")]
        checks = []
        for t in rows:
            s, c = score(t["head"], t["relation"], t["tail"]), cut(t["relation"])
            checks.append(s is not None and c is not None and s <= c)
        states[sid] = {
            "stage": kg.stage(sid), "prompts_customer": sid in kg.prompt_states,
            "accepts": kg.accepted_acts(sid), "reads_yes_as": y, "reads_no_as": n,
            "rules": [{"id": r, "stance": (kg.tails(r, "has_stance") or [None])[0], "text": kg.rule_text(r)}
                      for r in kg.rules_for(sid)],
            "limits": [{"key": k, "value": kg.limit(k)} for k in kg.limits_for(sid)],
            "tools": kg.tools_for(sid),
            "transitions": [{"to": t["dst"], "on": t.get("on"), "when": t.get("when"), "rules": t.get("rules")}
                            for t in kg.transitions(sid)],
            "triggers": kg.triggers_for(sid),
            "gms_facts_inside_cut": (sum(checks) / len(checks)) if checks else None,
        }
    acts = {}
    for t in kg.where("is_a", tail="dialog_act"):
        a = t["head"]
        acts[a] = {"slots": {k: {"values_from": q.get("values_from"), "required": q.get("required"),
                                 "values": store.value_sets.get(q.get("values_from"),
                                                                EXTRA_VALUE_SETS.get(q.get("values_from")))}
                             for k, q in kg.slot_schema(a).items()},
                   "accepted_in_states": [s for s in kg.prompt_states if kg.accepts(s, a)]}

    # ---------------------------------------------------------------- facts + relations
    facts, by_rel = [], {}
    for h, r, t in kg.hrt(gms_only=True):
        if r == "has_value":                       # limits are stored as (policy, has_<limit>, v)
            h, r = "charge_review_policy", f"has_{h}"
        s, c = score(h, r, t), cut(r)
        inside = s is not None and c is not None and s <= c
        facts.append({"head": h, "relation": r, "tail": t, "score": None if s is None else round(s, 4),
                      "cut": None if c is None else round(c, 4), "inside": inside})
        by_rel.setdefault(r, []).append(inside)
    relations = {r: {"cut": None if cut(r) is None else round(cut(r), 4), "facts": len(v),
                     "inside_rate": round(sum(v) / len(v), 4)} for r, v in sorted(by_rel.items())}

    entities = [{"id": n, "type": ty, "label": n.replace("_", " ").lower(), "aliases": store.aliases.get(n, [])}
                for n, ty in sorted(store.entity_types.items())]

    summary = {
        "entities": len(entities), "value_sets": len(value_sets), "numbers": len(numbers),
        "numbers_agree": all(x["agree"] for x in numbers), "states": len(states), "acts": len(acts),
        "facts": len(facts), "facts_inside_cut": round(sum(f["inside"] for f in facts) / max(1, len(facts)), 4),
        "alias_agreement": round(
            sum(a["agrees"] for vs in value_sets.values() for m in vs["members"] for a in m.get("aliases", []))
            / max(1, sum(len(m.get("aliases", [])) for vs in value_sets.values() for m in vs["members"])), 4),
    }
    return {"meta": meta, "summary": summary, "value_sets": value_sets, "numbers": numbers, "states": states,
            "acts": acts, "relations": relations, "facts": facts, "entities": entities}


# =====================================================================================================================
# PART 4 - The policy, rendered in the table shape knowlytix's GEODE ingest reads.
#
# ``knowlytix.benchmark.ingest.ingest_markdown`` turns every wide table into typed
# triples ``(row_entity, has_<column>, cell)`` and every numeric cell into an
# integrity-hashed ENM entry ``(type=<section>, id=<row_entity>/<column>)``. The
# source policy (policy/charge_review/charge_review_policy.md) mixes tables with a bulleted
# workflow, a head/relation/tail domain table and multi-alias cells, which that
# parser reifies or flattens. This module renders the *same compiled facts* as one
# table per relation, so GEODE proposes the graph the policy states:
#
# * ``## Limits``            -> (max_turns, has_value, 20.0) + ENM limits/max_turns/value
# * ``## Delivery options``  -> (expedited_delivery, has_fee_usd, 15.0) + ENM
# * ``## Workflow``          -> (authenticate_customer, has_enables, get_review_transactions)
# * ``## Aliases``           -> (recognized, has_alias, mine)  - the GEODE embed loop's supervision
# * ``## Opposites``         -> (recognized, has_opposite_of, unrecognized) - disjoint values and
#                              opposite stances, which knowlytix's tension calibration reads
# * one table per domain relation, the rules (reified), escalation triggers.
#
# Every row is generated from a compiled fact that keeps its source span, so the
# view is a deterministic function of the policy document: one source, one store.
# Boolean yes/no columns are written as words (``required``/``not_required``)
# because the parser treats yes/no columns as pass/fail flags.
# =====================================================================================================================

def _row(*cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _table(title: str, header: list[str], rows: list[list]) -> list[str]:
    if not rows:
        return []
    out = [f"## {title}", "", _row(*header), _row(*["---"] * len(header))]
    out += [_row(*r) for r in rows]
    return out + [""]


# relations the compiled graph carries that are rendered elsewhere (or are rule reifications)
_SKIP = {"has_fee_usd", "has_business_days_min", "has_business_days_max", "requires_fee_confirmation",
         "member_of", "has_alias", "enables", "has_stance", "constrains"}


def render_gms_markdown(store) -> str:
    """Render the compiled policy (a GMSStore built by s03_knowledge/build.py) as GEODE input."""
    lines = [f"# {store.title} (GMS view)", "",
             "Generated from the policy document; every row is a compiled fact with source provenance.", ""]
    lim = sorted((e.key, e.value, e.unit) for e in store.enm.values() if e.category == "limit")
    # numeric tables carry no other text column: the parser folds text cells into the ENM key
    lines += _table("Limits", ["limit", "value"], [[k, v] for k, v, _ in lim])
    lines += _table("Limit units", ["limit", "unit"], [[k, u] for k, _, u in lim])
    opts = sorted({k for (c, k) in store.enm if c == "delivery_fee_usd"})
    conf = {h: t for h, r, t in store.triples if r == "requires_fee_confirmation"}
    lines += _table("Delivery options", ["delivery_option", "fee_usd", "business_days_min", "business_days_max"],
                    [[o, store.enm[("delivery_fee_usd", o)].value, store.enm[("delivery_days_min", o)].value,
                      store.enm[("delivery_days_max", o)].value] for o in opts])
    lines += _table("Fee confirmation", ["delivery_option", "fee_confirmation"],
                    [[o, "required" if conf.get(o) == "yes" else "not_required"] for o in opts])
    lines += _table("Workflow", ["step", "enables"], [[w.src, w.dst] for w in store.workflow])
    vs_rows = [[v, vs] for vs, vals in sorted(store.value_sets.items()) for v in vals]
    lines += _table("Value sets", ["value", "member_of"], vs_rows)
    lines += _table("Aliases", ["entity", "alias"],
                    [[c, a] for c, als in sorted(store.aliases.items()) for a in als])
    opp = []
    for vs, vals in sorted(store.value_sets.items()):
        for i, a in enumerate(vals):
            for b in vals[i + 1:]:
                opp.append([a, b])
    opp.append(["must", "must_not"])
    lines += _table("Opposites", ["entity", "opposite_of"], opp)
    by_rel: dict[str, list[list[str]]] = defaultdict(list)
    for h, r, t in sorted(store.triples):
        if r in _SKIP or h.startswith("R") and h[1:].isdigit():
            continue
        if h == "charge_review_policy" and r.startswith("has_") and r[4:] in _LIMIT_KEYS:
            continue                                   # rendered in the Limits table
        by_rel[r].append([h, t])
    for r, rows in sorted(by_rel.items()):
        col = r[4:] if r.startswith("has_") else r
        lines += _table(f"Relation {col}", ["entity", col], rows)
    lines += _table("Rules", ["rule", "stance", "subject", "predicate", "object"],
                    [[r.rule_id, r.stance, r.subject, r.predicate, r.obj] for r in store.rules.values()])
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- vocabulary map
def kx_relation(rel: str) -> str:
    """Our compiled relation name -> the relation the knowlytix ingest produces for it."""
    special = {"enables": "has_enables", "requires_fee_confirmation": "has_fee_confirmation",
               "member_of": "has_member_of", "has_alias": "has_alias"}
    if rel in special:
        return special[rel]
    if rel.startswith("has_"):
        return rel
    return f"has_{rel}"


def kx_tail(tail: str) -> str:
    """A tail written the way the trained store expects it (yes/no -> required/not_required, numbers with one
    decimal, text lower-cased and single-spaced).
    """
    t = str(tail)
    if t in ("yes", "no"):
        return {"yes": "required", "no": "not_required"}[t]
    try:
        return f"{float(t):.1f}" if "." not in t or float(t) == int(float(t)) else str(float(t))
    except ValueError:
        return " ".join(t.split()).lower()


def kx_triple(h: str, r: str, t: str) -> tuple[str, str, str]:
    """Map a fact in our vocabulary onto the knowlytix store's vocabulary.

    Limits are stored as (charge_review_policy, has_<limit>, v) in the compiled
    graph and as (<limit>, has_value, v) in the GMS view."""
    from ..s01_packs.loading import current
    g = current().graph
    if g is not None:
        return _kx_triple_graph(g, h, r, t)
    if h == "charge_review_policy" and r.startswith("has_") and r[4:] in _LIMIT_KEYS:
        return r[4:], "has_value", kx_tail(t)
    return " ".join(h.split()).lower(), kx_relation(r), kx_tail(t)


_LIMIT_KEYS = {"max_transactions_per_session", "lookback_days", "max_auth_attempts",
               "max_clarification_retries", "max_unsure_retries", "max_turns"}


def _kx_triple_graph(g, h: str, r: str, t: str) -> tuple[str, str, str]:
    """kx_triple for a compact policy graph (s02_graph/compile.py render_gms_view)."""
    if h == g.root and r.startswith("has_") and r[4:] in g.limits:
        return r[4:], "has_value", kx_tail(t)
    from ..s02_graph.compile import yes_no_words
    vs = g.member_set(h)
    if vs is not None and str(t) in ("yes", "no"):
        for a in g.attributes(vs):
            from ..s02_graph.compact import attr_relation
            if attr_relation(a) == r:
                col, yes, no = yes_no_words(a)
                return " ".join(h.split()).lower(), f"has_{col}", yes if str(t) == "yes" else no
    return " ".join(h.split()).lower(), kx_relation(r), kx_tail(t)


# =====================================================================================================================
# PART 5 - The symbolic store of the active domain pack: compiled straight from its graph in about a second, no GMS
#
# training and no licence. Previews, the studio, factor relevance and the parity tests run on it; the campaign
# runs on the knowlytix GMS built from the same graph.
# =====================================================================================================================

_CACHE: dict = {}


def symbolic_store(policy_graph=None, kg=None) -> Any:
    """The store compiled from a policy graph (default: the active pack's), with ``store.kg`` the full graph and a
    symbolic stance check (the claimed stance equals the rule's, or not; the GMS reads tension instead)."""
    from ..s01_packs.loading import current
    from .build import build_store
    dom = current()
    path = policy_graph or dom.graph_path
    b = build_store(path)
    store = getattr(b, "store", b)
    if kg is not None:
        store.kg = kg
    elif str(path) == str(dom.graph_path):
        store.kg = dom.kg
    elif getattr(store, "kg", None) is None:
        from ..s02_graph.triples import PolicyKG
        store.kg = PolicyKG.load(path)

    def stance_check(subject, predicate, obj, polarity):
        rule = next((r for r in store.rules.values() if (r.subject, r.predicate, r.obj) == (subject, predicate, obj)), None)
        if rule is None:
            return {"verdict": "uncertain", "energy": None, "rule": None}
        claimed = "must" if polarity > 0 else "must_not"
        stance = "must" if rule.stance == "may" else rule.stance
        return {"verdict": "supported" if claimed == stance else "contradicted", "energy": None, "rule": rule.rule_id}

    if not hasattr(store, "stance_check"):
        store.stance_check = stance_check
    return store


def pack_store() -> Any:
    """The active pack's symbolic store, built once per pack."""
    from ..s01_packs.loading import current
    dom = current()
    key = (str(dom.dir), id(dom))
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = symbolic_store()
    return _CACHE[key]


def pack_oracle():
    """The graph oracle on the active pack's symbolic store (cached with it)."""
    store = pack_store()
    if getattr(store, "_flow_oracle", None) is None:
        from ..s02_graph.flow import GraphOracle
        store._flow_oracle = GraphOracle(store)
    return store._flow_oracle


def _reset() -> None:
    _CACHE.clear()


from ..s01_packs.loading import on_switch as _on_switch  # noqa: E402

_on_switch(_reset)
