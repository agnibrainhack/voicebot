"""Ingest: policy markdown -> candidate facts with provenance (book Ch. 4.1).

Two extraction paths run over the same document, mirroring the book's
"hybrid" ingest:

* a deterministic table/rule path that recovers exact figures byte-for-byte
  (the authoritative backbone), and
* a naive prose path that pairs each number in a knowledge-base sentence with
  the nearest entity mention. The prose path is intentionally simple, so the
  self-correction loop (geode.py) has real work to do: it must corroborate,
  merge, or reject every prose candidate against the table backbone.

Nothing here decides truth. Ingest only proposes; GEODE diagnoses and repairs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
    rule_id: str
    stance: str          # must | must_not | may
    subject: str
    predicate: str
    obj: str
    text: str
    source: Source


@dataclass
class WorkflowEdge:
    wid: str
    src: str
    dst: str
    condition: str
    source: Source


@dataclass
class IngestResult:
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
