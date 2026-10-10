"""Catalog loading, relevance resolution and validation (book Ch. 5-6, App. A).

Three layers, mirroring beyond-ship-and-pray's ``proofloop.suite``:

* ``catalogs/official/``   the book's domain-neutral base catalog (18 categories in
  5 families), factor catalog (40 factors, 10 groups, plus named factor groups)
  and profiles, vendored unmodified (Apache-2.0; see PROVENANCE.md).
* ``catalogs/custom_factors.yaml``  factors this application adds (Ch. 6.3.2),
  in the official schema.
* ``catalogs/application.yaml``  this bot's selection, level vocabulary with
  ``level_aliases`` back to catalog levels, exclusions with reasons, factors moved
  to the base, and how each base category reaches the bot.

``resolve()`` expands base tokens ("*", a family, a name) and factor tokens ("*", a
factor group, a name), drops factors the application excludes or moved to the base,
keeps only factors whose ``applies_to`` matches a selected base ("enrichment
enriches the base") and records every dropped factor with its reason.

``validate()`` enforces the official ``validate_catalogs.py`` contract (rules 1-5 and
profile references) plus the application contract: every catalog factor accounted
for, every application level traced to a catalog level (or declared custom), and
every catalog level of a selected factor realised, unrealised with a reason, or
moved to the base.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CATALOG_DIR = Path(__file__).resolve().parent / "catalogs"
OFFICIAL_DIR = CATALOG_DIR / "official"

# Vocabularies of the official validator (code/scripts/validate_catalogs.py).
FAMILIES = {"retrieval_recall", "comparison_ranking", "reasoning_composition",
            "consistency_integrity", "behavioral_operational"}
REQUIRES_VOCAB = {"enm", "triples", "path", "threshold", "ordered_sequence", "negatives", "provenance", "policy"}
ANSWER_TYPES = {"float", "int", "bool", "set_str", "str", "tuple", "mcq", "list", "decision"}
SURFACES = {"kb", "journey", "oracle"}
SNAKE = re.compile(r"^[a-z][a-z0-9_]*$")
_NON_FACTOR_SECTIONS = {"factor_groups"}

# sha256 of the vendored files (official/PROVENANCE.md); validate() reports drift.
OFFICIAL_SHA256 = {
    "base_catalog.yaml": "a52de1843b1cbc3dba8578a48a03d5e612cdc65e3a397da50f52c0a549e8b4be",
    "factor_catalog.yaml": "13b414fe3afe05f908d092cce2231648aea7b197646de99f7a3cee43ed7755ed",
    "profiles.yaml": "81a0fd97f310b77daeaf4586da086cc6f7d064456d0c9b13095ef4af8b8e7040",
}


@dataclass(frozen=True)
class BaseSpec:
    """One base question type (official schema) plus how it reaches this bot."""
    name: str
    family: str
    answer_type: str
    requires: tuple[str, ...]
    primitive: str                       # official `ground_truth`: the GMS primitive that defines the answer
    surface: tuple[str, ...]             # application overlay
    status: str                          # official status (of the knowlytix generator)
    description: str
    category: str = ""
    generator: str | None = None
    multiplicity: str = ""
    applicability: tuple[str, ...] = ()
    source: tuple[str, ...] = ()
    realisation: str = ""                # application overlay: what the category is for this bot


@dataclass(frozen=True)
class FactorSpec:
    name: str
    group: str
    levels: tuple[str, ...]
    gt_invariant: bool
    applies_families: tuple[str, ...]
    applies_answer_types: tuple[str, ...]
    description: str = ""
    cardinality: int = 0
    origin: str = "official"             # official | custom

    def applies_to(self, base: BaseSpec) -> bool:
        fam = "*" in self.applies_families or base.family in self.applies_families
        at = "*" in self.applies_answer_types or base.answer_type in self.applies_answer_types
        return fam and at


@dataclass(frozen=True)
class Profile:
    name: str
    description: str
    bases: tuple[str, ...]
    factors: tuple[str, ...]
    mode: str
    origin: str = "official"             # official | application


def official_dir(fallback: Path | None = None) -> tuple[Path, str]:
    """The official catalogs: from the installed ``proofloop`` package (proofloop/suite/catalogs,
    the same files beyond-ship-and-pray ships) when present, else the vendored copy."""
    import importlib.util
    spec = importlib.util.find_spec("proofloop")
    if spec is not None and spec.origin:
        d = Path(spec.origin).parent / "suite" / "catalogs"
        if all((d / f).exists() for f in OFFICIAL_SHA256):
            return d, "proofloop package"
    return (fallback or OFFICIAL_DIR), "vendored copy"


def _read(path: Path) -> Any:
    return yaml.safe_load(path.read_text())


def _factor_entries(doc: dict, origin: str) -> dict[str, FactorSpec]:
    out = {}
    for section, entries in doc.items():
        if section in _NON_FACTOR_SECTIONS:
            continue
        for e in entries:
            at = e.get("applies_to", {}) or {}
            levels = tuple(str(c) for c in e.get("categories", []))
            out[e["name"]] = FactorSpec(
                e["name"], section, levels, e.get("gt_invariant") is True,
                tuple(at.get("families", ["*"])), tuple(at.get("answer_types", ["*"])),
                e.get("description", ""), int(e.get("cardinality", len(levels))), origin)
    return out


@dataclass
class Catalog:
    bases: dict[str, BaseSpec]
    factors: dict[str, FactorSpec]
    app: dict[str, Any]
    factor_groups: dict[str, list[str]] = field(default_factory=dict)
    profiles: dict[str, Profile] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)        # documents, for validation

    @classmethod
    def load(cls, directory: str | Path | None = None) -> "Catalog":
        d = Path(directory) if directory else CATALOG_DIR
        off, origin = official_dir(d / "official")
        app = _read(d / "application.yaml")
        overlay = app.get("bases", {}) or {}

        bdoc = _read(off / "base_catalog.yaml")
        bases = {}
        for _fam, entries in bdoc.items():
            for e in entries:
                ov = overlay.get(e["name"], {}) or {}
                bases[e["name"]] = BaseSpec(
                    e["name"], e["family"], e["answer_type"], tuple(e.get("requires", [])),
                    e.get("ground_truth", ""), tuple(ov.get("surface", [])), e.get("status", "build"),
                    e.get("description", ""), e.get("category", ""), e.get("generator"),
                    e.get("multiplicity", ""), tuple(e.get("applicability", [])), tuple(e.get("source", [])),
                    ov.get("realisation", ""))

        fdoc = _read(off / "factor_catalog.yaml")
        factors = _factor_entries(fdoc, "official")
        custom_docs = {}
        for fname in app.get("custom_factor_files", []):
            cdoc = _read(d / fname)
            custom_docs[fname] = cdoc
            factors.update(_factor_entries(cdoc, "custom"))
        groups = {k: list(v["includes"]) for k, v in (fdoc.get("factor_groups") or {}).items()}

        profiles: dict[str, Profile] = {}
        pdoc = _read(off / "profiles.yaml") if (off / "profiles.yaml").exists() else {}
        for pname, p in ((pdoc or {}).get("profiles") or {}).items():
            profiles[pname] = Profile(pname, p.get("description", ""), tuple(p.get("bases", [])),
                                      tuple(p.get("factors", [])), p.get("mode", "embedded"), "official")
        for pname, p in (app.get("profiles") or {}).items():
            facs = list(app["selected"]) if p["factors"] == "*selected" else list(p["factors"])
            profiles[pname] = Profile(pname, p.get("description", ""), tuple(p.get("bases", ["*"])),
                                      tuple(facs), p.get("mode", "embedded"), "application")
        raw = {"base": bdoc, "factor": fdoc, "profiles": pdoc, "custom": custom_docs, "dir": str(d),
               "official_dir": str(off), "official_origin": origin}
        return cls(bases, factors, app, groups, profiles, raw)

    # ------------------------------------------------------------------ views
    def families(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for b in self.bases.values():
            out.setdefault(b.family, []).append(b.name)
        return out

    def summary(self) -> str:
        n_off = sum(f.origin == "official" for f in self.factors.values())
        groups = {f.group for f in self.factors.values() if f.origin == "official"}
        return (f"{len(self.bases)} base categories in {len(self.families())} families, "
                f"{n_off} catalog factors in {len(groups)} groups (+{len(self.factors) - n_off} custom), "
                f"{len(self.factor_groups)} factor groups, {len(self.profiles)} profiles; application selects "
                f"{len(self.app['selected'])}, excludes {len(self.app['excluded'])}, "
                f"moves {len(self.app['moved_to_base'])} to the base")

    def app_factor(self, name: str) -> FactorSpec:
        """The catalog factor with the application's level / relevance overrides applied."""
        f = self.factors[name]
        sel = self.app["selected"].get(name, {}) or {}
        at = sel.get("applies_to")
        return FactorSpec(
            f.name, f.group, tuple(sel.get("levels", f.levels)), f.gt_invariant,
            tuple(at["families"]) if at else f.applies_families,
            tuple(at["answer_types"]) if at else f.applies_answer_types, f.description,
            len(sel.get("levels", f.levels)), f.origin)

    def level_provenance(self, name: str) -> dict[str, str]:
        """Application level -> catalog level ('custom' for an application-only level)."""
        sel = self.app["selected"].get(name, {}) or {}
        if self.factors[name].origin == "custom":
            return {lv: "custom factor" for lv in sel.get("levels", self.factors[name].levels)}
        return dict(sel.get("level_aliases", {}))

    def profile_factors(self, profile: str) -> list[str]:
        return list(self.profiles[profile].factors)

    def expand_factors(self, tokens: list[str]) -> list[str]:
        out: list[str] = []
        for tok in tokens:
            if tok == "*":
                out += list(self.factors)
            elif tok in self.factor_groups:
                out += [n for n in self.factor_groups[tok] if n in self.factors]
            elif tok in self.factors:
                out.append(tok)
            else:
                raise KeyError(f"unknown factor or factor group {tok!r}")
        return list(dict.fromkeys(out))

    def expand_bases(self, tokens: list[str]) -> list[str]:
        fams = self.families()
        out: list[str] = []
        for tok in tokens:
            if tok == "*":
                out += list(self.bases)
            elif tok in fams:
                out += fams[tok]
            elif tok in self.bases:
                out.append(tok)
            else:
                raise KeyError(f"unknown base or family {tok!r}")
        return list(dict.fromkeys(out))


@dataclass
class ResolvedSuite:
    bases: list[BaseSpec]
    factors: list[FactorSpec]
    mode: str
    dropped_factors: list[tuple[str, str]] = field(default_factory=list)
    profile: str | None = None

    @property
    def base_names(self) -> list[str]:
        return [b.name for b in self.bases]

    @property
    def factor_names(self) -> list[str]:
        return [f.name for f in self.factors]

    def level_specs(self) -> list[tuple[str, list[str]]]:
        return [(f.name, list(f.levels)) for f in self.factors]

    def summary(self) -> str:
        return (f"suite[{self.profile or 'custom'}]: {len(self.bases)} bases x {len(self.factors)} factors, "
                f"mode={self.mode} ({len(self.dropped_factors)} factors dropped by applies_to/application)")


def resolve(cat: Catalog, bases: list[str], factors: list[str], mode: str = "embedded",
            profile: str | None = None) -> ResolvedSuite:
    if mode not in ("cross", "embedded"):
        raise ValueError("mode must be 'cross' or 'embedded'")
    bspecs = [cat.bases[b] for b in cat.expand_bases(bases)]
    kept, dropped = [], []
    for tok in cat.expand_factors(factors):
        if tok in cat.app["excluded"]:
            dropped.append((tok, f"application: {cat.app['excluded'][tok]}"))
            continue
        moved = [k for k in cat.app["moved_to_base"] if k.split(".")[0] == tok]
        if moved and tok not in cat.app["selected"]:
            dropped.append((tok, f"moved to base: {cat.app['moved_to_base'][moved[0]]}"))
            continue
        if tok not in cat.app["selected"]:
            dropped.append((tok, "not selected by the application"))
            continue
        f = cat.app_factor(tok)
        if not any(f.applies_to(b) for b in bspecs):
            dropped.append((tok, "applies_to matches no selected base"))
            continue
        kept.append(f)
    return ResolvedSuite(bspecs, kept, mode, dropped, profile)


def resolve_profile(cat: Catalog, name: str, bases: list[str] | None = None) -> ResolvedSuite:
    p = cat.profiles[name]
    return resolve(cat, bases or list(p.bases), list(p.factors), mode=p.mode, profile=name)


# ----------------------------------------------------------------------------- validation
def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_official(cat: Catalog) -> list[str]:
    """The official contract (validate_catalogs.py rules 1-5 + profile references),
    applied to the official catalogs together with the application's custom factors."""
    errs: list[str] = []
    bases = [e for entries in cat.raw["base"].values() for e in entries]
    factor_entries = [e for sec, entries in cat.raw["factor"].items() if sec not in _NON_FACTOR_SECTIONS
                      for e in entries]
    for cdoc in cat.raw["custom"].values():
        factor_entries += [e for entries in cdoc.values() for e in entries]
    base_names = {b["name"] for b in bases}
    factor_names = {f["name"] for f in factor_entries}
    for f in factor_entries:                                                # rule 1
        if f.get("gt_invariant") is not True:
            errs.append(f"factor '{f['name']}': gt_invariant must be true")
        if len(f.get("categories", [])) < 2:
            errs.append(f"factor '{f['name']}': needs at least two levels")
    for b in bases:                                                         # rule 2
        if b["family"] not in FAMILIES:
            errs.append(f"base '{b['name']}': unknown family '{b['family']}'")
        for cap in b.get("requires", []):
            if cap not in REQUIRES_VOCAB:
                errs.append(f"base '{b['name']}': unknown requires capability '{cap}'")
        if b["answer_type"] not in ANSWER_TYPES:
            errs.append(f"base '{b['name']}': unknown answer_type '{b['answer_type']}'")
        if not b.get("ground_truth"):
            errs.append(f"base '{b['name']}': no ground-truth primitive")
    for f in factor_entries:                                                # rule 3
        at = f.get("applies_to", {}) or {}
        for fam in at.get("families", []):
            if fam != "*" and fam not in FAMILIES:
                errs.append(f"factor '{f['name']}': applies_to.families has unknown '{fam}'")
        for atype in at.get("answer_types", []):
            if atype != "*" and atype not in ANSWER_TYPES:
                errs.append(f"factor '{f['name']}': applies_to.answer_types has unknown '{atype}'")
    for nm in base_names | factor_names:                                    # rule 4
        if not SNAKE.match(nm):
            errs.append(f"name '{nm}' is not snake_case")
    if len(base_names) != len(bases):
        errs.append("duplicate base names detected")
    if len(factor_names) != len(factor_entries):
        errs.append("duplicate factor names detected (custom factors must not shadow catalog factors)")
    overlap = base_names & factor_names                                     # rule 5
    if overlap:
        errs.append(f"names appear in BOTH catalogs: {sorted(overlap)}")
    groups = set(cat.factor_groups)
    for gname, members in cat.factor_groups.items():
        for m in members:
            if m not in factor_names:
                errs.append(f"factor group '{gname}': unknown factor '{m}'")
    for pname, p in cat.profiles.items():
        for b in p.bases:
            if b != "*" and b not in base_names and b not in FAMILIES:
                errs.append(f"profile '{pname}': unknown base/family '{b}'")
        for fc in p.factors:
            if fc != "*" and fc not in factor_names and fc not in groups:
                errs.append(f"profile '{pname}': unknown factor/group '{fc}'")
        if p.mode not in ("cross", "embedded"):
            errs.append(f"profile '{pname}': mode must be cross|embedded")
    return errs


def validate_application(cat: Catalog) -> list[str]:
    errs: list[str] = []
    app = cat.app
    sel, exc, moved = app["selected"], app["excluded"], app["moved_to_base"]
    accounted = set(sel) | set(exc) | {k.split(".")[0] for k in moved}
    for name in cat.factors:
        if name not in accounted:
            errs.append(f"factor {name} is neither selected, excluded nor moved to base")
    for name in sel:
        if name not in cat.factors:
            errs.append(f"application selects unknown factor {name}")
    for name in set(sel) & set(exc):
        errs.append(f"factor {name} both selected and excluded")
    for key in moved:
        fac, _, lvl = key.partition(".")
        if fac not in cat.factors:
            errs.append(f"moved_to_base names unknown factor {fac}")
        elif lvl and lvl not in cat.factors[fac].levels:
            errs.append(f"moved_to_base names unknown level {key}")
    for name, spec in sel.items():
        if name not in cat.factors:
            continue
        f, spec = cat.factors[name], (spec or {})
        levels = spec.get("levels", list(f.levels))
        if len(levels) < 2:
            errs.append(f"selected factor {name} has fewer than two levels")
        if f.origin == "custom":
            continue
        aliases = spec.get("level_aliases")
        if aliases is None:
            errs.append(f"selected factor {name} has no level_aliases back to the catalog")
            continue
        for lv in levels:
            tgt = aliases.get(lv)
            if tgt is None:
                errs.append(f"{name}: level '{lv}' has no level_alias")
            elif tgt != "custom" and tgt not in f.levels:
                errs.append(f"{name}: level '{lv}' aliases unknown catalog level '{tgt}'")
        if not any(aliases.get(lv) not in (None, "custom") for lv in levels):
            errs.append(f"{name}: no level is anchored to a catalog level")
        realised = {aliases.get(lv) for lv in levels}
        unreal = spec.get("unrealised", {}) or {}
        moved_lv = {k.split(".", 1)[1] for k in moved if k.startswith(name + ".")}
        for cl in f.levels:
            if cl not in realised and cl not in unreal and cl not in moved_lv:
                errs.append(f"{name}: catalog level '{cl}' is neither realised, unrealised with a reason, "
                            f"nor moved to the base")
        for cl in unreal:
            if cl not in f.levels:
                errs.append(f"{name}: unrealised names unknown catalog level '{cl}'")
    for b in cat.bases.values():
        if not b.surface:
            errs.append(f"base {b.name} has no application surface")
        elif not set(b.surface) <= SURFACES:
            errs.append(f"base {b.name}: unknown surface {sorted(set(b.surface) - SURFACES)}")
    for name in app.get("bases", {}):
        if name not in cat.bases:
            errs.append(f"application overlay names unknown base {name}")
    return errs


def validate(cat: Catalog) -> list[str]:
    """Return a list of contract violations (empty = valid)."""
    errs = validate_official(cat) + validate_application(cat)
    off = Path(cat.raw.get("official_dir", Path(cat.raw["dir"]) / "official"))
    for fname, sha in OFFICIAL_SHA256.items():
        p = off / fname
        if not p.exists():
            errs.append(f"official catalog {fname} missing")
        elif _sha(p) != sha:
            errs.append(f"official catalog {fname} differs from the vendored copy (sha256 drift)")
    return errs


def level_provenance_table(cat: Catalog) -> list[dict[str, Any]]:
    """One row per selected factor: app levels -> catalog levels, and unrealised levels."""
    rows = []
    for name, spec in cat.app["selected"].items():
        spec = spec or {}
        f = cat.factors[name]
        moved_lv = {k.split(".", 1)[1]: v for k, v in cat.app["moved_to_base"].items() if k.startswith(name + ".")}
        rows.append({"factor": name, "origin": f.origin, "catalog_levels": list(f.levels),
                     "levels": cat.level_provenance(name),
                     "unrealised": dict(spec.get("unrealised", {}) or {}),
                     "moved_to_base": moved_lv})
    return rows
