"""Domain packs: one policy graph plus one ``domain.yaml`` per bot under test.

A domain pack is a folder under ``gmsjourney/policy/`` (or anywhere on disk)::

    policy/<domain>/
        <graph>.json          the policy graph (facts, dialog states, tools, rules, aliases)
        domain.yaml           how to test it (roles, flow, tools, fixtures, figures, lexicon,
                              kb, rules, journey, enrichment, campaign)
        application.yaml      the enrichment catalog layer (factor selection and levels)
        custom_factors.yaml   factors this application adds to the official catalogs
        <transcription>.md    optional: the policy as structured markdown

The graph is the single source of truth for what is true and allowed; ``domain.yaml`` only
names graph elements, gives them roles and adds what a graph cannot carry. Every component
reads the same loaded ``Domain`` (``domain.current()``), so the ontology, journeys,
enrichment, oracle and runtime validation cannot drift apart. ``lint()`` checks every
cross-reference between the config and its graph before a run.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

POLICY_ROOT = Path(__file__).resolve().parent / "policy"
DEFAULT_DOMAIN = "charge_review"
CONFIG_FILE = "domain.yaml"


class DomainError(ValueError):
    pass


@dataclass
class Domain:
    name: str
    dir: Path
    config: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ files
    @property
    def graph_path(self) -> Path:
        return self.dir / self.section("domain")["graph"]

    @property
    def transcription_path(self) -> Path | None:
        t = self.section("domain").get("transcription")
        return (self.dir / t) if t else None

    @property
    def catalog_dir(self) -> Path:
        """Where application.yaml / custom_factors.yaml live (the official catalogs are shared)."""
        return self.dir

    # ------------------------------------------------------------------ config access
    def section(self, name: str) -> dict[str, Any]:
        return self.config.get(name) or {}

    def get(self, dotted: str, default: Any = None) -> Any:
        """``domain.get("roles.item.name")``: a nested config value, or ``default``."""
        cur: Any = self.config
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    @property
    def root(self) -> str:
        return self.section("domain")["root"]

    # ------------------------------------------------------------------ figures
    def limit_keys(self) -> list[str]:
        """The graph's limits, in graph order (is_a <figures.limits.class>)."""
        cls = self.get("figures.limits.class", "limit")
        return [t["head"] for t in self.kg.triples if t["relation"] == "is_a" and t["tail"] == cls]

    def enm_for_relation(self) -> dict[str, str]:
        """Numeric relation -> the ENM category that backs it."""
        return {r: sp["enm_category"] for r, sp in (self.get("figures.relations") or {}).items()}

    def kx_enm_map(self) -> dict[str, tuple[str, str]]:
        """ENM category -> (knowlytix register type, key format)."""
        f = self.section("figures")
        out = {f["limits"]["enm_category"]: (f["limits"]["kx"]["type"], f["limits"]["kx"]["key"])}
        for sp in (f.get("relations") or {}).values():
            out[sp["enm_category"]] = (sp["kx"]["type"], sp["kx"]["key"])
        return out

    # ------------------------------------------------------------------ the graph
    @property
    def kg(self):
        if "_kg" not in self.__dict__:
            from .knowledge.policy_kg import PolicyKG
            self.__dict__["_kg"] = PolicyKG.load(self.graph_path)
        return self.__dict__["_kg"]

    def __repr__(self) -> str:
        return f"Domain({self.name!r}, {self.dir})"


# ---------------------------------------------------------------------- loading
def _resolve_dir(spec: str | Path | None) -> Path:
    if spec is None or spec == "":
        spec = os.environ.get("GMSJOURNEY_DOMAIN") or DEFAULT_DOMAIN
    p = Path(spec)
    if p.suffix == ".json" and p.exists():          # a graph file: its folder is the pack
        return p.resolve().parent
    if p.suffix in (".yaml", ".yml") and p.exists():
        return p.resolve().parent
    if p.is_dir():
        return p.resolve()
    cand = POLICY_ROOT / str(spec)
    if cand.is_dir():
        return cand
    raise DomainError(f"no domain pack {spec!r}: expected a folder with {CONFIG_FILE} "
                      f"(looked in {p} and {cand})")


def load(spec: str | Path | None = None) -> Domain:
    """A domain by name (``charge_review``), folder, graph file or config file;
    default: $GMSJOURNEY_DOMAIN, else ``charge_review``."""
    d = _resolve_dir(spec)
    cfg_path = d / CONFIG_FILE
    if not cfg_path.exists():
        raise DomainError(f"{d} has no {CONFIG_FILE}")
    config = yaml.safe_load(cfg_path.read_text()) or {}
    dom = Domain(name=(config.get("domain") or {}).get("id", d.name), dir=d, config=config)
    if not dom.graph_path.exists():
        raise DomainError(f"{cfg_path}: graph {dom.graph_path.name} not found")
    return dom


_CURRENT: Domain | None = None


def current() -> Domain:
    """The domain every component reads (loaded on first use)."""
    global _CURRENT
    if _CURRENT is None:
        _CURRENT = load()
    return _CURRENT


def set_current(dom: Domain | str | Path | None) -> Domain:
    global _CURRENT
    _CURRENT = dom if isinstance(dom, Domain) else load(dom)
    return _CURRENT


# ---------------------------------------------------------------------- lint
def lint(dom: Domain | None = None) -> list[str]:
    """Every problem found between the config and its graph (empty = clean)."""
    dom = dom or current()
    errs: list[str] = []
    d = dom.section("domain")
    for key in ("id", "graph", "root"):
        if key not in d:
            errs.append(f"domain.{key} missing")
    try:
        kg = dom.kg
    except Exception as e:                                # noqa: BLE001 - reported, not raised
        return errs + [f"graph does not load: {e}"]
    errs += [f"graph: {e}" for e in kg.validate()]
    heads = {t["head"] for t in kg.triples} | {t["tail"] for t in kg.triples if isinstance(t["tail"], str)}
    if d.get("root") and d["root"] not in heads:
        errs.append(f"domain.root {d['root']!r} is not an entity of the graph")
    if dom.transcription_path is not None and not dom.transcription_path.exists():
        errs.append(f"domain.transcription {dom.transcription_path.name} not found")
    if not (dom.catalog_dir / "application.yaml").exists():
        errs.append("application.yaml (enrichment catalog layer) not found in the domain folder")
    from .domain_checks import CHECKS
    for chk in CHECKS:
        errs += chk(dom, kg)
    return errs


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Lint a domain pack against its policy graph.")
    ap.add_argument("domain", nargs="?", default=None)
    a = ap.parse_args(argv)
    dom = load(a.domain)
    errs = lint(dom)
    print(f"{dom.name} ({dom.dir}): " + ("clean" if not errs else f"{len(errs)} problem(s)"))
    for e in errs:
        print("  -", e)
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
