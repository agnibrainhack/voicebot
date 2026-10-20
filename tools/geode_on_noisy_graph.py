"""Feed the policy graph (clean or noisy) straight into knowlytix's GEODE loop - no cleaning of ours - and record
what GEODE itself removes, merges, flags or reports.

    python tools/geode_on_noisy_graph.py noisy output/geode_noise 80
    python tools/geode_on_noisy_graph.py clean output/geode_noise 80     # control: what GEODE drops from the clean graph

Every GMS triple of the graph is rendered, written exactly as the graph has it, as GEODE table input
(policy_<label>.gms.md), and GeodeLoop runs with its own duplicate resolution (dedup=True), surface-variant
canonicalisation (canonicalize=True, with the v encoder) and exact-number anchor checks. The result, including
GEODE's audit records, is written to geode_<label>.json. ~20 min per run on CPU at 80 epochs.
"""
import json, sys, time
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gmsjourney import backend

label, out = sys.argv[1], Path(sys.argv[2])
epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 80
out.mkdir(parents=True, exist_ok=True)
pol = ROOT / "gmsjourney" / "policy"
data = json.loads((pol / ("charge_review_kg.json" if label == "clean" else "charge_review_kg_noisy.json")).read_text())


def row(*c):
    return "| " + " | ".join(str(x) for x in c) + " |"


def render(triples):
    """Every GMS triple of the graph as GEODE table input, written as the graph has it (no normalising)."""
    lines = ["# Charge Review Voice-Bot Policy (policy graph, as extracted)", ""]
    num = defaultdict(list)
    by_rel = defaultdict(list)
    for t in triples:
        if not t.get("gms") and t["relation"] not in ("in_stage",):
            continue
        if t["relation"] in ("has_value",):
            num["Limits", "limit", "value"].append((t["head"], t["tail"]))
        elif t["relation"] in ("has_fee_usd", "has_business_days_min", "has_business_days_max"):
            col = t["relation"][4:]
            num[f"Delivery {col}", "delivery_option", col].append((t["head"], t["tail"]))
        elif t["relation"] in ("has_alias", "has_text", "states", "has_condition", "when_used"):
            continue
        else:
            by_rel[t["relation"]].append((t["head"], t["tail"]))
    for (title, a, b), rows in num.items():
        lines += [f"## {title}", "", row(a, b), row("---", "---")] + [row(h, v) for h, v in rows] + [""]
    for r, rows in sorted(by_rel.items()):
        col = r[4:] if r.startswith("has_") else r
        lines += [f"## Relation {col}", "", row("entity", col), row("---", "---")] + [row(h, v) for h, v in rows] + [""]
    return "\n".join(lines) + "\n"


view = out / f"policy_{label}.gms.md"
view.write_text(render(data["triples"]))
print("view rows:", sum(1 for l in view.read_text().splitlines() if l.startswith("| ") and "---" not in l))

import torch
from knowlytix.knowledge.geode import GeodeLoop, make_default_trainer
from knowlytix.knowledge.geode.rag import fact_only_triples
corpus = view.read_text().splitlines()
enc = backend.resolve_encoders(corpus)
print("encoders:", enc["kind"])
v_enc = None
try:
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(enc["v"], device="cpu")
    v_enc = lambda xs: st.encode(list(xs), normalize_embeddings=True)
except Exception as e:
    print("no v encoder for canonicalisation:", type(e).__name__, str(e)[:200])
torch.manual_seed(42)
dev = torch.device("cpu")
t0 = time.time()
loop = GeodeLoop(make_default_trainer(dev, epochs=epochs), device=dev, ingest_mode="regex", max_iters=6,
                 dedup=True, canonicalize=v_enc is not None, v_encoder=v_enc)
res = loop.run(str(view))
kept = fact_only_triples(res.triples)
rep = {"label": label, "epochs": epochs, "seconds": round(time.time() - t0), "converged": res.converged,
       "iterations": res.iterations, "n_final": len(res.triples), "n_fact": len(kept),
       "final_triples": [list(map(str, t)) for t in res.triples],
       "corrections": res.corrections, "anchor_violations": res.anchor_violations,
       "duplicates_removed": res.duplicates_removed, "duplicates_flagged": res.duplicates_flagged,
       "canonicalizations": res.canonicalizations, "canonicalize_flagged": res.canonicalize_flagged}
(out / f"geode_{label}.json").write_text(json.dumps(rep, indent=1, default=str))
print({k: (len(v) if isinstance(v, list) else v) for k, v in rep.items()})
