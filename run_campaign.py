"""Run the full GMS-grounded journey test campaign.

    python run_campaign.py --out output/campaign_out               # knowlytix if installed + licensed
    python run_campaign.py --out quick --runs-per-base 3 --no-mutation --no-power
    python run_campaign.py --backend builtin                       # open baseline, no knowlytix
    python run_campaign.py --llm                                   # + the LLM enrichment arm (Microsoft gateway)
    python run_campaign.py --llm --llm-mode replay                 # LLM arm from the cache only, no network
    python run_campaign.py --decision-rule quorum                  # gmsh quorum REVIEW in the runtime gates

The knowlytix licence is read from KNOWLYTIX_LICENSE_KEY (or ~/.knowlytix/license.key);
the gateway token from gcloud (or AEXP_ACCESS_TOKEN). Neither is ever written to any output.
"""
import argparse

from gmsjourney.pipeline import CampaignConfig, run_campaign

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=None, help="policy graph JSON or structured policy markdown (default: gmsjourney/policy/charge_review_kg.json)")
    ap.add_argument("--out", default="output/campaign_out")
    ap.add_argument("--profile", default="voice_full")
    ap.add_argument("--runs-per-base", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backend", default="auto", choices=["auto", "knowlytix", "builtin"])
    ap.add_argument("--kx-epochs", type=int, default=150, help="production store epochs (licence cap 200)")
    ap.add_argument("--kx-geode-epochs", type=int, default=80)
    ap.add_argument("--kx-store-dir", default=None, help="where the knowlytix store is built / reused")
    ap.add_argument("--rebuild-store", action="store_true")
    ap.add_argument("--no-mutation", action="store_true")
    ap.add_argument("--no-resilience", action="store_true")
    ap.add_argument("--no-power", action="store_true")
    ap.add_argument("--no-governance", action="store_true")
    ap.add_argument("--kx-dedup-mode", default="off", choices=["off", "audit", "accept"],
                    help="GEODE duplicate resolution: off (audit-only after training), audit (0.3), accept (stock gmsh)")
    ap.add_argument("--kx-v-sft-mode", default="ab", choices=["rotation", "full", "ab"],
                    help="v-space fine-tuning: rotation (book), full (gmsh default) or A/B both and keep the better")
    ap.add_argument("--decision-rule", default="strict", choices=["strict", "quorum"])
    ap.add_argument("--max-bot-latency-ms", type=float, default=2500.0)
    ap.add_argument("--max-bot-words", type=int, default=60)
    ap.add_argument("--llm", action="store_true", help="add the LLM enrichment arm")
    ap.add_argument("--llm-mode", default="live", choices=["live", "replay"])
    ap.add_argument("--llm-model", default=None, help="gateway deployment (default gpt-41; gpt-55 for 5.5)")
    ap.add_argument("--llm-temperature", type=float, default=0.7)
    ap.add_argument("--llm-share", type=float, default=1.0, help="share of design rows also realised by the LLM")
    ap.add_argument("--llm-dual-read", default="gate", choices=["gate", "report", "off"])
    ap.add_argument("--llm-adversarial", type=int, default=20, help="adversarial LLM rewrites used as controls")
    ap.add_argument("--llm-cache-dir", default=None)
    ap.add_argument("--llm-workers", type=int, default=8, help="concurrent gateway calls (1 = sequential)")
    a = ap.parse_args()
    cfg = CampaignConfig(out_dir=a.out, profile=a.profile, runs_per_base=a.runs_per_base, seed=a.seed,
                         run_mutation=not a.no_mutation, run_resilience=not a.no_resilience,
                         run_power=not a.no_power, run_governance=not a.no_governance, backend=a.backend,
                         kx_epochs=a.kx_epochs, kx_geode_epochs=a.kx_geode_epochs, kx_store_dir=a.kx_store_dir,
                         rebuild_store=a.rebuild_store, kx_dedup_mode=a.kx_dedup_mode,
                         kx_v_sft_mode=a.kx_v_sft_mode, decision_rule=a.decision_rule,
                         budgets={"max_bot_latency_ms": a.max_bot_latency_ms, "max_bot_words": a.max_bot_words},
                         llm_arm=a.llm, llm_mode=a.llm_mode, llm_model=a.llm_model or "", llm_temperature=a.llm_temperature,
                         llm_share=a.llm_share, llm_dual_read=a.llm_dual_read,
                         llm_adversarial_controls=a.llm_adversarial, llm_cache_dir=a.llm_cache_dir,
                         llm_workers=a.llm_workers)
    if a.policy:
        cfg.policy_path = a.policy
    run_campaign(cfg)
