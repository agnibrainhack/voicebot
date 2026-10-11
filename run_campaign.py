"""Run the full GMS-grounded journey test campaign.

    python run_campaign.py --out campaign_out                      # knowlytix if installed + licensed
    python run_campaign.py --out quick --runs-per-base 3 --no-mutation --no-power
    python run_campaign.py --backend builtin                       # open baseline, no knowlytix

The knowlytix licence is read from KNOWLYTIX_LICENSE_KEY (or ~/.knowlytix/license.key);
it is never written to any output.
"""
import argparse

from gmsjourney.pipeline import CampaignConfig, run_campaign

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=None, help="policy graph JSON or structured policy markdown (default: gmsjourney/policy/charge_review_kg.json)")
    ap.add_argument("--out", default="campaign_out")
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
    a = ap.parse_args()
    cfg = CampaignConfig(out_dir=a.out, profile=a.profile, runs_per_base=a.runs_per_base, seed=a.seed,
                         run_mutation=not a.no_mutation, run_resilience=not a.no_resilience,
                         run_power=not a.no_power, run_governance=not a.no_governance, backend=a.backend,
                         kx_epochs=a.kx_epochs, kx_geode_epochs=a.kx_geode_epochs, kx_store_dir=a.kx_store_dir,
                         rebuild_store=a.rebuild_store)
    if a.policy:
        cfg.policy_path = a.policy
    run_campaign(cfg)
