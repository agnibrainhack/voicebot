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
    ap.add_argument("--max-cases", type=int, default=None,
                    help="test only this many base cases (each with its enriched variants); default: all")
    ap.add_argument("--no-progress", action="store_true", help="no tqdm progress bars")
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
    ap.add_argument("--llm-max-scenarios", type=int, default=None,
                    help="at most this many design rows rewritten by the LLM (spread over the bases)")
    ap.add_argument("--llm-controls", type=int, default=None, help="deterministic controls per type on LLM text")
    ap.add_argument("--llm-max-attempts", type=int, default=3, help="attempts per turn (1 = no retries)")
    ap.add_argument("--llm-target", type=int, default=None,
                    help="prove LLM candidates until this many conversations are accepted")
    ap.add_argument("--llm-max-candidates", type=int, default=None, help="rows tried for --llm-target (default 3x)")
    ap.add_argument("--llm-asides", default="verbatim", choices=["verbatim", "paraphrase"])
    ap.add_argument("--llm-turn-fallback", default="template", choices=["template", "none"])
    ap.add_argument("--max-turn-limit-cases", type=int, default=2,
                    help="20+ turn conversations (turn-limit tests) among the journey bases; -1 = no cap")
    ap.add_argument("--kb-questions", default="standard", choices=["standard", "extended"],
                    help="extended: more policy questions over the same facts")
    ap.add_argument("--llm-all-rows", action="store_true",
                    help="let the LLM rewrite policy-question and turn-padded rows too (default: excluded)")
    ap.add_argument("--llm-kb-turns", default="template", choices=["template", "llm"],
                    help="policy-question turns inside LLM rows: keep template wording (default) or rewrite")
    a = ap.parse_args()
    cfg = CampaignConfig(out_dir=a.out, profile=a.profile, runs_per_base=a.runs_per_base, seed=a.seed,
                         max_cases=a.max_cases, progress=not a.no_progress,
                         run_mutation=not a.no_mutation, run_resilience=not a.no_resilience,
                         run_power=not a.no_power, run_governance=not a.no_governance, backend=a.backend,
                         kx_epochs=a.kx_epochs, kx_geode_epochs=a.kx_geode_epochs, kx_store_dir=a.kx_store_dir,
                         rebuild_store=a.rebuild_store, kx_dedup_mode=a.kx_dedup_mode,
                         kx_v_sft_mode=a.kx_v_sft_mode, decision_rule=a.decision_rule,
                         budgets={"max_bot_latency_ms": a.max_bot_latency_ms, "max_bot_words": a.max_bot_words},
                         llm_arm=a.llm, llm_mode=a.llm_mode, llm_model=a.llm_model or "", llm_temperature=a.llm_temperature,
                         llm_share=a.llm_share, llm_dual_read=a.llm_dual_read,
                         llm_adversarial_controls=a.llm_adversarial, llm_cache_dir=a.llm_cache_dir,
                         llm_workers=a.llm_workers, llm_max_scenarios=a.llm_max_scenarios,
                         llm_controls_per_type=a.llm_controls, llm_max_attempts=a.llm_max_attempts,
                         llm_target_accepted=a.llm_target, llm_max_candidates=a.llm_max_candidates,
                         llm_asides=a.llm_asides, llm_turn_fallback=a.llm_turn_fallback,
                         max_turn_limit_cases=None if a.max_turn_limit_cases < 0 else a.max_turn_limit_cases,
                         kb_questions=a.kb_questions,
                         llm_exclude=() if a.llm_all_rows else ("policy_questions", "turn_padding"),
                         llm_template_acts=("ASK_KB",) if a.llm_kb_turns == "template" else ())
    if a.policy:
        cfg.policy_path = a.policy
    run_campaign(cfg)
