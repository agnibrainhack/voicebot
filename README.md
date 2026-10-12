# gmsjourney — GMS-grounded testing for the charge-review voice bot

The method of *Beyond "Ship and Pray": Testing Agentic Systems with Geometric Ground Truth*, applied to the
charge-review journey in the operating procedure. The GMS core (ingest, GEODE self-correction, store training,
embedding fine-tuning, calibration, judging, design and attribution engines) comes from the **knowlytix**
package family, and the book's open companion packages **proofloop** and **forgeloop**
([beyond-ship-and-pray](https://github.com/knowlytix/beyond-ship-and-pray)) supply catalogs, emitters, runtime
gates and the audit chain. `gmsjourney` adds what those packages do not cover for a multi-turn voice bot: the
journey oracle, covering-array bases, the realiser/reader and invariance proof, runtime policy checks, the
reference bot and its mutants. Without knowlytix, the same pipeline runs on a built-in open baseline.

```
policy PDF -> policy graph JSON (head/relation/tail + dialog states + per-state policy; policy/charge_review_kg.json)
   -> GMS [knowlytix]: GEODE loop -> embed loop (v SFT) -> contradiction SFT (u) -> cap-trained store -> GMSJudge calibration
   -> base cases: 27 policy questions (knowlytix's two-way generator gate, else a stated symbolic check)
                  + 4 hallucination probes (near-miss option names, an unstated fee) + journeys (covering array + closure)
   -> enrichment: official catalogs [proofloop] + application layer, 16 voice factors, phi_p design;
                  two arms on the same design rows: template realiser (always) and LLM (--llm, transport chosen at run time)
   -> invariance proof: every enriched scenario shown to be the same test as its base case - the LLM arm
                  separately, behind extra per-turn gates and its own negative controls
   -> runtime gates calibrated (Appendix C) [forgeloop GMS gates + audit on the knowlytix store]
   -> run the bot under the runtime monitor (strict or quorum rule, numeric range attestor, per-turn latency /
      length budgets); KB answers also judged by GMSJudge [knowlytix via proofloop]
   -> diagnosis: DOEAnalyzer [knowlytix] + per component / weak link, severity + risk tiers + release gate,
      template-vs-LLM paired comparison, resilience, mutation analysis
   -> training corpora [proofloop emitters]
   -> exports: gms_canonicals.json (canonicals as the trained GMS holds them) + dataset/*.csv (every test case)
```

## The policy graph (the source every step reads)

`gmsjourney/policy/charge_review_kg.json` is the operating procedure as 720 head / relation / tail triples. Each one
names its PDF section (`pdf.2C`, ...) and carries an evidence quote that is a verbatim substring of that section;
`program` marks the two limits from the business brief and `lexicon` the curated customer synonyms (not in the PDF).

| kind | what it holds | example |
|---|---|---|
| trajectory | the tool workflow and the dialog-state machine | `ASK_AUTH leads_to PRESENT_TXN` (on PROVIDE_CREDENTIAL, when AUTHENTICATED and transactions returned, R3/R6, W3) |
| state | what holds in each of the 19 dialog states | `PRESENT_TXN accepts_act INFORM_DISPOSITION`, `PRESENT_TXN applies_rule R30`, `PRESENT_TXN limited_by lookback_days`, `PRESENT_TXN interprets_yes_as RECOGNIZED` |
| policy | rules, limits, delivery options, tools, escalation triggers, knowledge base, required opening | `R27 has_stance must_not`, `max_turns has_value 20` |
| vocabulary | value sets, aliases, dialog acts and their slots | `UNRECOGNIZED has_alias "not me"`, `INFORM_DISPOSITION has_slot delivery` (values from `delivery`) |
| domain | the domain model | `customer has_session review_session` |

Rebuild it after changing the transcription or the state layer:
`python -m gmsjourney.knowledge.policy_kg build --pdf fictional_poc_operating_procedure_gpt_readable.pdf`
(`check` validates a file). `PolicyKG.load()` is the read API; the GMS is trained on the graph, state layer
included, and the campaign checks that the trained store carries every fact (`report.json: policy_kg.consistency`).

**State-aware meaning cards.** Every customer turn is read in the dialog state it answers. The ontology resolves
slot values inside the slot's value set, reads a bare "yes"/"no" the way the state reads it (RECOGNIZED /
UNRECOGNIZED after "do you recognise this charge?", CONFIRM / DENY after "shall I go ahead?", END after "anything
else?"), treats a 4-digit number as a credential only where the state asks for one, and rejects a card the state
does not accept. The invariance proof compares (state, card) sequences.

## Quick start

```bash
# knowlytix backend: Python 3.12 (knowlytix-core ships cp312 wheels) and a runtime licence
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements-knowlytix.txt
export KNOWLYTIX_LICENSE_KEY=...            # or put it in ~/.knowlytix/license.key; never committed, never logged
python run_campaign.py --out output/campaign_out   # first run builds the GMS (CPU: ~10-45 min), later runs reload it (~2.5 min)
python tests/test_selfchecks.py             # 20 self-checks (the knowlytix one runs when GMSJOURNEY_KX_STORE is set)
python tests/test_parity_and_llm_arm.py     # 8 checks for 0.4 (LLM arm with a fake model: no key, no network)

# open baseline, any Python >= 3.10, no licence
pip install -r requirements.txt && python run_campaign.py --backend builtin --out campaign_builtin

# + the LLM enrichment arm (transport chosen at run time; see "Two enrichment arms")
pip install -r requirements-llm.txt
python tools/check_llm.py                    # one call through the configured transport: will it work here?
python run_campaign.py --llm --out output/campaign_llm                 # default transport: the model gateway
python run_campaign.py --llm --llm-provider deepseek --out campaign_ds   # or the DeepSeek API
export DEEPSEEK_API_KEY=...                  # deepseek only; read from the environment, never written to outputs
python run_campaign.py --llm --llm-mode replay --out rerun      # re-run from the cache, no network
# replay reads the cache of the provider it resolves to, so pass the same --llm-provider
# (and --llm-fallback) the recorded run used
```

Colab: upload `gmsjourney.zip`, add a secret named `KNOWLYTIX_LICENSE_KEY`, open
`notebooks/charge_journey_campaign.ipynb`, run top to bottom. Cell 9b inspects any case: set `BASE`, `VARIANT`,
`TURN` to see the seed and enriched text with their canonical frames, and every turn as Head / Relation / Tail
triples (seed vs enriched, plus the store facts grounding each value). The same views are available in code:
`gmsjourney.inspect_cases.show_case / compare_case / case_hrt`.

**Encoders.** knowlytix warm-starts and fine-tunes from HuggingFace sentence encoders (all-MiniLM-L6-v2,
nli-mpnet-base-v2). When the hub is unreachable (as in the environment these results were produced in),
`gmsjourney.backend` builds a deterministic hashing encoder in HuggingFace format so every knowlytix call still
runs; the build report records which encoders were used. With hub access the real encoders are used
automatically, and the embedding numbers below should improve.

## Results on the reference bot (knowlytix backend, trained on the policy graph; offline stand-in encoders)

| | |
|---|---|
| GMS store (knowlytix) | 229 entities, 607 triples (policy + dialog-state layer), 12 ENM entries; GEODE converged in 3 iterations; 12/12 ENM figures agree with the graph; the store carries every graph fact |
| Oracle property | 12/12 facts inside their learned cap radius, every cross-tail substitution outside |
| Embedding SFT (held out) | v: noisy-alias binding 25.4% → 73.7%; u: phrasing → relation routing 80.8% → 88.5% (your run with the real HuggingFace encoders on the previous graph: 45.6% → 81.6% and 98.7% → 100%) |
| Policy-question bases | 27: 17 two-way admitted by knowlytix's gate, 10 symbolic with stated reasons (thresholds, aggregations, absence, stance); forged triple rejected |
| Substrate (knowlytix generators) | exact_recall 12/12 two-way, multi_hop 19/19 two-way, counting 71 symbolic; graph answers its own questions 100% |
| Feasible content-level pairs covered | 523 / 523; rules 32/32 · workflow edges 21/21 · escalation triggers 11/11 |
| Enriched scenarios proved invariant | 486 / 486 (4,428 customer turns, each read in its dialog state); 632 / 632 meaning-changing edits rejected, incl. 40 wrong-state edits |
| Bot accuracy, base cases → enriched | 90.1% → 50.0% |
| Factors that drive failure (DOEAnalyzer, BH) | noise (p_adj 1.6e-10), numeric_format (1.5e-4), disfluency (0.012) |
| Runtime gate (Appendix C) | θ = 0.882 on `enables`: AUC 1.0, 0 false admits, 0 false rejects; probe sweep 15 TP / 27 TN, no errors |
| Runtime monitor | 0% false alarms on fully correct runs; 103 / 328 failed runs flagged live; hash chains verified; PSI 0.02 |
| Seeded defects (16) | 100% caught by the suites (covering 75% vs random of equal size 37.5%); **16/16 flagged live by the runtime monitor** |
| Silent failures under injected tool faults | 0 |
| GMS canonicals | 97.4% of aliases bound by the trained GMS to the canonical value the graph states; 90.2% of the 509 trained facts inside their relation's cut |
| Turn by turn (oracle vs bot) | 300 / 567 runs match the oracle at every turn; every wrong outcome diverges, first divergences mostly in PRESENT_TXN (139) and ASK_AUTH (115) |

The built-in backend reproduces the journey-level results (same bases, 486/486 proofs, 90.1% → 50.0%, 100% of
defects caught, 16/16 flagged live); its runtime gate separates less cleanly (θ = 0.979: 0 false admits, 1 of
21 policy edges rejected) and it emits no training corpora (that step needs proofloop).

The stand-in bot is a keyword-NLU state machine written independently of the oracle; its weaknesses are
realistic (no typo or speech-recognition repair, a narrow knowledge-base retriever). Swap in your bot to get the
same report for it.

## What comes from where

| Capability | Source |
|---|---|
| Ingest, GEODE self-correction, store training (cap admissibility), ENM, primitives | knowlytix `knowledge.geode` (`GeodeLoop`, `store_from_triples`), `knowledge.store.GMSExpertStore` |
| v-space SFT (rotation) and GEODE embed loop; u-space contradiction SFT (full) | knowlytix `knowledge.geode.embed_loop`, `knowlytix.embedding` |
| Calibration, geometric judging | knowlytix `harness.testing.judge.GMSJudge` (via proofloop `GeometricJudge` for the four hallucination bands) |
| Two-way admission, substrate generators | knowlytix `benchmark.eval.generators` (`QuestionGenerator._validate`, geometric=True) |
| Attribution | knowlytix `harness.graphdoe.DOEAnalyzer` (logistic + BH, joint model, interactions, failure table) |
| Official base/factor catalogs and profiles | proofloop package data (vendored copy as fallback); rules of `validate_catalogs.py` |
| Runtime gates, audit chain, injection baseline | forgeloop `gms_backend.GMSPlausibilityGate`, `audit.AuditLogger`, `governance.policies` |
| Training corpora, failure-mode taxonomy, keyword groundedness baseline, balanced design | proofloop `suite.emit`, `evaluation.failure_modes`, `evaluation.groundedness`, `evaluation.doe` |
| Policy compiler + GMS view (`policy.gms.md`) for GEODE | ours: renders the policy as one table per relation so the regex ingest proposes typed edges (`has_enables`, `has_alias`, `has_opposite_of`) |
| Journey oracle, covering array, realiser/reader, invariance proof, negative controls | ours (multi-turn journeys are outside the packages' single-question scope) |
| Runtime *policy* checks (R2/R3, R4, R11, R15, R16, R20-R22, R26, R28, R30, R31) and speech checks (claims, fraud stance, ENM figures) | ours, plugged into forgeloop's gate protocol |
| Design | ours (φp Sobol+refine with rank transform + swaps). knowlytix `DesignMatrix` sobol+refine is run on the same factors and reported alongside: at n=486 its own quality report shows 11 of 43 level cells empty (the middle level of every 3-level factor), pairwise level coverage 55%, marginal χ² p ≈ 1e-53 |
| Resilience probe, mutation analysis | ours (knowlytix's `ToolGateway` installs on a knowlytix `Harness`; our bot talks to a mock backend) |

## How the bot is validated, and the turn-by-turn view

Each test case runs the real bot (yours or the reference bot) live: a fresh bot and a fresh mock backend, a simulated
customer that answers whatever the bot actually asks (`journey.UserSimulator`), and every bot reply, act and tool call
recorded. Some checks happen during the run, turn by turn:

- each customer line is read back to its meaning card in its dialog state;
- the runtime monitor checks every tool call before it executes and every bot sentence (claims, fraud stance,
  figures);
- each policy answer is graded against the exact register.

The verdict is given on the whole run, against the oracle's answer key (`evaluation/runner.score`):

- the outcome (end state, transfer reason, what was truly recorded, which card action truly completed);
- each stage's decision;
- whether every step was admissible (workflow edges, login before account tools, limits, confirmation and fee
  disclosure before submit, no claim without SUCCESS, 20-turn limit).

The bot's reply at turn *i* is not required to equal the oracle's: a different but allowed route is accepted.

`gmsjourney/turn_compare.py` shows the two runs side by side, one row per customer turn:

- the state the customer answered and the words the bot received;
- the oracle's expected response: tool calls with arguments and results, then bot acts such as
  `PRESENT_TXN(T2: Starbucks $6.25)` or `TRANSFER(card_options_unavailable)`;
- the bot's actual tools, acts and text;
- whether they match, and the first turn where the runs diverge.

The oracle decides what to do, not the wording, so the comparison is on tool calls and decision acts.
Acknowledgements such as CONFIRM_RECORDED are ignored there and checked on the tool call itself.

On clean base cases the reference bot matches the oracle turn by turn on every run (a self-check). On enriched cases,
every wrong outcome has a divergence, and some correct outcomes do too: the bot took a different but recoverable
route.

Use `show_turns(runner, oracle, scenario)` in the notebook (cell 11b). The campaign writes
`dataset/turn_by_turn.csv` (every turn of every case) and `dataset/turn_by_turn_summary.csv` (per case: first
divergence, expected vs actual outcome, stages wrong, violations).

## How a base case and its enriched variants are proved to be the same test

Every customer turn has a meaning in the journey ontology: a **semantic frame** (dialog act + slots bound to
canonical individuals). Ground truth is a function of the frame sequence and the backend fixture only. For each
scenario: (1) ground truth identical; (2) every enriched turn read back to the same frame; (3) the oracle replays
the read-back frames and re-derives the same outcome; (4) geometric corroboration (surface moved, meaning did
not). Meaning-changing edits must all be rejected; they are. Evidence is exported as RDF (`ontology.ttl`).

## Two enrichment arms: template and LLM

Both arms realise the **same design rows** (same base, same factor levels), so their results can be compared pair by
pair. The template arm builds each customer turn from its meaning card with deterministic code. The LLM arm
(`gmsjourney/enrich/llm`, `--llm`) asks a chat model to write the turn. The transport is chosen at run time with
`--llm-provider`: `aexp` (default) is an OpenAI-compatible model gateway authenticated with a short-lived
`gcloud` access token — `AEXP_ACCESS_TOKEN` and `AEXP_USE_CASE_ID` override the CLI and the routing header —
and `deepseek` is the DeepSeek API through LangChain. `--llm-fallback` (default `deepseek`) names the transport
to use when the chosen one cannot be built or fails mid-run; the switch is recorded in the report, and because
the provider is part of the cache key, a fallback neither collides with nor discards the other's cached
generations. It is not trusted to construct the meaning, so it is validated separately and more strictly:

1. **Blind generation.** The model is given the agent's question for the dialog state, what the customer means (in
   plain words), the base wording and the presentation instructions. It never sees the expected bot behaviour, the
   oracle, tool names or rule ids (`prompts.py`).
2. **Per-turn gates** (`gates.py`, `validate.py`). A turn is accepted only when all of these pass:
   - **G0 form:** one spoken line, with no internal names.
   - **G1 facts:** no figure or merchant that the base meaning does not carry, and every decision value kept.
   - **G3 one clear decision:** no competing decision cues.
   - **G2 meaning:** the rule reader reads the final text, after surface noise, in its dialog state and gets the
     base meaning card. With `--llm-dual-read gate` (the default), an independent LLM reader that never saw the base
     must also get it.
   - **Retries:** a failing turn is regenerated, up to 3 times. The feedback names the failed gate, never the answer.
     A scenario that still fails is excluded and reported as *not materialised*.
3. **The same scenario-level test as the template arm:**
   - L1: the ground truth is identical;
   - L3: the oracle, replaying the read-back cards, re-derives the outcome and every stage;
   - the conversation signatures are equal.
4. **Negative controls on LLM text**, reported separately:
   - meaning-changing edits of the accepted LLM utterances (negation, digit, figure, delivery, card action,
     merchant, bare yes/no);
   - adversarial rewrites in which the LLM changes exactly one meaning element;
   - the wrong-state control.

   All of these go through the same acceptance function and must be rejected. Accepted controls are listed for
   audit in `dataset/llm_negative_controls_audit.csv`.
5. **Manipulation check.** Each requested factor level is checked for whether it is visible in the text. The LLM arm
   is attributed on the levels that actually reached the bot.

Accepted LLM utterances are pinned. The run sends the bot exactly the text that was proved. A turn the proof never
saw (the bot left the oracle's path) is generated live under the same gates, or falls back to the template arm and is
counted. Generations are cached by content hash, so a campaign re-runs byte for byte with `--llm-mode replay`.

## gmsh parity (0.4)

These additions follow from the gmsh comparison (`UPSTREAM_GMSH_ISSUES.md` lists the gmsh defects to report):

| addition | where |
|---|---|
| GEODE `dedup=False`, so the loop's retrains and anchor check see every fact; duplicate resolution runs audit-only on the trained store; anchor violations reported. `--kx-dedup-mode audit` keeps the 0.3 behaviour; `accept` is stock gmsh. | `knowledge/kx_store.py` |
| v-space fine-tuning A/B: rotation (book) vs full (gmsh default), keeping the better held-out noisy-alias binding (`--kx-v-sft-mode`) | `knowledge/kx_store.py` |
| gmsh's P=0.5 logistic operating point next to the regulated one, k-fold held-out accuracy, bootstrap 95% intervals for θ, the plan budget and every relation's cut | `governance/calibrate.py` |
| Quorum REVIEW (`--decision-rule quorum`). Always shadow-computed and reported: a syntax error or ≥2 blocking layers deny; one blocking layer is routed to review (escalation). | `governance/monitor.py` |
| Numeric range attestor on every figure the bot speaks: PASS (register value), REVIEW (a register value of another quantity), FAIL (invented) | `governance/attestor.py` |
| Per-turn latency and length budgets plus a tool-call budget, reported next to correctness | `evaluation/constraints.py` |
| Severity per run with gmsh's risk-tier escalation, and a tiered release gate | `evaluation/severity.py` |
| Hallucination probes: near-miss option names and an unstated fee. The bot must abstain; a new mutant (`kb_fabricate_unknown`) fabricates instead. | `bases/kb.py`, `kb_truth.py` |

## Testing your real bot

```python
from gmsjourney.pipeline import CampaignConfig, run_campaign
from gmsjourney.sut.adapter import CallableBotAdapter

def factory(tools):                       # point your bot's tool client at `tools`
    session = MyBot(tool_client=tools)
    return CallableBotAdapter(start_fn=session.start, respond_fn=session.respond)

run_campaign(CampaignConfig(out_dir="my_bot_run", bot_factory=factory, bot_name="MyBot"))
```

`HTTPBotAdapter` drives a deployed bot. If your bot returns text only, `BotActClassifier` infers its acts. The
runtime monitor wraps the same tool client, so the gates watch your bot exactly as they watch the reference bot;
`GovernanceConfig(mode="enforce")` turns DENY into a blocked call.

## Outputs (`output/campaign_out/`)

| file | content |
|---|---|
| `report.html` / `report.json` | the campaign report / every number in it |
| `gms_build_report.json` | knowlytix build: GEODE diagnostics, embed loop and SFT evaluations, judge thresholds, ENM agreement, timings |
| `policy.gms.md` | the GEODE input rendered from the policy |
| `results_enriched.csv`, `results_base_level.csv` | one row per scenario (`f_<factor>`, `correct`, `c_<stage>`, gate alarms, ...) |
| `failures.jsonl`, `scenarios.jsonl` | failed runs with transcripts; every scenario with its proof |
| `sft_classifier.jsonl`, `sft_draft.jsonl` | proofloop-schema training corpora |
| `ontology.ttl`, `store_graph.dot` | journey ontology + invariance evidence; store subgraph (Graphviz) |
| `policy_kg.json` | the policy graph the run was built from |
| `gms_canonicals.json` | canonical values, aliases and how the trained GMS binds them, exact numbers, states, acts, every trained fact with its score and cut |
| `dataset/test_cases.csv` | one row per test case (81 base + 486 enriched): category, policy question, canonicals, meaning cards, base case, enrichment, enriched output, expected outcome, complete trajectory, proof verdict |
| `dataset/test_turns.csv` | one row per customer turn: state, stage, the state's rules/limits/accepted cards, meaning card and its H/R/T, base and enriched text, read-back card, equivalence |
| `dataset/policy_questions.csv`, `dataset/test_cases.jsonl` | the 27 policy questions; the cases with nested structure |
| `dataset/turn_by_turn.csv`, `dataset/turn_by_turn_summary.csv` | per customer turn: oracle's expected tools and bot acts vs the bot's actual tools, acts and text; per case: first divergence and outcome |
| `results_enriched_llm.csv`, `scenarios_llm.jsonl` | LLM arm: one row per scenario (realised factor levels); every LLM scenario with its gates and proof |
| `dataset/llm_enrichment_turns.csv` | every LLM-written turn: base, template and LLM text, both readers' cards, each gate, attempts, factors requested / realised |
| `dataset/llm_negative_controls_audit.csv`, `dataset/arm_pairs.csv` | controls the LLM gate accepted (for human audit); template-vs-LLM pairs |

## Layout

```
gmsjourney/
  backend.py         knowlytix availability, licence from env, encoder resolution (HF or offline stand-in)
  knowledge/         policy_kg (the policy graph: build, load, validate), kx_store (knowlytix build/load + facade),
                     kx_view (GEODE input), canonicals (export), ingest/geode/store/training (open baseline)
  export_dataset.py  dataset/*.csv
  turn_compare.py    turn by turn: oracle's expected bot response vs the bot's actual response
  policy/            charge_review_kg.json (source), charge_review_policy.md (structured transcription of the PDF)
  reuse/             adapters: DOEAnalyzer, GMSJudge, forgeloop gates/audit, proofloop emitters/taxonomy
  catalogs/          application.yaml (+ level_aliases), custom_factors.yaml, official/ (fallback copy)
  governance/        gates (policy, speech, GMS), monitor (strict / quorum), attestor (numeric range), audit,
                     calibrate (Appendix C + P=0.5, holdout, bootstrap)
  enrich/llm/        the LLM enrichment arm: client (gateway or DeepSeek, replay cache), prompts, gates,
                     realizer, extractor (independent reader), validate (separate proof + controls), export
  evaluation/        runner, attribution, stats, constraints (budgets), severity (tiers + release gate), ...
  ontology.py journey.py oracle.py kb_truth.py bases/ enrich/ design/ sut/ evaluation/ pipeline.py report.py
run_campaign.py · validate_catalogs (python -m gmsjourney.validate_catalogs) · tests/ · notebooks/ · DESIGN.md · NOTICE
tools/             check_llm (one-call transport preflight), gmsh_upstream_repro,
                   answer_sheet (the case answer key), build_annotated (the core-file listing)
docs/              generated walkthroughs: EXPLAINER, INVARIANCE, ANSWER_SHEET, ANNOTATED
output/            generated campaign runs (report, dataset/, corpora) — gitignored
UPSTREAM_GMSH_ISSUES.md · fictional_poc_operating_procedure_gpt_readable.pdf (source document, not published)
```

The package, the notebook and the tools are the source; `docs/` and `output/` hold everything generated,
so a clean checkout is `gmsjourney/ notebooks/ tests/ tools/ *.py *.md *.txt *.toml` — regenerate the rest
with `run_campaign.py` and the two tools.

Licensing: knowlytix-core is used under the Knowlytix EULA and is not included in this package; the other
knowlytix packages, proofloop and forgeloop are Apache-2.0. See `NOTICE`.
