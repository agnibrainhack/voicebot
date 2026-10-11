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
   -> base cases: 27 policy questions (knowlytix's two-way generator gate, else a stated symbolic check) + 54 journeys (covering array + closure)
   -> enrichment: official catalogs [proofloop] + application layer, 16 voice factors, phi_p design, 486 scenarios
   -> invariance proof: every enriched scenario shown to be the same test as its base case
   -> runtime gates calibrated (Appendix C) [forgeloop GMS gates + audit on the knowlytix store]
   -> run the bot under the runtime monitor; KB answers also judged by GMSJudge [knowlytix via proofloop]
   -> diagnosis: DOEAnalyzer [knowlytix] + per component / weak link, resilience, mutation analysis
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
python run_campaign.py --out campaign_out   # first run builds the GMS (CPU: ~10-45 min), later runs reload it (~2.5 min)
python tests/test_selfchecks.py             # 19 self-checks (the knowlytix one runs when GMSJOURNEY_KX_STORE is set)

# open baseline, any Python >= 3.10, no licence
pip install -r requirements.txt && python run_campaign.py --backend builtin --out campaign_builtin
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

## Results on the reference bot (knowlytix backend)

| | |
|---|---|
| GMS store (knowlytix) | 195 entities, 336 triples, 12 ENM entries; GEODE converged in 3 iterations; 12/12 ENM figures agree with our compiler |
| Oracle property | 12/12 facts inside their learned cap radius, every cross-tail substitution outside |
| Embedding SFT (held out) | v: noisy-alias binding 25.4% → 68.4%; u: phrasing → relation routing 80.8% → 87.2% |
| Policy-question bases | 27: 17 two-way admitted by knowlytix's gate, 10 symbolic with stated reasons (thresholds, aggregations, absence, stance); forged triple rejected |
| Substrate (knowlytix generators) | exact_recall 12/12 two-way, multi_hop 19/19 two-way, counting 71 symbolic; graph answers its own questions 100% |
| Feasible content-level pairs covered | 523 / 523; rules 32/32 · workflow edges 21/21 · escalation triggers 11/11 |
| Enriched scenarios proved invariant | 486 / 486 (4,428 customer turns); 592 / 592 meaning-changing edits rejected |
| Bot accuracy, base cases → enriched | 90.1% → 50.0% |
| Factors that drive failure (DOEAnalyzer, BH) | noise (p_adj 1.6e-10), numeric_format (1.5e-4), disfluency (0.012) |
| Runtime gate (Appendix C) | θ = 0.824 on `enables`: AUC 1.0, 0 false admits, 0 false rejects; probe sweep 15 TP / 27 TN, no errors |
| Runtime monitor | 0% false alarms on fully correct runs; 103 / 328 failed runs flagged live; hash chains verified; PSI 0.04 |
| Seeded defects (16) | 100% caught by the suites (covering 75% vs random of equal size 37.5%); **16/16 flagged live by the runtime monitor** |
| Silent failures under injected tool faults | 0 |

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

## How a base case and its enriched variants are proved to be the same test

Every customer turn has a meaning in the journey ontology: a **semantic frame** (dialog act + slots bound to
canonical individuals). Ground truth is a function of the frame sequence and the backend fixture only. For each
scenario: (1) ground truth identical; (2) every enriched turn read back to the same frame; (3) the oracle replays
the read-back frames and re-derives the same outcome; (4) geometric corroboration (surface moved, meaning did
not). Meaning-changing edits must all be rejected; they are. Evidence is exported as RDF (`ontology.ttl`).

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

## Outputs (`campaign_out/`)

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

## Layout

```
gmsjourney/
  backend.py         knowlytix availability, licence from env, encoder resolution (HF or offline stand-in)
  knowledge/         policy_kg (the policy graph: build, load, validate), kx_store (knowlytix build/load + facade),
                     kx_view (GEODE input), canonicals (export), ingest/geode/store/training (open baseline)
  export_dataset.py  dataset/*.csv
  policy/            charge_review_kg.json (source), charge_review_policy.md (structured transcription of the PDF)
  reuse/             adapters: DOEAnalyzer, GMSJudge, forgeloop gates/audit, proofloop emitters/taxonomy
  catalogs/          application.yaml (+ level_aliases), custom_factors.yaml, official/ (fallback copy)
  governance/        gates (policy, speech, GMS), monitor, audit, calibrate (Appendix C)
  ontology.py journey.py oracle.py kb_truth.py bases/ enrich/ design/ sut/ evaluation/ pipeline.py report.py
run_campaign.py · validate_catalogs (python -m gmsjourney.validate_catalogs) · tests/ · notebooks/ · DESIGN.md · NOTICE
```

Licensing: knowlytix-core is used under the Knowlytix EULA and is not included in this package; the other
knowlytix packages, proofloop and forgeloop are Apache-2.0. See `NOTICE`.
