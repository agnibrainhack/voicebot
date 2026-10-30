# gmsjourney — GMS-grounded testing for the charge-review voice bot

The method of *Beyond "Ship and Pray": Testing Agentic Systems with Geometric Ground Truth*, applied to the
charge-review journey in the operating procedure. The GMS core (ingest, GEODE self-correction, store training,
embedding fine-tuning, calibration, judging, design and attribution engines) comes from the **knowlytix**
package family, and the book's open companion packages **proofloop** and **forgeloop**
([beyond-ship-and-pray](https://github.com/knowlytix/beyond-ship-and-pray)) supply catalogs, emitters, runtime
gates and the audit chain. `gmsjourney` adds what those packages do not cover for a multi-turn voice bot: the
journey oracle, covering-array bases, the realiser/reader and invariance proof, runtime policy checks, the
reference bot and its mutants. knowlytix, proofloop and forgeloop are required (`requirements.txt`, Python 3.12,
a knowlytix licence); there is no built-in fallback.

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
      length budgets); KB answers judged exactly against the register
   -> diagnosis: deviance tests per factor + per component / weak link, severity + risk tiers + release gate,
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
# Python 3.12 (knowlytix-core ships cp312 wheels), knowlytix + proofloop + forgeloop, and a runtime licence
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export KNOWLYTIX_LICENSE_KEY=...            # or put it in ~/.knowlytix/license.key; never committed, never logged
python run_campaign.py --out output/campaign_out   # first run builds the GMS (CPU: ~10-45 min), later runs reload it (~2.5 min)
python tests/test_selfchecks.py             # self-checks on the campaign's knowlytix store (built once, then reloaded)
python tests/test_parity_and_llm_arm.py     # LLM arm checks with a fake model: no gateway key, no network
python tests/test_llm_sut.py                # the LLM bot under test, with a fake model

# + the LLM enrichment arm (the Microsoft model gateway; see "Two enrichment arms")
pip install -r requirements-llm.txt
export GMSJOURNEY_LLM_URL="https://gaiapis.aexp.com/gcpapi/<path>/completions"   # exact endpoint
python tools/check_llm.py                    # one call through the gateway: will it work here?
python run_campaign.py --llm --out output/campaign_llm
python run_campaign.py --llm --llm-mode replay --out rerun      # re-run from the cache, no network
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
| Bot accuracy, base cases → enriched | 89.7% → 50.3% (outcome alone 90.8% → 52.0%; the difference is runs whose outcome was right and whose words were not) |
| Factors that drive failure (BH-adjusted; this run used knowlytix DOEAnalyzer, since replaced by our deviance tests) | noise (p_adj 1.6e-10), numeric_format (1.5e-4), disfluency (0.012) |
| Runtime gate (Appendix C) | θ = 0.882 on `enables`: AUC 1.0, 0 false admits, 0 false rejects; probe sweep 15 TP / 27 TN, no errors |
| Runtime monitor | 0 false alarms on 244 fully correct enriched runs; 46 / 277 failed runs flagged live; hash chains verified |
| Seeded defects (21, four of them in the words only) | 100% caught by the suites (covering 85.7% vs random of equal size 52.4%); **21/21 flagged live by the runtime monitor**. Two of them (the opposite answer said back, a summary that leaves one out) are flagged only by `speech_grounding` |
| What the bot said (speech check) | 789 claims on 87 base runs and 3,975 on 521 enriched runs. 85/87 and 501/521 runs fully grounded. Every wrong figure left is a real KB defect of the reference bot ("Expedited delivery costs $0." to "how much more is expedited?", the wrong delivery days). The flow findings are almost all real too: a policy sentence recited to "could you say that again?", a decline acted on after "no wait, yes". The exception is 2 safe re-asks of the fee. 392/392 word-only control edits flagged. knowlytix's AnswerVerifier agrees on 56/56 and 260/260 of the figures it can read |
| Silent failures under injected tool faults | 0 |
| GMS canonicals | 97.4% of aliases bound by the trained GMS to the canonical value the graph states; 90.2% of the 509 trained facts inside their relation's cut |
| Turn by turn (oracle vs bot) | 300 / 567 runs match the oracle at every turn; every wrong outcome diverges, first divergences mostly in PRESENT_TXN (139) and ASK_AUTH (115) |

The stand-in bot is a keyword-NLU state machine written independently of the oracle; its weaknesses are
realistic (no typo or speech-recognition repair, a narrow knowledge-base retriever). Swap in your bot to get the
same report for it.

## What comes from where

| Capability | Source |
|---|---|
| Ingest, GEODE self-correction, store training (cap admissibility), ENM, primitives | knowlytix `knowledge.geode` (`GeodeLoop`, `store_from_triples`), `knowledge.store.GMSExpertStore` |
| Build parameters | knowlytix's reference build (`knowledge.geode.rag.build_calibrated_rag_store`): GEODE trainer 300 epochs × 8 rounds, embed loop 2 rounds on a 150-epoch trainer, v SFT `full` rank 8, u contradiction SFT 400 epochs, production store `DocGMSConfig` defaults. The reference takes the store's `DocGMSConfig` from its caller; ours sets `loss_mode="cap"`, `ingest_mode="regex"` (the reference default), `train.epochs=150`. Stated differences: the encoders are passed by local path (the same models), dedup runs audit-only (`--kx-dedup-mode`), u is trained on our attribute phrasings. All settable, see below |
| Run-time geometry | knowlytix `GMSExpertStore`: `score_triple` (an unstated tail via `score_triple_emb` on the tuned v encoder), `cap_radius`, `tension_energy`, `check_holonomy`, `is_path_consistent` (with the store's calibrated `tau_path` written to `config.verify`), `lookup_enm`. Grounding is `score_triple ≤ cap_radius`; knowlytix's per-head calibrated admissibility (`core.graph.admissibility`) is reported next to it in the oracle check |
| v-space SFT and GEODE embed loop; u-space contradiction SFT | knowlytix `knowledge.geode.embed_loop`, `knowlytix.embedding` |
| Calibration | knowlytix `harness.testing.judge.GMSJudge` |
| Two-way admission, substrate generators | knowlytix `benchmark.eval.generators` (`QuestionGenerator._validate`, geometric=True) |
| Attribution | ours: deviance tests per factor with BH adjustment, joint model, interactions, per component / weak link |
| Official base/factor catalogs and profiles | proofloop package data (a vendored copy pins the expected files); contract rules in `catalog.validate` |
| Runtime gates, audit chain, injection baseline | forgeloop `gms_backend.GMSPlausibilityGate`, `audit.AuditLogger`, `governance.policies` |
| Training corpora, failure-mode taxonomy | proofloop `suite.emit`, `evaluation.failure_modes` |
| Policy compiler + GMS view (`policy.gms.md`) for GEODE | ours: renders the policy as one table per relation so the regex ingest proposes typed edges (`has_enables`, `has_alias`, `has_opposite_of`) |
| Journey oracle, covering array, realiser/reader, invariance proof, negative controls | ours (multi-turn journeys are outside the packages' single-question scope) |
| Runtime *policy* checks (R2/R3, R4, R11, R15, R16, R20-R22, R26, R28, R30, R31) and speech checks (claims, fraud stance, ENM figures) | ours, plugged into forgeloop's gate protocol |
| Design | ours (φp Sobol+refine with rank transform + swaps; knowlytix `DesignMatrix` was compared earlier and left 11 of 43 level cells empty at n=486, so it is not used) |
| Resilience probe, mutation analysis | ours (knowlytix's `ToolGateway` installs on a knowlytix `Harness`; our bot talks to a mock backend) |

### Setting the build parameters at run time

Every build parameter can be set from `CampaignConfig`, `run_campaign.py` or the notebook, and all of them
are part of the store's cache key (a changed value builds a new store; an unchanged one reloads it):

| `CampaignConfig` field | CLI flag | default |
|---|---|---|
| `kx_geode_epochs` | `--kx-geode-epochs` | 300 |
| `kx_geode_max_iters` | `--kx-geode-max-iters` | 8 |
| `kx_embed_trainer_epochs` | `--kx-embed-trainer-epochs` | 150 |
| `kx_embed_max_iters` | `--kx-embed-max-iters` | 2 |
| `kx_sft_epochs` | `--kx-sft-epochs` | package default (100) |
| `kx_contradiction_epochs` | `--kx-contradiction-epochs` | 400 |
| `kx_v_sft_mode` | `--kx-v-sft-mode` | `full` |
| `kx_dedup_mode` | `--kx-dedup-mode` | `off` (audit-only) |
| `kx_epochs` | `--kx-epochs` | 150 (shortcut for `train.epochs`; licence cap 200) |
| `kx_store_config` | `--kx-store-config` (JSON) | `{}`: any `DocGMSConfig` field by dotted key, e.g. `{"cap.margin": 0.3, "geometry.d_v": 256, "loss_mode": "point"}` |

The overrides are saved in the build report and applied again when the store is reloaded.

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

## Validating what the bot says (`gmsjourney/speech/`)

The checks above grade what the bot *did*: its tool calls and its dialog-act labels. A bot can do everything right
and still say the wrong thing, for example by labelling a turn `CONFIRM_RECORDED(T1, UNRECOGNIZED)` and recording
UNRECOGNIZED, but telling the customer "I've noted that you **recognize** the Amazon charge". The speech check reads
every sentence the customer hears and grades it against the policy graph, the GMS and what actually happened in the
session.

```
 bot turn: acts + words                                     what is true
 ───────────────────────                                    ────────────────────────────────────────────
 CONFIRM_RECORDED(T1, UNRECOGNIZED)                         policy graph: what each act must say, and
 "I've noted that you recognize the Amazon charge."           where each spoken value's truth lives
        │                                                   session: transactions returned, writes that
        ▼  reader (graph phrasings, no LLM)                   truly succeeded, what the customer said
 claims: disposition=RECOGNIZED about T1, mode=recorded     GMS: exact register (fees, days, limits),
        │                                                     rule stances, geometric distance vs cap
        ▼  verifier
 contradicted: the successful record for T1 says UNRECOGNIZED (R13)
 finding: text_contradicts_label (CONFIRM_RECORDED)
```

**What the graph declares** (a `speech` section, `gms=False`, so the GMS view and the trained store do not change):

| Relation | Example | Meaning |
|---|---|---|
| `says` | `PRESENT_TXN says merchant` (from `item.merchant`, required) | the facts an act must speak, and where their truth lives |
| `spoken_cue` | `ACTION_DONE spoken_cue "that's done"` | words that show the act was said, when a bot gives no labels |
| `spoken_as` | `UNRECOGNIZED spoken_as "not mine"`, `max_transactions_per_session spoken_as "up to {n} charges"` | how a value or a limit is spoken |
| `returns_field` | `get_review_transactions returns_field date` | the fields a tool's result carries |
| `asserted_by` / `denied_by` | `R27 denied_by "not, by itself, proof of fraud"` | how a rule's stance is asserted or denied in words |

The sources are `item.<field>` (a returned transaction), `write.<tool>.<arg>` (a successful write),
`writes.<tool>` (every successful write), `caller.<slot>` (what the customer said), `policy.<category>` /
`policy.limit:<key>` (the exact register) and `text.opening`. `PolicyKG.validate()` checks that every source
resolves and every dialog state is declared.

**Each claim gets a verdict:**

- `supported`: matches its source (a write must have truly succeeded, with the echo matching the request);
- `contradicted`: the source holds a different value;
- `unsupported`: no source backs it (for example "recorded" with no successful record for that charge, R13, or
  "done" before any card action succeeded, R17);
- `unverifiable`: nothing to check it against.

**Each turn also gets findings:**

- `text_contradicts_label`: the words say something other than the act's label;
- `label_not_said`: a labelled act whose required facts never appear;
- `missing`: required information absent, for example a charge presented without its date (pdf.2B), or a summary
  that leaves out a recorded answer (pdf.2F). A re-ask of a charge already presented does not need the facts again
  (pdf.3);
- `not_allowed_here`: an act that is not a graph transition from the prompt the customer answered.

Policy figures (fees, delivery days, limits) are graded on the exact register. Two readings are reported next to
each one, and neither decides the verdict: the GMS distance of the claimed value from the fact against the relation's
cap, and knowlytix's own `AnswerVerifier` (geometric mode), which confirms a correct figure and does not find a wrong
one.

**Scores per case** (in `results.csv`):

- `c_speech`: every turn's claims are supported, faithful to the labels and complete;
- `c_flow`: every act follows the graph;
- `outcome_correct`: the earlier verdict, kept as it was;
- `speech_claims`, `speech_supported`, `speech_contradicted`, `speech_unsupported`, `speech_missing` and the rest;
- `speech_problems`: the first few problems in words.

`--speech-gate` decides what `correct` means:

- `report`: speech is shown but not graded;
- `fail` (default): `correct` also needs `c_speech`;
- `strict`: `correct` also needs `c_flow`.

**Checking the checker.** A check that never fires proves nothing, so the campaign runs eleven negative controls
(`speech/controls.py`; skip them with `--no-speech-controls`). Each control edits only the words of one act in a real
conversation and leaves the labels, tools and outcome untouched:

- swap the merchant, change the amount, shift or drop the date;
- flip the recorded answer;
- a wrong fee, wrong delivery days or a wrong limit;
- "your card has been replaced" said early;
- "this means fraud";
- a summary that drops a recorded answer.

The report's section 5a shows the detection rate for each control, the claims by where their truth was read, and the
policy figures checked against the GMS (register verdict, GMS distance vs cap, knowlytix). `bot_validation.html` has
the full view by factor and response by response (notebook CELL 11d shows the same for the runs of CELL 11). Four word-only mutants
(`says_wrong_fee`, `misstates_recorded_answer`, `names_wrong_amount`, `summary_omits`) have the same outcome as the
reference bot and are killed only by this check.

```python
from gmsjourney.speech import SpeechContract, UtteranceVerifier
v = UtteranceVerifier(store, SpeechContract.from_graph(store.kg))
conv = v.conversation(session_date="2026-03-16")          # one per conversation
report = conv.check(acts, text, tool_log, customer_frames)  # after every bot turn
report.ok, [x.as_dict() for x in report.verdicts], [f.as_dict() for f in report.findings]
conv.summary()                                            # c_speech, c_flow and the counts
```

`JourneyRunner` does this on every bot turn, before the customer answers, when the store's graph has a speech
section (`check_speech=True`). With governance on, each turn's findings are also sealed live into the runtime
monitor's hash-chained audit as the gate `speech_grounding`:

- a false or baseless claim is sealed as DENY;
- a missing or unfaithful act is sealed as ESCALATE;
- flow is graded but not sealed.

The governance `SpeechMonitor` stays as it was. It is a guard for a few things that must never be said (a claim with
no success behind it, an asserted fraud stance, a figure off the register, a card number). The speech check reads
which charge and which value each sentence names and grounds every one of them. On the campaign, two seeded word
defects get past `SpeechMonitor` and are flagged live only by `speech_grounding`: saying the opposite of the
recorded answer, and a summary that leaves one out.

## How a base case and its enriched variants are proved to be the same test

Every customer turn has a meaning in the journey ontology: a **semantic frame** (dialog act + slots bound to
canonical individuals). Ground truth is a function of the frame sequence and the backend fixture only. For each
scenario: (1) ground truth identical; (2) every enriched turn read back to the same frame; (3) the oracle replays
the read-back frames and re-derives the same outcome; (4) geometric corroboration (surface moved, meaning did
not). Meaning-changing edits must all be rejected; they are. Evidence is exported as RDF (`ontology.ttl`).

## Two enrichment arms: template and LLM

Both arms realise the **same design rows** (same base, same factor levels), so their results can be compared pair by
pair. The template arm builds each customer turn from its meaning card with deterministic code. The LLM arm
(`gmsjourney/enrich/llm`, `--llm`) asks a model on the internal Microsoft (Azure OpenAI) gateway to write the turn
(default deployment `gpt-41`; `--llm-model gpt-55` for 5.5). It is authenticated with a short-lived `gcloud` access
token — `AEXP_ACCESS_TOKEN` and `AEXP_USE_CASE_ID` override the CLI and the routing header. Set the exact endpoint with
`GMSJOURNEY_LLM_URL` (full chat-completions URL; `{model}` is filled with the model name, e.g. `gpt-41`) and the
corporate CA bundle with `GMSJOURNEY_CA_BUNDLE` (else `REQUESTS_CA_BUNDLE`, else `/etc/ssl/certs/ca-certificates.crt`).
If the deployment rejects `temperature`/`max_tokens` or the system role, the client adapts once and records it
(`report.json: invariance_llm.clients.*.adaptations`). There is no other transport and no fallback. It is not trusted to construct the meaning, so it is validated separately and more strictly:

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

These additions follow from the gmsh comparison:

| addition | where |
|---|---|
| GEODE `dedup=False`, so the loop's retrains and anchor check see every fact; duplicate resolution runs audit-only on the trained store; anchor violations reported. `--kx-dedup-mode audit` keeps the 0.3 behaviour; `accept` is stock gmsh. | `knowledge/kx_store.py` |
| v-space fine-tuning: full (the reference default), rotation, or `ab` to train both and keep the better held-out noisy-alias binding (`--kx-v-sft-mode`) | `knowledge/kx_store.py` |
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

### An LLM bot as the system under test

`gmsjourney/sut/llm_bot.py` is a second SUT: a tool-using agent on the same model gateway and replay cache as
the LLM arm (cache tag `sut`). It is given the operating procedure (`policy/charge_review_policy.md`), the tool
API with its transfer reason codes, and the dialog-act inventory - nothing the oracle computes. Each step the
model returns one JSON object, a tool call (made on the harness backend, so faults, gates and the trace see it)
or the acts it says; a malformed reply is sent back once, then surfaces as an `UNKNOWN` turn (a bot defect,
never repaired by the harness). Notebook CELL 15b runs the full campaign with it:

```python
from gmsjourney.enrich.llm import LLMSettings
from gmsjourney.sut.llm_bot import make_llm_factory
bot = make_llm_factory(store, settings=LLMSettings(model="gpt-41", temperature=0.0, max_tokens=900))
run_campaign(CampaignConfig(out_dir="output/campaign_llm_sut", bot_factory=bot, bot_name="LLMBot", runs_per_base=2))
```

`max_cases=N` tests only N base cases (each with its enriched variants), picked in proportion to the base
categories; the default `None` tests all of them. Progress bars (tqdm) show each run phase with the number correct
so far (`progress=False` turns them off). The report's section 5b lists every failed run with the reasons it failed
(outcome, recorded responses, card action and stages expected vs. what the bot did, policy violations, wrong policy
answers) and its transcript, most severe first, filterable by arm and severity.

`case_comparison.html` (next to `report.html`) is the same campaign case by case, in the answer sheet's format: for
every run the outcome signature expected by the oracle beside what the bot produced, field by field (how the call
ended, transfer reason, responses recorded, card action completed, and the policy answer for policy-question cases -
the comparison that decides correct / incorrect), the stage, admissibility, invariance and budget checks reported
next to it, the reasons a run failed, and every customer turn with EXPECTED (oracle tools and acts) beside ACTUAL
(bot tools, acts and words). Incorrect runs are shown by default; buttons show the correct ones, all, or one arm.
It is built from `dataset/case_results.jsonl` and the turn-by-turn files, so an existing campaign folder can be
redone without re-running the bot: `python tools/case_comparison.py --campaign output/campaign_llm_sut`.

Each customer turn costs 1-4 gateway calls; replies are cached (temperature 0), so repeated runs of the same
conversation and `mode="replay"` cost nothing. `bot.stats` / `bot.llm.report()` give calls, cache hits,
malformed replies and tool errors. Tests: `python tests/test_llm_sut.py` (a scripted fake model, no network).

## Outputs (`output/campaign_out/`)

| file | content |
|---|---|
| `report.html` / `report.json` | the campaign report / every number in it |
| `bot_validation.html` | what the bot did and what it said. Every run is shown with two verdicts: did the right thing (the outcome) and said it right (every claim grounded). Runs are split by type of enrichment, base category, DOE factors and enrichment factors. Each claim shows where its truth was read and how it fared against the GMS. Response cards show the failing responses, each with its case's factor levels. Rebuild it from the files with `python -m gmsjourney.bot_validation output/campaign_out` |
| `case_comparison.html` | every case, expected vs actual, turn by turn, with the speech verdict |
| `dataset/case_results.jsonl` | every graded run, including `speech_turns`: one response card per bot turn with its claims, verdicts, truth, where it was read, and the GMS reading |
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
                     kx_view (GEODE input), canonicals (export), ingest + geode (the policy compiler: symbolic
                     facts, exact numbers, correction log) + store (the compiled symbolic store, no geometry)
  export_dataset.py  dataset/*.csv
  turn_compare.py    turn by turn: oracle's expected bot response vs the bot's actual response
  policy/            charge_review_kg.json (source), charge_review_policy.md (structured transcription of the PDF)
  reuse/             adapters: forgeloop plausibility gate + audit chain, proofloop emitters/taxonomy
  catalogs/          application.yaml (+ level_aliases), custom_factors.yaml, official/ (pinned copy of proofloop's)
  governance/        gates (policy, speech, whole-plan), monitor (strict / quorum; forgeloop per-step gate + audit),
                     attestor (numeric range), calibrate (Appendix C + P=0.5, holdout, bootstrap)
  enrich/llm/        the LLM enrichment arm: client (Microsoft gateway, replay cache), prompts, gates,
                     realizer, extractor (independent reader), validate (separate proof + controls), export
  enrich/            realizers, reader, invariance proof, noise, textsim (the reported surface-similarity metric)
  evaluation/        runner, attribution, stats, constraints (budgets), severity (tiers + release gate), ...
  ontology.py journey.py oracle.py kb_truth.py bases/ enrich/ design/ sut/ evaluation/ pipeline.py report.py
run_campaign.py · tests/ · notebooks/ · DESIGN.md · NOTICE
tools/             check_llm (one-call transport preflight), llm_rejections (why LLM turns were rejected),
                   answer_sheet (the case answer key), build_annotated (the core-file listing), build_docs,
                   case_comparison, graph_showcase, geode_before_after
docs/              generated walkthroughs: EXPLAINER, INVARIANCE, ANSWER_SHEET, ANNOTATED
output/            generated campaign runs (report, dataset/, corpora) — gitignored
fictional_poc_operating_procedure_gpt_readable.pdf (source document, not published)
```

The package, the notebook and the tools are the source; `docs/` and `output/` hold everything generated,
so a clean checkout is `gmsjourney/ notebooks/ tests/ tools/ *.py *.md *.txt *.toml` — regenerate the rest
with `run_campaign.py` and the two tools.

Licensing: knowlytix-core is used under the Knowlytix EULA and is not included in this package; the other
knowlytix packages, proofloop and forgeloop are Apache-2.0. See `NOTICE`.
