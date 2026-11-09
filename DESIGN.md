# Design notes

## 1. What was wrong with the original notebook

The baseline (`deepseek_python_20261010_d9a1ca.py`) has the right outline but could not produce a valid
result. In order of consequence:

| # | Problem | Consequence | Fix here |
|---|---|---|---|
| 1 | Enrichment is never realised: `evaluate.run` sends `scn.base.query` to the bot, so all scenarios of a base send identical text. | Factor attribution has no signal by construction; the design is decoration. | Realizers generate every turn from its frame under the scenario's levels (`s05_rewording/wording.py`); the runner sends the realised text. |
| 2 | The SUT is one shared stateful object, never reset, driven one message per scenario. | After 20 calls every answer is `conversation_ended`; results depend on scenario order. | A fresh bot + backend per scenario; a reactive user simulator plays the whole multi-turn journey (`s06_bots/harness.py`, `s04_cases/journey.py`). |
| 3 | The bases are knowledge questions ("What is the maximum number of charges?") sent to a bot whose job is a journey. | The bot is graded on a task it does not perform. | Journey bases (multi-turn content) are the core; policy questions are asked *inside* journeys, as the PDF allows. |
| 4 | Ground truth is invented (`charge_c > charge_a`, `145.67`, `230.45`, "3 verified"). | Violates "generated, not annotated" (Ch. 1.2, 12.4); a failure cannot be blamed on the bot. | Every answer is computed: policy questions by store primitives, journeys by an oracle reading limits from exact memory. |
| 5 | Factors that change the correct answer are used as factors (`tool_availability`, `clarity=Ambiguous`, turn-adding history, `intent_count=conflicting`, `unit_variation=EUR`). | Breaks the invariance contract; scores are wrong for those rows. | Moved to the base (`application.yaml: moved_to_base`) or excluded with a reason; invariance is proved per scenario. |
| 6 | Wrong / non-existent APIs: `gmstest` (the real module is `knowlytix.harness.suite`), `lookup_enn`, `result.n_enn`, `scenarios['base_id']`, `scn.seed`, `scn.ground_truth`, `functionality_check(seed)`, `ToolGateway(sut, profile)`. | The notebook cannot run. | The knowlytix family, proofloop and forgeloop are called through their real public APIs (§10); our modules keep book-compatible names (`lookup_enm`, `score_triple`, `QAItem`, `Scenario`, `SUTResult`, `resolve`, `compose`, `attribute`, `weak_link`). |
| 7 | `attribute(...)` receives `[f for f in suite.factor_names if f in rows[0]]`, but row keys are `f_<factor>`. | Empty list: no attribution is ever computed. | Attribution reads `f_<factor>` columns. |
| 8 | Workflow lists tools the PDF does not have (`search_policy`, `check_limit`, `mark_verified`); components are keyed `authentication`, `verification`. | `weak_link` never finds a `c_<tool>` key and returns `{}`. | Components are per workflow stage, derived from the observed tool log, identical for oracle and bot. |
| 9 | The policy markdown diverges from the PDF: "dispute → escalate" (the PDF says record UNRECOGNIZED; it is not proof of fraud), no dispositions, no SUCCESS-gated claims, no card intent, no 30-day window, no voice rules. | Tests would enforce behaviour the business never specified. | The policy document is a structured transcription of the PDF plus your two program limits, each row with provenance. |
| 10 | The grounding check compares `parser.parse(seed) != parser.parse(enriched)` with an undefined parser and has no negative control. | Nothing is proved; a checker that accepts everything would pass. | Four-layer proof + negative controls with measured rejection rates. |
| 11 | Resilience counts `status in ("escalated","failed")`, but the SUT returns `terminated`/`refused`; the gateway has no executor to wrap. | Detection rate is always 0. | Faults are injected at the mock backend; detection is judged against the oracle under the same fault. |
| 12 | Calibration is run and discarded; hand thresholds elsewhere. | Unknown false-accept rates. | Every gate reads a persisted operating point. Store gates are fit by `GMSJudge.calibrate` (knowlytix); runtime gates use the Appendix C regulated point (zero false admits, then fewest false rejects) with recalibration triggers (§11). |
| 13 | `from google.colab import files` at import time. | Breaks outside Colab. | Removed. |

## 2. The oracle (book Ch. 2-4)

The store is knowlytix's (§10). This section covers what is ours and runs before it: the policy compiler,
the exact register the trained store is checked against, and the policy oracle.

*Ingest* parses the structured policy into candidate triples and exact numeric entries, each with a source
span. A deliberately naive prose path pairs numbers in knowledge-base sentences with the nearest entity.

*Self-correction* (symbolic) diagnoses every candidate: schema/type critics, reconciliation of each prose
figure against the exact register (corroborate or drop) and unbound subjects. A prose fact with no register
counterpart is kept and flagged; the geometric checks are knowlytix's GEODE loop, which runs next on the
compiled facts. All 12 prose figures in the policy corroborate; an injected OCR misread (expedited fee $150)
is dropped (`injected_misread_caught: true`).

*The compiled store* (`s03_knowledge/store.py`) holds the symbolic layer: **Exact Numeric Memory** (every
authoritative figure keyed by (category, key) with a SHA-256 integrity hash, read by `lookup_enm` and never
parsed from text), the typed triples with provenance (530, the dialog-state layer included), the rules, the
workflow, value sets, synonyms and the knowledge-base sentences. It carries no geometry: `KXPolicyStore`
(`s03_knowledge/build.py`) subclasses it and answers every geometric read (score_triple, cap radii, tension,
holonomy, the tuned encoders) from the trained knowlytix store, and every exact read from knowlytix's own
register, cross-checked against the compiled one.

*Like for like with knowlytix.* The store is built with the parameters of knowlytix's reference pipeline
(`build_calibrated_rag_store`): `GeodeLoop(max_iters=8, residual_threshold=None, canonicalize=False)` on a
300-epoch trainer, `GeodeEmbedLoop` with a 150-epoch trainer and `EmbedLoopConfig(max_iters=2, use_geometry=True)`,
`EmbeddingSFTConfig(rank=8, mode="full", out_dim=d_v)`, `contradiction_sft(epochs=400)`, then `store_from_triples`.
The reference takes the production store's `DocGMSConfig` from its caller (and defaults `ingest_mode` to
`"regex"`); ours is `loss_mode="cap"` (the cap loss is what makes `cap_radius` the admissibility cut),
`ingest_mode="regex"`, `train.epochs=150`. The stated differences are the encoders passed by local path (the same models), dedup (audit-only by default) and the u
training sentences (our attribute phrasings). Every one of these is a run-time setting.

At run time each geometric read is the knowlytix method itself: `score_triple` (a tail the store never trained as
an entity is scored with `score_triple_emb` on the tuned v encoder's embedding, π/2 only when the head or relation
is unknown), `cap_radius`, `tension_energy`, `check_holonomy`, and `is_path_consistent`, which reads
`config.verify.tau_path`; the store ships without one, so the calibrated cut is written there, the way GMSJudge
writes its own. A claim is grounded when `score_triple ≤ cap_radius`, the test the store is trained for and the
one knowlytix's benchmark generators apply. knowlytix's per-head calibrated admissibility
(`calibrate_cap_margins_per_head`, `calibrate_tension_threshold`, `admissible`, 5% FPR) is reported next to it in
the oracle check but does not decide: on this policy the delivery relations have two heads, the per-head margins
(0.74-0.88) come out far wider than the cap radius (0.31) and admit 3 of 12 cross-tail substitutions (for example
a $15 fee on standard delivery). The tuned-encoder entity binding and the rule-stance check have no knowlytix
equivalent and stay ours.

The *policy oracle* is an executable reading of the policy that consumes semantic frames (never text), reads
all limits from exact memory and records which rules fire and which workflow edges are traversed.

*The oracle is the graph's flow, walked.* The dialog-state machine and everything the oracle does on each
transition is in the policy graph: a `leads_to` triple's qualifiers say on which customer act it is taken (`on`;
`auto`, `else` and `any` for the bot's own moves, an unhandled act, and a turn-limit pre-emption), the condition
(`when`, an expression over the call so far: `call('record_charge_response')`, `tail(get('delivery'),
'requires_fee_confirmation') == 'yes'`, `turns() > limit('max_turns')`), what it does (`do`: `set`, `inc`,
`load`, `fire`, `end`), what the bot says on the way (`say`), the rules it exercises (`rules`), and the
procedure text it implements. `carries` and `has_arg` say what a bot act carries and what a tool is called
with, as expressions of the same language. The language is a vetted subset of Python expressions (names,
literals, comparisons, boolean and arithmetic operators, calls to a fixed function table), compiled once; no
attribute access, subscripts or builtins. `s02_graph/flow.py` reads it (`FlowSpec.from_kg`, validated: every state,
function and act it names must exist) and `GraphOracle` walks it against the mock bank and the simulated
customer. It replaced a hand-written oracle with identical results on 86 base cases, 1500 random cases, 2064
faulted and 1092 replayed runs (`tests/test_flow_parity.py`, against the old oracle kept in `tests/`), and a
mutation of a guard in the graph changes the walk.

*Journey factors are data.* Each pack's `project.yaml` holds the mock bank, the customer's scripts and the factors.
A level patches the bank (`auth_results: [FAILED, AUTHENTICATED]`) and the customer (an act said at a state
before the scripted answer, once or every time; a different answer; an extra slot), so a new scenario needs no
oracle code: the graph's flow decides what the bot should do with that bank and that customer. The project is
checked against the graph (every act, slot, value, status, state, trigger and expression), and the old
hand-written factor code is kept in `tests/` for the parity check (`tests/test_scenario_parity.py`: relevance and
validity on every one of 1.2M level combinations, fixtures, categories and tags on 200k, the customer on 3000
fuzzed runs, oracle walks and the generator's covering array and closure).

*Names come from the graph.* `s01_packs/names.py` collects every name the graph defines; core code spells them through
it (`N.state.PRESENT_TXN`) and finds tools by role (`s01_packs/names.py`), so a renamed concept fails where it is used.
Severities, release-gate profiles, speech-check leniencies and the speech reader's cue words are project
settings, edited in the studio (`python -m gmsjourney.s10_console.studio`).

## 3. Base cases

**Policy questions (18-category taxonomy).** One generator per category, answers computed by the primitive
named in the base catalog, distractors and counterfactuals drawn from the store (multiple-choice options are
numeric neighbours of the register value; false plausibility claims come from `negatives_for`; window checks
straddle the lookback limit). Each passes the functionality check (symbolic re-derivation, plus geometric
recovery where one fact triple carries the answer): 17 doubly grounded, 10 symbolic-only, on either backend. Contradiction and
reasoning-path consistency are not customer-facing and run as store self-tests. The precondition of Ch. 4.3
holds: the graph answers its own questions at 100%.

**Journeys — why "optimal".** Thirteen content dimensions (authentication path, retrieval, transaction count,
30-day window, disposition pattern, record result, card intent, card options, expedited confirmation, submit
result, human request, mid-journey policy question, turn pressure) give millions of combinations. Three ideas
make a small set sufficient:

1. *Relevance*: dimensions that cannot affect the ground truth of a case collapse to N/A (after a failed
   authentication nothing downstream matters), and a level pair counts as covered only when both are relevant.
2. *Covering array*: AETG-style greedy construction covers all 523 feasible level pairs with 50 journeys.
3. *Closure*: the oracle reports rules fired, edges traversed and transfer reasons; journeys are added until
   every rule, edge, trigger and boundary is exercised (4 more). Turn-limit cases are padded with policy
   questions until the oracle reports exactly 20 turns (must complete) or 21 (must transfer).

Mutation analysis confirms the value: the covering suite catches 75% of seeded defects, a random suite of the
same size 37.5%.

## 4. Enrichment factors

The factor catalog keeps the book's 40 domain-neutral factors plus two voice factors (`disfluency`,
`self_correction`, per Ch. 6.3.2). The application layer selects 16, overrides level vocabularies for a voice
channel (`noise: [none, typo, asr]`, `clarity: [clear, indirect, downplayed]`, `numeric_format: [digits, spoken]`)
and records the other 26 as excluded or moved to the base, each with a reason. Two decisions matter most:

* `clarity=Ambiguous` and `tool_availability` change the correct behaviour of this bot (R22/R23 require a
  clarification; a failed tool requires a transfer), so they live in the base as content dimensions.
* `history` is realised as an in-turn preamble, never as an extra turn, because an extra turn changes the
  turn count and can flip the 20-turn outcome. The negative control `insert_turn` shows the validator catching
  exactly that.

Realizers build each turn from its frame in three layers (core decision, meaning-neutral wrappers, surface
noise) and protect decision-critical tokens (negations, yes/no, digits) from noise.

## 5. The invariance proof

**Claim.** For every enriched scenario *s′* of base *s*: the expected outcome and per-stage expected
behaviour of *s′* equal those of *s*.

**Argument.** The oracle's output is a function of (frame sequence, fixture) only. The fixture is shared by
construction. Layer 2 shows the frame sequence read back from *s′*'s text equals *s*'s frame sequence, turn by
turn, under ontological canonicalisation. Layer 3 closes the loop empirically by replaying the read-back frames
through the oracle and comparing outcome and components. Layer 1 rules out drift between carried and recomputed
truth.

**Instrument validity.** The reader is independent of the realizer: value words bind through the store's
alias edges, typos are repaired by edit distance to the ontology vocabulary (never repairing valid English
words), and it implements policy rules (latest clear answer, reported speech) rather than realizer templates.
Its sensitivity is measured, not assumed: 592 meaning-changing edits across 17 kinds (frame-level, raw-text
and sequence-level), 100% rejected. Raw-text edits are generated only where they are unambiguous (exactly one
cue of the edited kind, on a turn type where the edit changes meaning). A 10,500-sample round-trip stress test
over random factor combinations recovered every frame.

**What the proof does not cover.** It certifies the customer's *meaning*; whether a given bot understands it is
what the test measures. A turn the reader cannot recover is re-seeded (up to 4 times) or the scenario is
rejected and excluded, as GEODE discards questions it cannot vouch for (0 needed in this run). For extra
assurance, an LLM adjudicator can be added as a second reader; agreement would then be required in layer 2.

## 6. Design of experiments (Ch. 7, Appendix C)

φp is minimised exactly as in Appendix C (sigmoid reparametrisation, log-space evaluation, analytic gradient,
p-annealing, Sobol initialisation, cardinality weights). The continuous optimum piles points onto the cube's
faces, which starves middle levels after binning (marginal χ² p ≈ 0, pairwise level coverage 54% in a first
attempt), the geometric-versus-marginal tension the book discusses. The fix keeps the optimiser's arrangement
but rank-transforms each column (Latin-hypercube marginals) and refines with within-column level swaps, which
lower φp while preserving marginal counts exactly. Result at 486 runs: log φp 0.263 (random 0.670, plain Sobol
0.684), minimum distance 0.83 vs 0.53, perfect marginal balance, full pairwise level coverage.

The base enters as a balanced blocking factor. Rows are dealt to bases by a maximin rule along a
farthest-point order so each base sees dissimilar conditions (mean within-base minimum distance 1.40 vs 1.11
for a random shuffle). A Monte Carlo power curve (App. C.6) sets the budget: 6 runs per base gives 0.93 power
to detect a level that triples the odds of failure.

## 7. Evaluation

* **Outcome**: end state, transfer reason, what was truly recorded (SUCCESS with matching echo), which card
  action truly completed. For a policy-question base, the graded answer is the outcome.
* **Components**: authenticate, retrieve, present, kb, capture, record, card_action, transfer, summary — read
  from the observed tool log and bot acts, identical for oracle and bot.
* **Trajectory**: every consecutive tool pair must be a workflow edge of the store; gate checks for
  authentication before account tools (R2/R3), attempt limits (R4), transaction limit (R31), claims only after
  SUCCESS (R13/R17), confirmation before submit (R16), fee disclosure before expedited (R26), turn limit (R28).
* **Attribution**: per-factor likelihood-ratio deviance on L−1 df, Benjamini–Hochberg, McFadden pseudo-R²,
  Wilson intervals, odds ratios, joint model (AIC/BIC/AUC/Hosmer–Lemeshow), interactions among significant
  factors, per-category and per-component attribution, weak link.
* **Resilience**: error / timeout / stale / plausible-wrong per tool. Silence is counted only where the oracle
  itself escalates under the same fault; faults nothing at the bot boundary can detect are reported separately.
* **What the bot said** (§15): every sentence's claims are graded against the policy graph, the exact register and
  the session's true writes. `c_speech` is folded into `correct` by default (`--speech-gate`).
* **Mutation analysis**: 21 realistic defects. Four of them change only the words. A scenario kills a mutant when
  the reference passes and the mutant fails.

## 8. Reading of the operating procedure (assumptions to confirm)

| Topic | PDF / your brief | Implemented as |
|---|---|---|
| Authentication attempts | PDF: transfer on failure. Brief: at most 2 incorrect attempts. | Two attempts; after the 2nd failure, transfer (`auth_failed`). The baseline's "terminate" was replaced by the PDF's transfer. |
| Turn limit | Brief: max 20 turns. | At most 20 customer turns. If servicing is complete at the limit, summarise and end; otherwise summarise and transfer (`turn_limit`). |
| More than 4 transactions | PDF: up to 4 per session. | Review the first 4, explain the limit, transfer for the rest (`transaction_limit`). |
| 30-day window | PDF KB rule; tools "read only returned data". | A returned transaction older than 30 days is not presented; day 30 is included, day 31 excluded. |
| Card options tool failure | PDF §7: if a required tool fails, transfer. | Transfer (`card_options_unavailable`, R32). |
| Echoed SUCCESS that does not match the request | Implied by "advance only after SUCCESS". | Treated as failure (R33); exercised by plausible-wrong faults. |
| Disputed / unrecognized charge | PDF: record UNRECOGNIZED; not proof of fraud. | Recorded, no escalation by itself (the baseline's "dispute → escalate" removed); a fraud claim is a contradicted stance. |
| Refund amounts | Not stated. | Absence detection: the bot must not invent one. |
| Opening statement | pdf.2A: play the required opening; pdf.6: "Before I review account-specific charges, I need to authenticate you." as the required/static example. | That sentence is the required opening, said word for word (`basis: interpreted`). |
| "Overnight" delivery | pdf.5: expedited arrives in 2 business days. | Not a synonym of expedited (removed from the aliases, the customer reader and the LLM arm's word lists). |

Change any of these in `policy/charge_review_policy.md` (and, for flow changes, `s02_graph/flow.py`); every expected
outcome follows.

## 9. Limitations and extensions

* The reference bot stands in for your bot. Its numbers describe it, not your system.
* The results shipped with this package were produced where the HuggingFace hub was unreachable, so knowlytix
  was warm-started from the offline hashing stand-in (§10). The embedding figures (alias binding 68%, routing
  87%, surface cosine) are therefore lower bounds; with the real encoders they should rise. Nothing in the
  invariance proof, the exact register or the oracle depends on them.
* The reader is rule-based and domain-specific by design (that is what makes it auditable). Extending the
  journey means extending the ontology, realizer templates and reader cues together, then re-running the
  round-trip and negative-control checks.
* Speech is simulated at the text level. To test the ASR stack itself, route realised text through TTS and your
  recogniser; the proof still applies to the text the customer *intended*.

## 10. The knowlytix backend (what is reused, and the seams)

**Rule followed.** Everything the knowlytix family, proofloop and forgeloop provide is called through their
public APIs; `gmsjourney` implements only what they do not cover for a multi-turn voice journey. knowlytix-core
is a compiled package under the Knowlytix EULA: it is installed from PyPI, never decompiled, modified or
redistributed, and neither its code nor a trained store is shipped in this package. Its runtime licence is read
from `KNOWLYTIX_LICENSE_KEY` (or `~/.knowlytix/license.key`); no file here contains a key and nothing logs it.

**Build (`s03_knowledge/build.py: build_kx_store`).** Eleven steps, each a knowlytix call except where noted:

1. compile the policy (ours: the structured transcription of the PDF, every fact with its source span);
2. render the GMS view (`s03_knowledge/store.py`, ours), described below;
3. resolve encoders (HF sentence encoders, or the offline stand-in);
4. `GeodeLoop` with `make_default_trainer`: ingest, propose, train, diagnose, correct until converged;
5. `GeodeEmbedLoop` with `EmbeddingSFTConfig(mode="rotation")` on v, supervised by the alias table; the
   fine-tuned v vectors are stored for Mode B (`v_vectors_path`);
6. `contradiction_sft` (full mode) on u, warm-started from the base u encoder;
7. `calibrate_relevance_thresholds`;
8. `store_from_triples` on `fact_only_triples` with cap loss, giving a `GMSExpertStore`;
9. `coverage_report` / `graph_coverage`;
10. `GMSJudge.calibrate` and save (`calibration.json`);
11. a build report (ours) with every diagnostic, timing and the ENM agreement table.

A store is cached by the hash of the policy and settings, so later campaigns reload it in seconds.

**The GMS view.** knowlytix's markdown ingest turns each wide table into typed triples `(row, has_<column>, cell)`
and each numeric cell into an integrity-hashed ENM entry. The source policy mixes bullet lists, a
head/relation/tail table and multi-alias cells, which that parser reifies or flattens. The view re-renders the
same compiled facts as one table per relation. Three findings from fitting it to the parser:

* numeric tables must carry no other text column, or the text folds into the ENM key;
* yes/no columns are read as pass/fail flags, so fee confirmation is written `required` / `not_required`;
* alias and opposite tables become `has_alias` / `has_opposite_of` edges, which are the supervision for the
  embed loop and the tension calibration.

The ENM ids are `limits/<key>/value` and `delivery_options/<option>/<column>`. All 12 agree with the compiled
register byte for byte; the build fails loudly if one does not.

**Dedup audit.** GEODE's duplicate critic removes a (head, relation) edge whose tail is a cap outlier among the
head's tails. For `enables` that is wrong by construction: a step legitimately enables several next steps, and
3 true workflow edges were proposed for removal (`start → transfer_to_human`,
`authenticate_customer → authenticate_customer` (the retry), `authenticate_customer → get_review_transactions`).
Each is stated verbatim in a source table, so it is reinstated and recorded in the build report
(`duplicates_rejected`, with location and cap distance). Nothing else was corrected; GEODE converged in 3
iterations.

**Calibration view.** The framework reads knowlytix's operating points, not its own:

* per relation, `cap_radius` (e.g. 0.527 for `has_value`), with realised true/false accept rates on the store's
  own triples and counterfactual tails. `enables` admits 18 of 21 true edges inside its cap. The 3 outside are
  the ones the dedup critic flagged, which is why the runtime gate is calibrated separately (§11);
* holonomy: the judge's cut (6.86) is global over every relation triangle. The self-test asks a narrower
  question (does a declared composition close?), and the declared compositions sit above that global cut. So
  the self-test cut is fitted on the declared-composition cohort (7.21), and the judge's cut is reported
  alongside;
* tension: the judge reported its tension calibration as degenerate on this small graph. The cut is therefore
  taken from the store's own same/opposite cohort (0.444, between 0.0005 and 0.887).

**Two-way admission** of policy-question bases calls knowlytix's own generator gate,
`ExactRecallGenerator(geometric=True)._validate`. A question is admitted two-way only when:

* the symbolic answer matches;
* the true tail ranks first among decoys;
* the true tail sits inside the relation's cap radius.

A forged triple is rejected. The knowlytix substrate generators run on the store as a precondition check: the
graph answers its own questions at 100%. `cross_reference` is skipped by default because it costs
O(questions × decoys) geometric validations; `kx_substrate(full=True)` runs it.

**Performance.** `score_triple`, `cap_radius`, `tension_energy` and `check_holonomy` are pure functions of a
frozen store, so `CachedKX` memoises them. The reader's attribute classifier is memoised too. With both, a
campaign over a cached store runs in about 2.5 minutes on CPU.

**Offline encoders.** The hub was unreachable where the shipped results were produced. The stand-in is an
HF-format WordPiece tokenizer plus a one-layer BERT. Its word embeddings are signed hashes of sub-words, and its
attention and FFN outputs are zeroed, so with mean pooling it is a bag-of-sub-words hashing encoder. Every
knowlytix step runs on it unchanged. Held-out gains are still real:

* v alias binding under typo/ASR noise: 25% → 68%;
* u phrasing → relation routing: 81% → 87%.

They start from a semantically empty prior, though, so the absolute figures are lower bounds.

**Design.** The deployed design is ours. knowlytix `DesignMatrix` sobol+refine was compared on the same 16 factors
and starved the middle level of every 3-level factor (11 of 43 level cells empty, 55% pairwise level coverage,
marginal χ² p ≈ 1e-53: the face-piling effect of §6), so it is no longer run.

**Attribution** is ours: per-factor deviance tests with Benjamini–Hochberg, a joint model, interactions and the
failure table. (knowlytix `DOEAnalyzer` ran alongside earlier and found the same drivers - noise, numeric_format
and disfluency; it is no longer run.)

## 11. Runtime governance (book Appendix C)

The test campaign also certifies a runtime monitor that runs with the bot in production.

**Gates.** Gates run in order, and the first DENY wins:

* **syntax**: known tool, required arguments;
* **policy**: R2/R3 authentication before account tools, R4 attempt limit, R7/R24/R30/R31 transaction limits,
  R11 UNSURE handling, R15 card action among the returned options, R16/R26 confirmation and fee disclosure, R20/R21/R22 transfer
  and clarification, R28 turn limit. These are compiled from the policy rules (ours);
* **GMS plausibility**: forgeloop's `GMSPlausibilityGate` over the knowlytix store scores each proposed
  transition `(previous tool, enables, proposed tool)`. A safe tool (`transfer_to_human`) is escalated, never
  denied, so the bot can always hand over;
* **plan gate**: the same scores aggregated over the whole trajectory.

**Speech.** Each bot utterance is checked by `SpeechMonitor` for:

* claims of completion without SUCCESS (R13/R17);
* fraud stance (R27);
* PII read-back;
* every figure spoken against the ENM, with the delivery option bound from the last knowledge-base frame.

**Completeness at end of call.** The monitor checks that no reviewed transaction was left unrecorded
(R12/R18) and that an UNSURE without a transfer is caught (R11). Every decision goes to forgeloop's
`AuditLogger` hash chain, and tampering is detected.

**Calibration.** The calibration cohort is the 21 policy `enables` edges against the 43 transitions the policy
does not allow, scored by the store. Appendix C's regulated operating point is used: zero false admits, then
fewest false rejects. That gives θ = 0.824 with AUC 1.0 and no errors.

The plan gate compares a mean-drift criterion with a max-step criterion on 24 admissible against 188
inadmissible plans. Max-step separates (AUC 1.0); mean does not (one false reject), so max-step is selected.

A multi-threshold balanced design varies four thresholds jointly and counts errors per gate. The probe sweep
replays every (previous, proposed) pair: 15 true allows and 27 true denies, with no errors.

**Recalibration triggers.** The calibration record persists:

* the store and policy fingerprints;
* reference scores taken from base-level correct runs.

The campaign then checks:

* PSI of runtime scores against those reference scores (0.04, limit 0.2);
* false-reject creep;
* store retraining;
* policy change.

None fired.

**Results.**

* 0 false alarms on fully correct runs (correct outcome, adherent workflow, correct knowledge-base answer);
* 103 of 328 failed runs flagged live;
* all 16 seeded defects flagged live.

The regex injection guard from forgeloop flags none of the paraphrased authority-override turns. It is
reported as a baseline, not relied on.

## 12. The policy graph and state-aware meaning cards

**One machine-readable source.** `policy/charge_review_kg.json` holds the operating procedure as head / relation /
tail triples, and every later step reads it through `PolicyKG`:

- GMS training (both backends);
- the oracle's limits;
- the ontology mapper;
- the exports.

It is built by `s02_graph/triples.py` from two inputs:

- the PDF, which supplies the evidence: section split, tools table read by column, a verbatim quote per triple;
- the structured transcription, which supplies the facts.

The dialog-state layer is written in the same module from PDF sections 2, 3 and 7.

Every triple carries:

- `kind`: trajectory, state, policy, vocabulary or domain;
- `source`: a PDF section, `program` or `lexicon`;
- `evidence`;
- optional `qualifiers`, such as a transition's customer act, condition, rules and workflow edges;
- `gms`: whether the store is trained on it.

`validate()` checks the following:

- every evidence quote is a verbatim substring of its source;
- every state is reachable from OPENING;
- every rule applies in some state;
- every transition and accepted act refers to a declared state and act;
- the content hash matches.

The compiler still ingests the structured form. `PolicyKG.to_markdown()` renders it from the JSON, and a self-check
proves that the JSON and the original transcription compile to identical facts, value sets, aliases and entity types.

**The state layer.** The layer has 19 dialog states; each is the bot act a customer turn answers. 7 of them prompt
the customer. The other 12 are bot-only acts such as CONFIRM_RECORDED, SUMMARIZE or TRANSFER.

For each state the graph records:

- its stage;
- the meaning cards it accepts (REQUEST_HUMAN, ASK_KB and UNCLEAR everywhere, by R20, R25 and R22);
- how it reads a bare yes or no;
- the rules and limits that apply there;
- the tools the bot may call next;
- the escalation triggers that can fire there.

The 64 `leads_to` transitions each name the customer act or `auto`, the condition, the rules and the workflow
edges. A self-check runs the covering-array journeys through the oracle and requires two things:

- every customer turn is admissible in its state;
- every consecutive pair of bot acts is a declared transition.

So the graph and the executable oracle cannot drift apart silently.

The GMS is trained on 10 state relations as well as the compiled policy:

- `in_stage`, `accepts_act`, `leads_to`, `applies_rule`, `limited_by`, `calls_tool`;
- `interprets_yes_as`, `interprets_no_as`;
- `fires_in_state`, `has_slot`.

The geometry therefore also scores state plausibility, for example whether PRESENT_TXN accepts PROVIDE_CREDENTIAL;
it does not.

**State-aware mapping.** A meaning card is canonicalised in its state (`Ontology.canonicalize(frame, state)`):

- slot values resolve inside the value set of their slot;
- a bare CONFIRM or DENY becomes what the state's `interprets_yes_as` / `interprets_no_as` says;
- `FrameReader.read(text, state)` reads a 4-digit number as a credential first only where the state accepts
  PROVIDE_CREDENTIAL.

A card is admissible when three things hold:

- the state accepts its act;
- its slots are declared for the act;
- its values belong to their slots' value sets.

Equivalence, as used by the proof, the runtime check, the classifier-corpus filter and the inspector, means the same
canonical card in the same state, admissible there. The proof's signature hashes (state, card) pairs.

The negative controls gain a `wrong_state` edit: the same words answered in a state that does not accept that
meaning. It must be rejected, and it is.

**Exports.**
- `s03_knowledge/store.py` writes `gms_canonicals.json` from the trained store: every canonical value with its
  aliases and the trained GMS's binding of each alias (agreement reported), membership score against the cut, tension
  to its disjoint siblings, exact numbers with digests, states, acts, and every trained fact with its score and cut.
- `s08_campaign/outputs.py` writes the test suite as CSV: one row per case and one row per customer turn, with the state
  policy, cards, texts, the expected outcome and the complete expected trajectory.

## 13. The LLM enrichment arm and its separate proof

**Why it exists.** gmsh can rephrase test questions with an LLM; the template realiser cannot produce the open-ended
variety of real callers. The arm adds that variety without weakening the guarantee that an enriched case is the same
test as its base.

**Why it is proved separately.** The template realiser constructs text *from* a meaning card, so its proof checks a
construction. The LLM is an untrusted writer: it can drop a value, add a figure, answer its own question, hedge a
decision into ambiguity, or leak internal names. So every LLM utterance passes per-turn gates before the shared
scenario-level test (`InvarianceValidator._conclude`: L1 truth identity, L3 oracle replay, signatures):

| gate | on | rejects |
|---|---|---|
| G0 form | raw LLM text | empty, too long, speaker labels, stage directions, several lines, tool / act names, rule ids, tool statuses |
| G1 facts | raw LLM text | a figure or merchant not licensed by the base wording, the template wording at the same levels, or the card; a dropped decision value |
| G3 one clear decision | final text | more disposition or delivery cues than the requested presentation allows (only a self-correction may carry two) |
| G2 meaning | final text | the rule reader's card in the dialog state differs from the base card or is not admissible there |
| G2' dual read | final text | (`dual_read=gate`) an independent LLM reader, blind to the base, recovers a different card |

**Blind generation.** The prompt holds only:

- the generic question for the state;
- the meaning in plain words;
- the base wording;
- the factor instructions (wrapper factors as content to paraphrase);
- the words the rule reader understands.

The expected outcome, the oracle and the policy figures are never in it. The independent reader sees only the
utterance and the question.

**Reader coverage is the limiting factor, never a weakness.** An LLM turn the rule reader cannot read is rejected (a
false reject) and regenerated with feedback. The acceptance rate therefore measures how well the reader covers what
the LLM writes; it can never let a meaning change through.

**Instrument power on LLM text.**

- *Text edits.* The deterministic meaning-changing edits (negation, digit, figure, delivery, card action, merchant,
  bare yes/no) and the wrong-state control are applied to the accepted LLM utterances. They go through the same
  `_accept_turn` the proof uses.
- *Adversarial rewrites.* The LLM is asked to change exactly one meaning element. An accepted rewrite is either a gate
  miss or a rewrite that kept the meaning. The audit list lets a person decide, and the rejection rate is reported both
  over all rewrites and over those the reader reads as the target meaning.

**Attribution on realised levels.** A factor counts as realised when the check detects it on at least half of the
turns it applies to. Unrealised factors are attributed at the base level, so a failure is never credited to a factor
the bot never received (a defect in gmsh's suite path).

**Reproducibility.** Completions are cached under the hash of (prompt version, model, temperature, messages, salt).
The salt is the scenario seed, so a re-run with `--llm-mode replay` uses no network. Proved utterances are pinned for
the run. A turn the proof never saw is generated live under the same gates, or falls back to the template arm and is
counted.

## 14. gmsh parity (0.4)

- **GEODE dedup (`kx_dedup_mode`).** 0.3 let the loop remove duplicate tails and reinstated those stated verbatim
  afterwards, so the loop's last retrains and its anchor check ran on a reduced set. `off` (the default) runs
  `GeodeLoop(dedup=False)`, then runs `resolve_duplicates` audit-only on the trained production store. `audit` keeps
  0.3; `accept` is stock gmsh. Anchor violations are now counted in the report.
- **v-space SFT (`kx_v_sft_mode`).** The default is `full`, as in knowlytix's reference build; the book uses rotation.
  `ab` trains both and keeps the better held-out noisy-alias binding (a tie keeps rotation). Both scores and the choice are in the build report. The
  u vectors are still not loaded into the GMS, as in gmsh.
- **Operating points.** Each gate cohort also gets:
  - gmsh's P=0.5 logistic point (L2, like sklearn's default), next to the regulated zero-false-admit point;
  - stratified k-fold held-out accuracy for each rule;
  - a 200-resample bootstrap interval for each threshold.

  Every relation's cut gets a bootstrap interval for a re-fitted cut and for the realised TAR/FAR at the deployed cut.
- **Quorum.** gmsh's `check_with_quorum` is computed for every call next to the strict rule:
  - a syntax error, or two or more blocking layers, denies;
  - exactly one blocking layer is routed to review, which here means an escalation and the call proceeds.

  `decision_rule` picks which one enforces.
- **Numeric range attestor.** Every figure the bot speaks gets a three-way verdict against the ENM register, after
  `policy_numeric.py`. The attestor never re-flags a figure another speech check already flagged.
- **Constraint budgets.** Per-turn latency (max and p95), words per bot turn, and tool calls over the oracle's count.
  They are reported, not folded into `correct`.
- **Severity, tiers, release gate.** Failure codes get base severities. Action-taking journeys promote moderate and
  high by one level, as in gmsh's tier escalation. Each tier has an accuracy floor and failure budgets.
- **Hallucination probes.** Near-miss option names ("priority delivery", "same day delivery", "economy delivery") and
  a low-coverage fact (a fee to secure the card). The truth is absence, and the bot must abstain. The GMS binding of
  each fake name is reported, because a strong binding to a real option is where an embedding-driven bot would
  fabricate. The policy graph gains two ASK_KB slots (`name`, `subject`) and two intents. That is four triples; the
  rest of the graph is unchanged.

## 15. Validating what the bot says

**The gap.** Every check in §7 grades what the bot *did*: tool calls, the dialog-act labels it emits, and the end
state. The English the customer hears was never graded. A bot built on dialog acts renders words from the acts, but
the rendering step (a template, an LLM, a TTS prompt) can still say the wrong merchant, the opposite answer or the
wrong fee, or leave out the date. The governance `SpeechMonitor` (§11) blocks a few dangerous sentences on the spot.
It checks that *some* write succeeded, not that the words name the right charge or the right answer.

**Design.** The fix grades the words against three things the harness already holds:

1. **The policy graph** declares what each act must say and where the truth of each spoken value lives (the
   `speech` section: `says`, `spoken_cue`, `spoken_as`, `returns_field`, `asserted_by`/`denied_by`). These
   triples are `gms=False`. The GMS view, its key and the trained store are unchanged, so nothing is retrained. The
   phrasings are read off the graph the same way the reader of customer turns reads its aliases: no LLM and no
   hard-coded rule map. A finding's rule is the act's own PDF section in the graph.
2. **The session** (`SessionFacts.from_run`) is what truly happened:
   - the transactions the tool returned;
   - the writes that succeeded, meaning status SUCCESS with every echoed argument equal to the request (the same
     standard as the outcome grader);
   - the options offered;
   - the customer's own slots.
3. **The GMS**:
   - the exact register for fees, delivery days and limits (`lookup_enm`);
   - rule stances (`stance_check`, the R27 u-space tension);
   - the claimed figure's distance from the fact against the relation's cap (`score_triple`, `tau`), reported per
     figure;
   - knowlytix's `AnswerVerifier` in geometric mode, run on the same sentence as an independent reader. It reads
     one sentence without the dialog, so it is only given sentences that name the option and give the figure as
     a number. "No, it's $0.", where the option is in the customer's question, is graded but not passed to it.

   The last two are reported and never decide. The register is the authority for a number, as in the book's ENM
   (Ch. 3.4).

**Reading** (`s07_checks/speech_reading.py`). Each sentence gives claims of the form `(field, value, about, mode)`:

- Charges are named by their returned merchant. A sentence that names none refers to the act's transaction or the
  last one talked about.
- Dispositions and actions are matched to the graph's phrasings, longest first and without overlap, and associated
  with the nearest merchant mention.
- The mode separates a statement, a claimed recording, a claimed completion and an offer. A question or a
  conditional asserts no disposition and no completion ("Do you recognize it?" claims nothing; "If you'd like, I can
  replace it" offers).
- When the turn answers a policy question, the figures in it are policy figures about the asked subject, not
  account facts.

**Verdicts and findings** (`s07_checks/speech_checks.py`):

- Every claim is supported, contradicted, unsupported or unverifiable against its declared source.
- Turn-level findings:
  - the words contradict the label (`text_contradicts_label`);
  - a labelled act is not said (`label_not_said`);
  - required information is missing (`missing`). The item facts are required only on a charge's first
    presentation, after pdf.3 ("do not ask the customer to repeat information without a reason");
  - the act is not a graph transition from the prompt the customer answered (`not_allowed_here`). TRANSFER,
    SUMMARIZE and END are allowed anywhere, and so is ASK_CLARIFY. The bot cannot know what the customer meant,
    and the policy prefers asking to guessing. An unneeded clarification costs a turn, and the outcome and the
    turn-by-turn comparison grade that. A policy answer to a question nobody asked is out of flow: the reference
    bot's narrow KB retriever reads a policy sentence aloud when the customer says "could you say that again?".
- When a bot gives no labels, the acts it expresses are inferred from cues and content, and graded the same way.

**At run time.** The check runs on every bot turn before the customer answers. With governance on, its findings
are sealed into the runtime monitor's hash-chained audit as the gate `speech_grounding`:

- contradicted or unsupported claims are DENY;
- missing or unfaithful acts are ESCALATE;
- flow findings are not sealed, as they are graded separately.

The word defects are then flagged live as well as graded, and the audit records what the customer was told.

**Scoring.**

- `c_speech` = claims supported, faithful and complete on every turn; `c_flow` = every act follows the graph.
- `outcome_correct` keeps the old verdict, so the effect of the new gate is visible.
- Severity codes:
  - `SPEECH_CONTRADICTED`, `SPEECH_UNSUPPORTED`: critical (the customer is told something false);
  - `SPEECH_UNFAITHFUL`: high;
  - `SPEECH_MISSING`: moderate;
  - `SPEECH_FLOW`: low.

**Validity.**

- **False alarms:** on the base suite the reference bot's words are all grounded, except one sentence that the
  policy-answer grader independently marks wrong: J040, "Expedited delivery costs $0.", which is a known
  limitation of the reference bot's narrow KB retriever.
- **Sensitivity:** eleven negative controls edit only the words of one act in a real conversation, and every
  edited turn must be flagged (section 5a of the report).
- **Mutants:** four word-only mutants have the reference bot's exact outcome, and only this check kills them.

**Reporting** (`gmsjourney/s07_checks/diagnosis.py` -> `bot_validation.html`). Each run has two verdicts:

- **did the right thing**: the outcome, the semantic check;
- **said it right**: `c_speech`, the response check.

The page splits runs by four dimensions only:

- the type of enrichment (none / template / LLM);
- the base-case category;
- the DOE factors, from the official catalogs (`origin: official`);
- the enrichment factors this bot's application layer adds (`origin: custom`).

Each factor table shows the three rates per level. Factors the deviance test finds significant come first. Below
that, the page has four detail views:

- the stage where the semantic check failed;
- every claim by where its truth was read (session or GMS);
- the GMS-checked figures, with the register verdict, the GMS distance vs cap and knowlytix's reading;
- response cards, which show the failing responses with their case's factor levels.

The cards are kept per run (`speech_turns` in `dataset/case_results.jsonl`), so the page can be rebuilt without a
rerun.

**Assumptions to confirm.**

- PDF section 2B ("present one transaction at a time using merchant, amount and date") is read as: all three are
  spoken.
- The transaction's date is the posting date the tool returns (`date`, added to the tool result next to
  `age_days`).
- "Expedited" is also spoken as "express", "rush" or "fast shipping". It is not "overnight": the PDF says expedited
  arrives in 2 business days, so "overnight" would misstate it, and a near-miss name must not bind to a real option.
  The full set of phrasings is in the graph's `bot_lexicon` and its aliases.
- Every "what this act must say" triple (`says`) carries a `basis` and a `note`, so a reviewer can check the speech
  layer against the PDF:

  | basis | meaning | triples |
  |---|---|---|
  | stated | the PDF says it | 7: the transaction's merchant, amount and date (pdf.2B); the fee (pdf.5); the action being confirmed (pdf.2E); the summary's records and actions (pdf.2F) |
  | inferred | follows from the PDF's words | 8: the context offered to an unsure customer (merchant, amount, date); the disposition said back; naming the option with its fee; the delivery option in a confirmation; that "describe it as completed" names the action and its delivery |
  | interpreted | the PDF read one way | 1: the opening statement is pdf.6's required/static example |
  | design | not in the PDF | 4: explaining the transaction limit; the "no option" state; delivery days with the fee; the age of a charge in the context |

  The tool field `age_days` is also marked design: pdf.4 lists transaction_id, merchant, amount and date, and the mock
  backend adds the age so the 30-day window can be checked.


## 16. Domain packs: one engine, any journey

A journey is a pack: its policy graph, `project.yaml`, `domain.yaml` (lexicon, tool behaviour, bot wording) and its
enrichment catalogs. Charge review's graph is built from the PDF; supplementary card's is a compact graph
(`gmsjourney.policy_graph/2`), compiled by `s02_graph/compile.py` into the same full graph. The compiler writes
the compact state machine as executable flow: call states become quiet tool steps left on the tool's status (an
echo of another value reads as a mismatch), the always-available behaviours become transitions out of every prompt
in the interpreter's order of precedence (a person, a policy question, the state's own acts, an unclear answer),
and the turn limit is checked before a prompt is said (`any` with `retract`). It also writes what each act carries,
what each tool is called with, the outcome and claim acts, and the speech layer. The compiled flow walks every
journey the compact interpreter mines identically (85 for supplementary card, `tests/test_pack_compile.py`).

What differs between packs is read from the pack: the tools' roles (`has_role`), the scoring (the per-item scoring
configured in charge review's `domain.yaml`, the graph-generic scoring of `s07_checks/scoring.py` for a compiled pack:
every successful write, every outcome act, every machine-checkable rule), the policy questions (charge review's
intents, the graph's figures for a compiled pack), the reference bot (charge review's own, or the graph's
interpreter with the pack's wording), the invariance controls, the speech check (`s07_checks/speech_checks.py` reads any
pack's value sets, figures and limits) and the LLM arm's prompts (`s05_rewording/llm_pack.py`). A factor of a compiled pack
is relevant when the case reaches its decision point (`relevant_when: reached`, decided by walking the case), and
`s01_packs/drafting.py` drafts a pack's factors from its graph.

Two decisions that read differently across packs are derived, not named. The state a policy question is asked in
when nothing else is pending (`GraphNames.question_state`) is the first prompt the journey reaches once the gate tool
succeeds (charge review: the transaction being presented; supplementary card: the relationship question). A
turn-pressure level that must end in a transfer pads to its target and, if the limit lets the last turn through
(supplementary card counts the limit "before the application is complete", so a final goodbye is not over it),
takes the smallest padding past the target that triggers the transfer; charge review's first try already does, so
its cases are unchanged.

**Loops.** A compact graph can loop over a list a tool returns (an `each` state: `each`, `as`, `where`, `cap`, `do`,
`empty`, `over_cap`, `done`; a body state's `@next` comes back for the next item). The compiler writes it as a quiet
step that advances the loop (`loop_step`: load and filter the items the first time, then the next item, bound to the
loop variable, with the per-item counters reset) and branches on item / empty / over / done; `$txn.transaction_id`
reads the current item (`at`), a `$writes` field and a transfer `summary` read what the write tool truly did
(`writes`). charge_review_graph (charge review as a compact graph) compiles to the same 276 journeys as its
interpreter (`tests/test_pack_compile.py`). In journey generation a loop is data like everything else: the bank
builds the list from the pack's item records (`items` how many, default 1; `last_item_age`; `extra_item_age`), the
loop's prompts get one customer queue per item (`per_item`, a level for every item with `item: all`), and
`s01_packs/drafting.py` drafts three loop factors: the item count (one, two, at the cap, over it), the window's edges, and
the second item answered differently (`tests/test_loops.py`).

## 17. The journey console

`gmsjourney/s10_console` is a local web app over the same objects the notebook uses; nothing in it is specific to a pack.
`s10_console/server.py` (the standard library's threaded HTTP server) serves one page and a JSON API. The journey calls go
through `s10_console/studio.py: Studio` (state, check, preview, save) under a lock, because the active pack is process-wide. A
campaign never runs in the server: `runs.py` starts `python -m gmsjourney.s10_console.runs <run>` as its own process (one at
a time), with the device in `GMSJOURNEY_DEVICE`. The worker calls `run_campaign` with an `on_step` hook and appends
events to `events.jsonl` (start, step_start, step_done with that step's summary, log lines, done with the headline,
error). The page polls from the last event it has, so a closed tab or a restarted kernel loses nothing.
`s10_console/runs.py` turns the campaign's result dict into what each step produced (metrics, small tables, notes); it
reads only keys the pipeline writes, so a step that was switched off reads as skipped.

The page keeps two sources apart everywhere, in colour and in words: what the graph decides (read-only) and what
the project sets (editable). The journey factors are the centre: each edit is re-checked against the graph after a
short pause, problems are shown on the factor they belong to, and Preview walks the graph for one level with
everything else at its default. The design follows Vercel's web interface guidelines: keyboard-operable with
visible focus, labelled controls, the URL as state (`#/factors?f=...`, `#/run/<id>?step=7`), `aria-live` for runs
and previews, confirmation before discarding or cancelling, status in words and colour together, light and dark
themes, and reduced motion respected.

The console shares its process with the notebook, so every journey call puts back the journey that was active
before it (`server._on`); the console never leaves the kernel on a journey it chose. A run's worker is the leader
of its own process group, so Cancel stops everything the campaign started. Its pid is in the run's first event, so
a run started before a kernel restart is still found and can still be cancelled. The headline carries both release
gates. Showing only the original-wording gate made a bot that fails every reworded case look releasable.


## 18. Amounts and policy in what the bot says

The readers read what the graph names in the wordings it lists, so everything else used to pass unread. Three
changes close that gap, and all three use only what every graph provides.

**Figures are tied to their subject.** The reader attaches a figure to the member named nearest it. A figure that
names nothing goes to the asked member the first time and to `ANY` after that (checked against every member),
because "There's also a $15 option" is about some option, not the one already priced. The policy-answer grade
(`s03_knowledge/questions.py: anchored_figures`) uses the same reading: every figure tied to the asked subject must be the truth, so a
correct answer is correct in any order. A figure about a sibling of the asked subject (same value set, or another
limit when a limit was asked) is `volunteered`. Siblings are the only case the graph can call irrelevant without
guessing; a fee volunteered in answer to a fraud question is not flagged.

**Every clause has an outcome** (`s07_checks/speech_policy.py`). `PolicyCorpus` collects the graph's statements (its `has_text`
triples, plus a compact graph's `knowledge`), figures (limits with units, members' numeric triples), actions (tool
verbs, write-value names) and forbidden stances (PDF phrase lists, a compact rule's `pattern`). A clause no reader
claimed is checked in order:
1. a number with a unit the graph uses is compared with the graph's figure for that subject;
2. a forbidden stance asserted is `contradicted`;
3. a commitment's main verb must be an action;
4. a clause with a policy marker must match a statement (rare shared stems plus equal figures, at least two pieces of
   evidence);
5. otherwise the clause is `unread` (advisory).

Invented policy, wrong figures and promises of actions the graph lacks become claims with `unsupported` or
`contradicted` verdicts, so they flow into the existing SPEECH_* codes. Internal limits (a compact graph's flag) and
the project's `never_volunteer` list give `disclosed`.

**Limits.** Topic matching is not entailment: a matched statement whose polarity flipped, with no figure and no
stance pattern, is still `supported`. `judge: llm` covers that, restricted to the candidate statements it is shown.
Promise checking is lenient for any clause that names a write value ("I'll waive the expedited fee" passes), and
strict for verbs outside the graph. The marker lists are plain English and can be overridden per project.
