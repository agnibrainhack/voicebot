# `gmsjourney/s07_checks/` — Checks

Grading a run (outcome, stages, words), severities and the release gate, the runtime monitor and its calibration, and the statistics of why a bot fails.

The full explanation, with the theory and diagrams, is chapter 7 of the guide: [docs/guide/07-checks.md](../../docs/guide/07-checks.md).

```
gmsjourney/s07_checks/
├── __init__.py        Checks: scoring, what the bot said, the runtime monitor, diagnosis
├── diagnosis.py       Diagnosis: why the bot fails, and whether the test suite can be trusted
├── gates.py           The runtime gates and their calibration
├── llm_judges.py      Two LLM judges beside the rule-based verdict: a second opinion on every run, never the decider
├── runtime.py         The runtime monitor: the checks that would run next to the bot in production
├── scoring.py         Scoring a run: was it correct, how bad is a failure, may the bot be released
├── speech_checks.py   Checking what the bot said: every claim against its source, every act for completeness
├── speech_policy.py   Policy in what the bot says, and proof that the speech check catches errors
├── speech_reading.py  Reading what the bot said: the contract, the session's facts and the utterance reader
└── speech_trace.py    How the speech check read one sentence, step by step (for the speech check page)
```

## `__init__.py`

Checks: scoring, what the bot said, the runtime monitor, diagnosis.

## `diagnosis.py`

Diagnosis: why the bot fails, and whether the test suite can be trusted.

Parts:

* PART 1 - Statistics for failure attribution (book Ch. 10), dependency-light (numpy/scipy).
* PART 2 - Which inputs break which part of the bot (book Ch. 10, 12.6).
* PART 3 - Mutation testing: does the suite catch real bugs?
* PART 4 - Resilience under tool faults: does the bot fail loud? (book Ch. 11, 12.8)
* PART 5 - Bot validation: what the bot did and what it said, case by case and by what each test case contained.

| name | kind | what it is |
|---|---|---|
| `logit_fit` | def | Logistic regression by Newton-Raphson (IRLS) with a small ridge on the slopes (not the intercept). |
| `bh_adjust` | def | Benjamini-Hochberg adjusted p-values (controls the false discovery rate over many factor tests). |
| `wilson` | def | Wilson score interval for k successes out of n (better than the plain +-1.96*se near 0 and 1). |
| `deviance_test` | def | Does one factor change the metric? Likelihood-ratio test of a logit with the factor against the null. |
| `level_table` | def | Failure rate per level with Wilson CI, and odds ratio of success vs the reference level. |
| `auc` | def | Area under the ROC curve from ranks (Mann-Whitney U / (n_pos * n_neg)). |
| `hosmer_lemeshow` | def | Hosmer-Lemeshow goodness-of-fit p-value: observed vs predicted counts in deciles of predicted risk. |
| `joint_model` | def | One logit with every factor at once: pseudo-R2, AIC, BIC, ROC AUC and the Hosmer-Lemeshow p-value. |
| `interaction_test` | def | Do two factors interact? Likelihood-ratio test of the model with their product terms against without. |
| `attribute` | def | Rank the factors by how much they change ``metric``: a deviance test per factor, BH-adjusted, with a per-level table (failure rate, Wilson interval, odds ratio against the reference level). |
| `diagnose` | def | The full failure attribution of a run set: per-factor tests, the joint model, interactions among the significant factors, the worst single levels, and per-category / per-component breakdowns. |
| `default_fault_plan` | def | The tool faults the mutants are also run under: for a pack with per-item records, its writes erroring and returning plausible wrong results, its reads erroring; otherwise every tool but the handoff, both ways. |
| `mutation_analysis` | def | governance: a GovernanceConfig; when given, every run is also watched by the runtime monitor and a mutant counts as *flagged at runtime* when the monitor raises an alarm (deny/escalate) on a scenario where it raised none of that kind for the reference bot - detection without an oracle, as in production. |
| `fault_tools` | def | Every tool of the pack but the handoff, in graph order (the auth tool and the account tools). |
| `probe` | def | Resilience probe: inject each fault kind into each tool and check that the bot fails loud (escalates or transfers) where the oracle, under the same fault, says it should. |
| `did_right` | def | The semantic check: the outcome (before any speech gate). |
| `said_right` | def | The response check: every claim grounded, faithful to the labels and complete (``c_speech``). |
| `rates` | def | n, and the share of runs that did the right thing, said it right, and both (correct). |
| `two_by_two` | def | How many runs did / did not do the right thing, against said / did not say it right. |
| `factor_groups` | def | (DOE factors from the official catalogs, enrichment factors this bot adds), in design order. |
| `by_level` | def | The three rates for each level of one dimension. ``key`` is a row field name or a function of a row. |
| `factor_performance` | def | Performance by type of enrichment, base category, DOE factors and enrichment factors. |
| `stage_failures` | def | For each journey stage, the share of runs (per type of enrichment) where the bot's decision there was wrong, plus 'every step allowed' (workflow admissibility). |
| `claims_of` | def | (row, response card, claim) for every claim in every run. |
| `basis_breakdown` | def | Claims by where their truth was read, with their verdicts. |
| `gms_figures` | def | The policy figures and stances checked against the GMS. |
| `problems_in` | def | (problem kind, field or act) for every failed claim and finding of one response. |
| `failing_cards` | def | The failing responses grouped by problem (graded problems first, then the most frequent): per group the number of responses and of cases, and up to ``per_group`` example cards from different cases (different arms where possible). |
| `example_card` | def | A fully grounded response with a GMS-checked figure, to show how a card reads. |
| `basis_table_html` | def | Claims by where their truth was read (``basis_breakdown``), as a table. |
| `gms_table_html` | def | The GMS-checked figures per kind (``gms_figures(...)["kinds"]``), as a table. |
| `render_card` | def | One bot response card: the case and its factors (significant ones first and highlighted), what the customer said, what the bot said (and its act labels), every checked claim with its truth and where it was read, and the turn's findings. Mentions with nothing to check against are counted, not listed. |
| `render_bot_validation_html` | def | The whole page (self-contained HTML; same look as report.html, light and dark). |
| `cards_page` | def | One case's response cards as a small self-contained page (every response with a claim or a finding). |
| `as_iframe` | def | A page wrapped in an ``<iframe srcdoc>`` so its styles stay inside the frame (for IPython's ``HTML``). |
| `validation_summary` | def | The page's numbers without the cards (for report.json and report.html). |
| `write_bot_validation` | def | Write the page into the campaign folder and return its summary. |
| `write_bot_validation_from_dir` | def | Rebuild the page from a finished campaign's files (report.json, dataset/case_results.jsonl). |

## `llm_judges.py`

Two LLM judges beside the rule-based verdict: a second opinion on every run of the bot, never the decider. Judge A
(the auditor) compares each turn with the graph's expected steps; judge B (the reviewer) sees only the conversation
and the policy rules of the states it went through.

| name | kind | what it is |
|---|---|---|
| `Judges` | class | The two judges (cache tags ``judge_a``, ``judge_b``): ``judge_case`` (one test run), ``judge_turn`` (one Playground turn), ``judge_many`` (in parallel). |
| `combine` | def | The judges beside the rules' verdict: agree, judges split, or judges disagree with the rules. |
| `rules_of` | def | The policy rules of the dialog states a call went through (id -> text). |
| `summarize` | def | Agreement of each judge with the rules, how often the judges agree, and the runs to review. |

## `gates.py`

The runtime gates and their calibration.

Parts:

* PART 1 - Runtime governance gates (book Ch. 12; forgeloop's gate protocol, re-implemented).
* PART 2 - Calibrating the runtime gates (book Appendix C; beyond-ship-and-pray appendix_C notebook).

| name | kind | what it is |
|---|---|---|
| `GateDecision` | class | What a gate decides about a proposed step: let it through, stop it, or let it through and raise an alarm. |
| `GateResult` | class | One gate's decision on one proposed step, with its reason (and the score and rule when there is one). |
| `ToolCall` | class | forgeloop-shaped action: kind / tool_name / arguments. |
| `GateState` | class | What a runtime monitor can observe of one conversation (no oracle, no fixture). |
| `ToolRegistry` | class | Tool names from the policy's tool table (store entity type 'tool'); argument schemas from the backend contract (domain.yaml -> tools.*.args: default "" = required, null = optional); enumerated argument values from the store's value sets (tools.*.arg_values). |
| `SyntaxGate` | class | The first gate: the tool exists and its arguments match its schema. |
| `compile_policies` | def | The policy's call checks, compiled from domain.yaml -> rules.calls (s07_checks/runtime.py); every number comes from the ENM. |
| `completeness_check` | def | End of conversation (domain.yaml -> rules.completion): without a handoff, every returned item due for review must have been written with success, and no terminal value left behind. |
| `PolicyGate` | class | The policy gate: the compiled rule checks in order; the first that does not allow decides. |
| `PlanVerdict` | class | A whole plan's verdict: admissible or not, its mean transition distance and its worst single step. |
| `GMSPlanGate` | class | Validate a whole tool plan against a budget. |
| `SpeechMonitor` | class | Checks on bot utterances (the book's draft verifier, generalised to a voice bot). The checks, their patterns and the figures they compare with come from domain.yaml -> rules (speech, spoken_figures); every figure is read from exact numeric memory. |
| `injection_flag` | def | The first prompt-injection pattern the text matches (forgeloop's regex baseline), else None. |
| `LabeledEdge` | class | One calibration example: (context, relation, tail) labelled admit (a policy edge) or block. |
| `workflow_cohort` | def | The calibration cohort: every workflow edge of the policy (admit) and every other possible transition between its tools (block). |
| `sweep` | def | False admits, false rejects and accuracy of the rule "admit when score <= theta" at each theta. |
| `operating_points` | def | The candidate thresholds (midpoints between observed scores) and the two operating points: the regulated one (zero false admits, then fewest false rejects) and the most accurate one; plus AUC and separability. |
| `logistic_p50` | def | One-feature logistic P(admit / score) (L2 on the slope, like sklearn's default C=1) and the score where P = 0.5 - gmsh GMSJudge's operating-point rule - with its errors on the cohort. |
| `holdout_eval` | def | Stratified k-fold: fit each rule on k-1 folds, count its errors on the held-out fold. |
| `bootstrap_thetas` | def | Stratified bootstrap of every rule's threshold: mean, sd and 95% interval. |
| `tau_bootstrap` | def | Per relation: the cut re-fitted on bootstrap resamples of (true facts, counterfactual tails) with the store's own operating-point rule, and the realised TAR/FAR at the deployed cut. |
| `calibrate_plausibility` | def | Calibrate the transition-plausibility gate on the workflow cohort scored by the store. |
| `plan_cohort` | def | Admissible tool plans (oracle journeys) and inadmissible perturbations of them. |
| `calibrate_plan` | def | Calibrate the plan gate: admissible against inadmissible plans, scored by mean drift and by worst step; deploy the criterion with fewer false rejects at zero false admits. |
| `multi_threshold_design` | def | The appendix's balanced design over the thresholds that matter, each row scored. |
| `psi` | def | Population stability index between two score samples (quantile bins of `expected`). |
| `store_fingerprint` | def | A short hash of the store's facts and training settings (a change triggers recalibration). |
| `policy_fingerprint` | def | A short hash of the policy's rules (a change triggers recalibration). |
| `recalibration_triggers` | def | record = what was stored at calibration time (fingerprints, reference scores, FR). |
| `gate_probe_sweep` | def | Book 12.5.1: every (previous step, proposed tool) pair through the geometric gate, compared with the policy graph. Escalation actions are safe-harboured (never denied). |

## `runtime.py`

The runtime monitor: the checks that would run next to the bot in production.

Parts:

* PART 1 - Runtime rule checks as generic templates (domain.yaml -> rules).
* PART 2 - Numeric range attestor for figures the bot speaks (gmsh harness/governance/policy_numeric.py).
* PART 3 - Runtime monitoring of one conversation (book Ch. 12.5; forgeloop GovernanceHarness re-done
* PART 4 - The runtime monitor for a compact policy graph (book Ch. 12.5).
* PART 5 - forgeloop's governance pieces driving our runtime monitor.

| name | kind | what it is |
|---|---|---|
| `figure` | def | A limit or named figure from exact numeric memory. |
| `compile_call_checks` | def | Compile the pack's call rules (domain.yaml -> rules.calls) into checks on each proposed tool call: verified before account tools, attempt limits, one write per item, confirmation before a write, nothing after a handoff. |
| `completion_results` | def | End-of-call completeness checks (domain.yaml -> rules.completion), e.g. every reviewed item written. |
| `Attestation` | class | One spoken figure's verdict against the exact register: PASS, REVIEW or FAIL, with the expected values. |
| `NumericRangeAttestor` | class | The families, their spoken patterns, registers and related figures come from domain.yaml -> rules.attestor; every register value is read from exact numeric memory. |
| `state_hash` | def | A short, stable hash of any JSON-able state (used to seal audit events). |
| `GovernanceConfig` | class | The runtime monitor's settings: the calibrated plausibility cut and plan budget, monitor or enforce, and the decision rule (strict: first DENY wins; quorum: one blocking layer goes to review). |
| `RuntimeMonitor` | class | Watches one conversation as it happens: every proposed tool call through the gates, every bot utterance through the speech gates, completeness at the end; every decision sealed in the audit chain. |
| `GovernedTools` | class | Tool client wrapper: gates before every call; records results for the monitor. |
| `GraphSpeechMonitor` | class | The speech gates for a compact-graph pack: the pack's speech rules (patterns) and every spoken money figure against the graph's figures. |
| `GraphRuntimeMonitor` | class | The runtime monitor for a compact-graph pack (the same gates, the graph's speech monitor). |
| `ForgeloopPlausibilityGate` | class | forgeloop's GMS plausibility gate behind our gate protocol: scores (previous tool, enables, next tool) on the store; the safe handoff tool is escalated, never denied, so the bot can always hand over. |
| `ForgeloopAudit` | class | forgeloop's AuditLogger with a logical clock (deterministic, replayable chains). |

## `scoring.py`

Scoring a run: was it correct, how bad is a failure, may the bot be released.

Parts:

* PART 1 - Per-turn constraint budgets (gmsh ConstraintOracle style, applied to a voice bot).
* PART 2 - Severity, risk tiers and the tiered release gate (gmsh testing/severity.py + audit.py, for this bot).
* PART 3 - Run scenarios through the SUT and score them (book Ch. 9.5).
* PART 4 - Scoring a run against a compact policy graph.

| name | kind | what it is |
|---|---|---|
| `ConstraintBudget` | class | Per-turn budgets: bot latency (max and p95), words per bot turn, tool calls over the oracle's count. |
| `check_constraints` | def | Measure a run against the budgets; returns the measurements and the violations (reported, not failed). |
| `code_severity` | def | failure code -> base severity: the built-in table with the project's overrides. |
| `tier_escalation` | def | risk tier -> {severity: promoted severity}. |
| `risk_tier` | def | action_taking when the correct journey writes to the account; informational otherwise. |
| `failure_codes` | def | The failure codes of a scored run (violations, wrong record, wrong card action, speech findings ...). |
| `classify` | def | (run severity, [{code, base, tiered}]) - the run's severity is its worst tiered code. |
| `TierProfile` | class | The release-gate thresholds of one risk tier: accuracy floor, critical and high-severity budgets, budget compliance, and per-code budgets. |
| `default_profiles` | def | The project's release-gate profiles (evaluation.release_gate), else the built-in ones. |
| `release_gate` | def | Per-tier verdicts and the overall release decision (all tiers must pass). |
| `response_cards` | def | The bot's responses that said something checkable, one compact card each, for the reports. |
| `score` | def | Score one scenario's run against its expected answer: correctness, per-stage components, workflow adherence, rule violations, clean termination, the speech check, budgets, severity and risk tier. |
| `progress_bar` | def | A tqdm bar (notebook-aware) when tqdm is installed and show is True, else None. |
| `run` | def | Run and score every scenario; with `desc`, a progress bar (tqdm) shows runs done and accuracy so far. |
| `summary` | def | Headline rates over scored runs: accuracy, workflow adherence, clean termination, runtime invariance, budget compliance, policy-answer accuracy and the speech rates. |
| `weak_link` | def | Charge each failed run to the first stage, in workflow order, that erred (Ch. 10.7). |
| `scored_stages` | def | The stages a run is scored on, in order (the conversation stages that decide something). |
| `GraphTrace` | class | A Trace scored against a compact policy graph: outcome acts said, success claims, states visited and the decisions taken on a mined path. |

## `speech_checks.py`

Checking what the bot said: every claim against its source, every act for completeness.

Parts:

* PART 1 - Verifying a bot turn: every fact it states, and whether it says what it should.
* PART 2 - The speech check for any pack compiled from a compact graph: what the bot says, read and checked against the
* PART 3 - Validating what the bot actually says, sentence by sentence, against the policy graph and the GMS.

| name | kind | what it is |
|---|---|---|
| `ClaimVerdict` | class | The verdict on one claim: was what the words state true, and where was the truth read? |
| `Finding` | class | A problem with a turn as a whole (not with one claim): unfaithful, incomplete, or out of flow. |
| `TurnReport` | class | Everything the speech check found in one bot turn. |
| `UtteranceVerifier` | class | Checks utterances against a GMS store and the speech contract read from its policy graph. |
| `ConversationChecker` | class | Checks every bot turn of one conversation, remembering what was talked about. |
| `turn_where` | def | A bot turn in the turn-by-turn table's numbering: bot turn 0 is the opening, bot turn n answers customer turn n-1. |
| `spec_source` | def | The PDF section that says what ``act`` must say (pdf.2B ...): the rule a finding cites. |
| `PackModel` | class | What the generic check needs to know about a compiled pack, read from its compact graph. |
| `GenericReader` | class | Reads a compiled pack's bot turns into claims on its value sets, figures and limits. |
| `GenericChecks` | class | Claim checks and completeness for a compiled pack (used by UtteranceVerifier when the contract is generic). |
| `claim_field` | def | The claim field a ``says`` binding is read as (None: free text the reader does not read). |

## `speech_policy.py`

Policy in what the bot says, and proof that the speech check catches errors.

Parts:

* PART 1 - Policy in what the bot says: every clause is read, checked against the graph, or flagged.
* PART 2 - Checking the checker: negative controls on what the bot says.

| name | kind | what it is |
|---|---|---|
| `markers` | def | The marker lists: the defaults, with any the project sets (project.yaml speech.policy.markers). |
| `settings` | def | project.yaml speech.policy: advisory (finding kinds that never fail a turn), never_volunteer, judge. |
| `stem` | def | A crude stem shared by a word's common forms: 'transactions'/'transaction' -> 'transa', 'charges' -> 'charg', 'eligible'/'eligibility' -> 'eligib' (prefixes of six letters after stripping a common ending). |
| `terms` | def | The content stems of a text (no stop words, nothing shorter than three letters). |
| `clauses` | def | A sentence cut where a new statement starts: at ';', and at ', and' / ', but' / ', so' / ', plus' / ', while'. |
| `Statement` | class | A text the graph carries: a rule, a limit, a tool's description, the opening or a knowledge line. |
| `Figure` | class | A number the graph states: a limit (with its unit), or a member's numeric attribute. |
| `PolicyCorpus` | class | Everything the policy check compares a clause with, read from the graph: statements, figures with their units, how subjects are worded, action verbs, forbidden stances, and word rarity (idf) for matching. |
| `ClauseVerdict` | class | What one clause was found to say about policy. |
| `PolicyChecker` | class | Reads the clauses the readers left, against the graph (see the module docstring). |
| `LLMPolicyJudge` | class | Re-reads policy-sounding clauses with an LLM given the graph's closest statements (project.yaml ``speech.policy.judge: llm``). A restates / contradicts verdict must cite one of the statements offered, or it is ignored; answers are cached (enrich/llm/client.CachedLLM), so a campaign replays them exactly. When the LLM is unavailable the graph's verdict stands and the reason is kept in ``errors``. |
| `controls` | def | The controls for the active pack. |
| `run_controls` | def | Apply every control to every scenario where it can apply; count the edited turns the speech check flags. |

## `speech_reading.py`

Reading what the bot said: the contract, the session's facts and the utterance reader.

Parts:

* PART 1 - The speech contract: everything the validator needs from the policy graph, in one object.
* PART 2 - Session facts: what is true in one conversation so far.
* PART 3 - Reading a bot utterance into claims, without looking at the bot's act labels.

| name | kind | what it is |
|---|---|---|
| `also_allowed` | def | Customer act (or "any") -> bot acts the speech-flow check accepts after it in any prompt state, beyond what the graph's transitions allow (project config: speech.also_allowed). |
| `FieldSpec` | class | One piece of information an act tells the customer. |
| `ActContract` | class | What one bot act (dialog state) says and the words that mark it. |
| `StanceSpec` | class | A rule whose fact the bot can be heard asserting or denying (e.g. R27, 'not proof of fraud'). |
| `SpeechContract` | class | Everything the reader and the verifier need to know about speech, read once from the policy graph. |
| `write_key` | def | What a write is about: the field it echoes (graph ``echoes``), else for the per-item write its item id and for the other write its action argument. |
| `Truth` | class | The true value for a claim and where it was read. ``known`` is False when the session has nothing to compare against (e.g. a 'recorded' claim before any record succeeded). |
| `SessionFacts` | class | What is true for this customer, so far in this conversation (see the module docstring). |
| `signal_words` | def | The project's cue lists (project.yaml, speech.signals). |
| `words_to_digits` | def | 'fifteen dollars' -> '15 dollars', 'twenty-five' -> '25' (0-99; enough for spoken figures). |
| `Claim` | class | One fact an utterance states. |
| `ReadUtterance` | class | What the reader understood from one utterance. |
| `UtteranceReader` | class | Reads utterances with the wordings the speech contract (the policy graph) declares. |
| `fallback` | def | What a figure that names nothing is about: the asked-about thing the first time, ``ANY`` after that. |

## `speech_trace.py`

How the speech check read one sentence, step by step: for the speech check page (`s09_reports/speech_playground.html`).
It re-runs the readers' own patterns on the same words and reports what they saw; it decides nothing. The verifier
keeps it per turn in `TurnReport.trace` (with the rules each claim's check tried, `ClaimVerdict.steps`, and the
comparison made, `ClaimVerdict.compare`).

| name | kind | what it is |
|---|---|---|
| `explain_sentence` | def | One sentence as read: the words after normalising, the cue words and the mode, every reader's matches (kept, and dropped because a longer phrase took the words), and the readers that found nothing. |
| `readers` | def | How many phrases each value-set reader tries (kept once per turn). |
| `facts_snapshot` | def | What the checker knew when the agent spoke: the customer's meaning cards, the confirmed and failed writes, the charges and each tool's latest result. |
