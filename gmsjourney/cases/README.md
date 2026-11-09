# `gmsjourney/cases/` — Cases

What a test case is (meaning cards, journeys, the bank fixture, the simulated customer) and which cases to run (covering arrays for journeys, space-filling designs for wording).

The full explanation, with the theory and diagrams, is chapter 4 of the guide: [docs/guide/04-cases.md](../../docs/guide/04-cases.md).

```
gmsjourney/cases/
├── __init__.py  Test cases: what a case is, the journeys and their factors, and which cases to run
├── design.py    Which cases to run: covering designs for journeys, space-filling designs for wordings
├── factors.py   Journeys mined from a compact policy graph: the case structure, the bank, the simulated customer
├── journey.py   Journey content: what a test case *is*, before any presentation factor
└── meaning.py   Journey ontology and semantic frames
```

## `__init__.py`

Test cases: what a case is, the journeys and their factors, and which cases to run.

## `design.py`

Which cases to run: covering designs for journeys, space-filling designs for wordings.

Parts:

* PART 1 - Space-filling designs and scenario composition (book Ch. 7, Appendix C).
* PART 2 - Optimal journey base cases: constrained covering array + coverage closure.

| name | kind | what it is |
|---|---|---|
| `log_phi_p` | def | log phi_p of a design X (rows = runs, columns in [0,1]) with column weights w. |
| `phi_p_design` | def | A space-filling design of ``n_runs`` rows over the factor levels (Appendix C): Sobol start, phi_p minimised by L-BFGS with p annealed from p_start to p_final, then rank-transformed (Latin-hypercube marginals) and refined by level swaps that keep every level's count; ``method`` picks a simpler variant for comparison. |
| `discretize` | def | Map a continuous design in [0,1] to factor levels by equal-width bins. |
| `simple_design` | def | A random design (each level drawn uniformly): the baseline every other design is compared with. |
| `balanced_design` | def | proofloop.evaluation.doe.balanced_design (Apache-2.0, Knowledge/beyond-ship-and-pray). |
| `coverage_report` | def | proofloop.evaluation.doe.coverage_report: level counts per factor. |
| `encode` | def | Map level rows back into [0,1] (the centre of each level's bin), to measure distances. |
| `design_quality` | def | How good a design is: log phi_p, minimum distance, marginal balance (worst chi-square p) and the share of level pairs covered. |
| `compare_designs` | def | The quality of random, balanced (proofloop), Sobol and Sobol+refine designs on the same factors. |
| `compose` | def | Scenarios = bases x design rows. ``cross``: every base with every row; ``embedded``: rows dealt to bases as a balanced blocking factor, in a farthest-point order so each base sees dissimilar conditions. |
| `per_base_dispersion` | def | Mean over bases of the minimum distance between that base's rows (maximin; higher = more spread). |
| `power_curve` | def | Monte Carlo power to detect one level of one 3-level factor that raises the odds of failure by `odds_ratio`, with a Bonferroni-adjusted LR test over all factors. |
| `CoverageReport` | class | What a journey suite covers: level pairs (required, covered, infeasible), rules, workflow edges, handoff triggers and boundaries, covered and missing. |
| `JourneyBaseGenerator` | class | Builds the journey bases: a covering array over the factors (AETG-style greedy: each new case is the candidate that covers most still-uncovered feasible level pairs), then a closure that adds cases until every rule, edge, trigger and boundary is exercised; ``finalize`` adds policy questions and turn padding. |
| `primary_category` | def | Map a journey to the base category its distinguishing content exercises (project ``categories``). |
| `to_item` | def | A journey spec as a base case: its fixture, the oracle's expected answer and components, and the first customer turn's canonical wording. |

## `factors.py`

Journeys mined from a compact policy graph: the case structure, the bank, the simulated customer.

Parts:

* PART 1 - Journeys mined from a compact policy graph: base cases, fixtures, the simulated customer.
* PART 2 - Base-case factors for a compact policy graph: you choose the dimensions, the graph decides the truth.

| name | kind | what it is |
|---|---|---|
| `GraphSpec` | class | One mined base case. Immutable; compared and hashed by content. |
| `build_fixture` | def | The bank for a mined journey: the credential and, per tool, the statuses its calls return. |
| `GraphSimulator` | class | The customer of a mined journey: answers each prompt from the case's queue for that state (explore mode tries every accepted act), remembering what it said. |
| `default_frame` | def | The happy-path answer at a prompt: its first accepted act with every required slot at its first value. |
| `loop_items` | def | ``n`` items of a returned list ({"items": type, "id": field, "age": field}), built from the pack's records (domain.yaml -> fixtures.items.<type>: records, ages, id_prefix): the last one ``last_age`` days old, and one more ``extra_age`` days old after them when given (outside a window). |
| `mine_paths` | def | Every path through the machine with at most ``max_deviations`` deviations from the default, one per distinct behaviour. Returns (spec, oracle trace) in discovery order; each trace carries its ``decisions`` (decision point -> option, defaults included). With ``ledger``, every explored conversation is appended as (decisions, behaviour key), duplicates included. |
| `coverage_closure` | def | Paths with up to ``extra`` more deviations, kept only when they reach a workflow edge, a state, a rule or a transfer reason no mined path reaches yet. |
| `behaviour_key` | def | A hash of everything observable in a trace (outcome, acts, slots, tool results): two cases with the same key test the same behaviour. |
| `perturbations` | def | Failure situations over every prompt state, each on the first mined path reaching it. |
| `kb_base` | def | The default path with one policy question asked at the first prompt after the first tool. |
| `deviation_kind` | def | What one deviation of a mined path is: a tool failure, a tool's decline (a failure status), returned data at a boundary (item counts, windows, option lists), or a customer choice. |
| `category_of` | def | The base category of a mined case, from what it perturbs (a person asked for, a failed tool ...). |
| `tags_of` | def | Every tag of a mined case: category, non-ok tool statuses, outcome acts said, transfer, writes. |
| `GraphSimulatorStatus` | def | (tool, status) for every call in a trace, "ok" when it is the tool's success status. |
| `to_item` | def | A mined journey as a base case (fixture, expected answer, components, first turn's wording). |
| `parse_frame` | def | ``"ACT(slot=VALUE, ...)"`` -> a SemanticFrame. |
| `frame_text` | def | A SemanticFrame -> ``"ACT(slot=VALUE, ...)"``. |
| `load` | def | The pack's base_factors.yaml, or None when it has none. |
| `check` | def | Every level is a script in the graph's terms. |
| `Case` | class | One assignment of a level to every factor, compiled into a GraphSpec. |
| `tool_defaults` | def | What every tool returns on the default path (the miner's first option: success, one item, all options). |
| `generate` | def | A covering array of strength t over the factors (relevance-aware), then compacted. |
| `structural_items` | def | The graph items a conversation reaches: workflow edges, states, rules fired, its transfer reason. |
| `closure` | def | Cases from the oracle's exploration for every state, edge, rule or transfer reason the array misses. |
| `draft` | def | A starting base_factors.yaml from the graph's decision points. |
| `outcome` | def | What a played case comes to, in the terms an ``expect`` block uses. |
| `expect_problems` | def | Where a played case departs from a level's ``expect`` block (empty = as expected). |
| `representative` | def | The case that shows what one level does: that level with every other factor at its default, or, when the level is not reached that way, the array case reaching it with the fewest other changes. |
| `review` | def | Every check the base journeys must pass, and what each level leads to, for a person to read. |
| `apply_edits` | def | Apply edits to a pack's base_factors.yaml (a copy is kept as base_factors.yaml.bak). |
| `main` | def | Command line: draft, check or review a compact-graph pack's base_factors.yaml. |

## `journey.py`

Journey content: what a test case *is*, before any presentation factor.

| name | kind | what it is |
|---|---|---|
| `activate` | def | Build every case from ``proj`` from now on (the studio previews unsaved edits this way). |
| `active_project` | def | The project the case functions currently read (set by ``activate``). |
| `JourneySpec` | class | One case: a level for every configured factor, plus the customer's policy questions. |
| `relevant_dims` | def | The factors that can influence ground truth for this case (others are N/A). A factor with ``relevant_when: reached`` is relevant when the case reaches it: walking the case through the graph, its anchor is reached (default level) or everything its level scripts is used (another level). |
| `is_valid` | def | The project's constraints between factors. |
| `primary_category` | def | The base category the case's distinguishing content exercises (the project's first matching rule). |
| `tags_for` | def | Every project tag that applies to the case, with its category. |
| `policy_question_case` | def | The case a single policy question is asked in (project ``generator.policy_question_case``). |
| `build_fixture` | def | The mock bank for a case: the project's defaults, patched by every factor's level in order. |
| `loop_ids` | def | The ids of the items a compiled pack's loop will handle: the list its tool returns, inside the window. |
| `reviewable_ids` | def | The ids of the items a customer is asked about: inside the lookback window (or a loop's items). |
| `UserSimulator` | class | The customer of one case: answers each bot act from the case's scripts (project ``customer``). |

## `meaning.py`

Journey ontology and semantic frames.

| name | kind | what it is |
|---|---|---|
| `SemanticFrame` | class | The meaning of one customer turn: a dialog act and its slot values, e.g. ``INFORM_DISPOSITION(disposition=UNRECOGNIZED)``. Hashable and order-free, so two frames compare by meaning. |
| `sequence_signature` | def | Ontological signature of a whole conversation (order-sensitive). With `states`, each meaning card is hashed together with the dialog state it answers. |
| `Ontology` | class | Maps meaning cards to canonical form, taking the dialog state into account. |
