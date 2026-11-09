# 4. Cases: what a test case is, and how few of them still cover everything

```
voicebot/
└── gmsjourney/
    ├── packs/  graph/  knowledge/
    ├── cases/            ◀ this chapter
    │   ├── __init__.py
    │   ├── meaning.py    meaning cards (semantic frames) and the ontology
    │   ├── journey.py    one case's levels, relevance, validity, the bank, the customer
    │   ├── factors.py    journeys and factors mined from a compact graph
    │   └── design.py     covering arrays (journeys), space-filling designs (wording)
    ├── rewording/  bots/  checks/  campaign/  reports/  console/
    └── policy/
```

## The idea

A test case has two parts that must be kept apart:

1. **content**: what happens in the conversation. The bank's answers (authentication fails once, the card tool is
   down) and the customer's decisions (recognises the first charge, not the second; asks for a person halfway).
   Content changes **what the bot must do**.
2. **presentation**: how the customer words it (casual, typos, numbers spoken). Presentation must **not** change
   what the bot must do. That is chapter 5.

The book's test for which is which: *a knob that would move the correct answer belongs to the base, not to the
wording factors*. This chapter is about content: the **base cases**.

## meaning.py — meaning cards and the ontology

```
cases/
└── meaning.py   ◀   SemanticFrame, sequence_signature, Ontology
```

### Meaning cards

A customer turn's *meaning* is a **meaning card** (a semantic frame): an act plus slot values that are canonical
individuals of the policy.

```
"no, I don't recognise that one"      ->  REPORT_DISPOSITION(disposition=UNRECOGNIZED)
"my wife, she's 34"                   ->  INFORM_RELATIONSHIP(relationship=SPOUSE)
"how much is the express one?"        ->  ASK_KB(intent=value, attr=fee, entity=expedited_delivery)
```

The ground truth of a journey is a function **only** of its sequence of meaning cards and the bank fixture. So if two
utterances map to the same card, they are the same test input, whatever their words. That one sentence is the
foundation of the invariance proof.

`SemanticFrame` is immutable and hashable (act plus sorted slots). `sequence_signature(frames, states)` reduces a
whole conversation to one string (order-sensitive; with states, each card is paired with the prompt it answered), so
two conversations can be compared with a single equality.

### The ontology

`Ontology.from_store(store)` reads three things from the store:

1. **domain individuals**: dispositions, card actions, delivery options, tools, rules, limits, triggers. Members of
   one value set are pairwise disjoint (OWL `AllDifferent`) and their aliases are alternative labels (`skos:altLabel`)
   of one canonical individual;
2. **dialog acts**: the closed set of things a customer turn can mean in this journey;
3. **states**: which acts each prompt accepts and how it reads a bare yes or no.

Its operations:

| method | does |
|---|---|
| `canonicalize(frame, state)` | resolve every slot value inside its own value set; turn a bare CONFIRM / DENY into what the state's `interprets_yes_as` / `interprets_no_as` says |
| `admissible(frame, state)` | the state accepts the act, the slots are declared for it, the values belong to their sets |
| `equivalent(a, b, state)` | same canonical card in the same state, and admissible there |
| `frame_distance(a, b, state)` | 0 = same meaning; 0.5 = a slot missing or extra; about 0.9–1.0 = contradiction or a different act. Swapping a value for a disjoint sibling (RECOGNIZED → UNRECOGNIZED) scores as a contradiction, using the store's tension |

The ontology can also be exported as Turtle (`ontology.ttl`) with the invariance evidence attached.

Analogy: a **dictionary with a grammar**. Many words point to one dictionary entry (aliases), entries in the same
group exclude each other (a charge is either recognised or not), and the grammar says which kinds of sentence are
acceptable as an answer to which question.

## journey.py — one case

```
cases/
└── journey.py   ◀   JourneySpec, relevant_dims, is_valid, build_fixture, UserSimulator, activate
```

A case is a `JourneySpec`: **one level for every journey factor** of the project, plus the policy questions the
customer asks. Everything about it is computed from the project file (chapter 1):

* `relevant_dims(spec)`: which factors can influence this case's truth. The others read **N/A**: after a failed
  authentication, nothing downstream matters, so the card-tool factor is irrelevant there. A factor written
  `relevant_when: reached` is relevant when walking the case reaches its decision point;
* `is_valid(spec)`: the project's constraints between factors;
* `primary_category`, `tags_for`: the report groupings (expressions in `project.yaml`);
* `policy_question_case`: the case a lone policy question is asked in;
* `build_fixture(spec)`: the **mock bank** for the case: the project's defaults, patched by every factor's level in
  order (`loop_ids` and `reviewable_ids` give the items a loop handles and the ones inside the review window);
* `UserSimulator(spec, fixture)`: the **customer**.

### The simulated customer

The customer answers each bot act from the case's scripts. Each script is a **queue of meaning cards, then a
default**: a level can push an act to be said at a state before the scripted answer, once or every time, replace the
answer, or add a slot. For a loop, each item gets its own queue (`per_item`).

The simulator is **reactive**: it answers whatever the bot just asked, not a fixed transcript. Analogy: an actor who
knows their character's intentions rather than a recording. A bot that asks the questions in a different order, or
phrases a prompt differently, still gets the right answers, so a test does not break for reasons that have nothing
to do with the policy.

`activate(project)` builds every case from a given project from now on (the studio previews unsaved edits this way);
`active_project()` returns it.

## design.py — which cases to run

```
cases/
└── design.py   ◀   PART 1 space-filling designs for wording factors (phi_p, compose, power_curve ...)
                    PART 2 JourneyBaseGenerator: covering array + closure for journeys
```

There are millions of possible journeys and wordings; a campaign runs a few hundred. Two designs pick them so little
is missed. Analogy: **seating a dinner party so every pair of guests meets at least once**, with the fewest tables.

### Part 2 first: journeys by covering array and closure

Charge review has 13 content dimensions (authentication path, retrieval, transaction count, 30-day window,
disposition pattern, record result, card intent, card options, expedited confirmation, submit result, human request,
a mid-journey policy question, turn pressure). Three ideas make a small set sufficient:

**1. Relevance.** Only relevant levels count. A pair of levels `(a=x, b=y)` is *feasible* if some valid case makes
both relevant; `feasible_pairs()` finds them by sampling valid cases with the pair fixed.

**2. Covering array (pairwise, AETG-style greedy).**

```
uncovered = every feasible pair
while uncovered pairs remain:
    try ~40 candidates:
        start from a random valid case containing one uncovered pair
        hill-climb: for each other factor, switch to the level that covers the most uncovered pairs
    keep the candidate covering the most uncovered pairs
    finalise it (attach policy questions, pad to a turn target) and remove the pairs it covers
```

This covers all 523 feasible level pairs of charge review with about 50 journeys, where every combination would be
millions.

**3. Closure.** Pairs are not everything: some rule may fire only in one rare path. The oracle reports, for each
journey, the rules fired, workflow edges traversed, transfer reasons and boundaries reached. `closure` keeps drawing
random valid cases and adds one only if it reaches something the suite has not reached yet, until every rule, edge,
trigger and boundary is exercised (4 more journeys for charge review). Long turn-limit conversations are added only
when they reach a boundary or rule nothing else does (`max_turn_limit_cases`).

**Turn padding.** A turn-limit case must hit its target exactly: padded with policy questions until the oracle
reports exactly 20 customer turns (must complete) or 21 (must transfer). `finalize` does this, and returns None when a
target is infeasible.

The value of all this is measured: the covering suite catches 75% of planted defects, a random suite of the same size
37.5% (mutation analysis, chapter 7).

`to_item(spec)` turns a journey into a base case (`QAItem`): fixture, the oracle's expected answer and components,
the first turn's wording. `CoverageReport` records pairs required, covered and infeasible, and rules, edges, triggers
and boundaries covered or missing.

### Part 1: wording factors by space-filling design

Wording factors (chapter 5) do not change the truth, so they are spread out, not paired. Each scenario is a point in
a cube `[0,1]^m` (one axis per factor); we want the points as far apart as possible so every region of the cube is
tried.

**The criterion** (`log_phi_p`). For a design `X` with `n` runs and column weights `w`:

```
phi_p(X) = ( Σ_{i<j}  d_ij^(-p) )^(1/p),     d_ij = || w ⊙ (x_i - x_j) ||
```

Small `phi_p` means no two runs are close; as `p` grows it approaches the *maximin* criterion (maximise the smallest
distance). It is computed in log space with `logsumexp` so a large `p` never overflows. Weights reflect each factor's
number of levels (cardinality).

**The optimiser** (`phi_p_design`, Appendix C):

1. start from a scrambled **Sobol** sequence (a low-discrepancy sequence that already fills space evenly);
2. reparametrise each coordinate with a sigmoid (`x = expit(z)`) so the optimiser is unconstrained;
3. minimise `phi_p` with L-BFGS using the analytic gradient, **annealing** `p` from 2 to 50 over 5 stages;
4. the continuous optimum piles points onto the cube's faces, which would starve the middle levels after binning; so
   each column is **rank-transformed** (Latin-hypercube marginals: every level gets the same count);
5. **refine** with within-column level swaps that lower `phi_p` while keeping every level's count exactly.

Results at 486 runs: log `phi_p` 0.263 (random 0.670, plain Sobol 0.684), minimum distance 0.83 against 0.53,
perfect marginal balance, full pairwise level coverage. `simple_design` (random), `balanced_design` and
`coverage_report` (proofloop's) and `compare_designs` measure the alternatives on the same factors;
`design_quality` reports log `phi_p`, minimum distance, marginal balance (worst chi-square p) and pair coverage;
`discretize` / `encode` map between the cube and levels.

**Composition** (`compose`). Scenarios are bases × design rows. `cross`: every base with every row. `embedded`
(default): rows are dealt to bases as a **balanced blocking factor**, in a farthest-point order, so each base sees
conditions as different from each other as possible (`per_base_dispersion`: mean within-base minimum distance 1.40
against 1.11 for a random shuffle).

**The budget** (`power_curve`). How many runs per base are enough? A Monte Carlo simulation plants one level of a
3-level factor that multiplies the odds of failure by 3, simulates outcomes at a base failure rate of 15%, runs the
same likelihood-ratio test the attribution uses (Bonferroni-adjusted over all factors), and counts how often the
planted effect is detected. Six runs per base gives power 0.93.

## factors.py — journeys of a compact graph

```
cases/
└── factors.py   ◀   PART 1 mining journeys from the state machine (mine_paths, perturbations, GraphSimulator ...)
                     PART 2 base-case factors (load, check, generate, closure, review, draft, apply_edits)
```

For a compact pack, cases can also be **mined** from the state machine directly (the older graph campaign path, still
used by the studio's review and by tests).

### Part 1: mining

* `mine_paths(graph, max_deviations)`: every path through the machine with at most N deviations from the default
  (the default is the happy path: each prompt's first accepted act with every required slot at its first value,
  `default_frame`). The compact interpreter runs in explore mode and branches each time it raises `NeedChoice`;
* `perturbations`: failure situations over every prompt state (a person asked for, an unclear answer, a policy
  question, the turn limit), each on the first mined path that reaches that prompt;
* `coverage_closure`: paths with more deviations, kept only when they reach a new workflow edge, rule or transfer
  reason;
* `behaviour_key`: a hash of everything observable in a trace, so two paths with the same behaviour are kept once;
* `build_fixture` (the bank for a path), `loop_items` (items of a returned list), `kb_base` (the default path with
  one policy question), `deviation_kind`, `category_of`, `tags_of`, `to_item`;
* `GraphSimulator`: the customer of a mined journey; `parse_frame` / `frame_text` convert meaning cards to and from
  `ACT(slot=VALUE)`.

### Part 2: base-case factors

`base_factors.yaml` lets the tester choose the dimensions while the graph decides the truth. `generate` builds a
**covering array of strength t** over its factors (relevance-aware: a tuple counts only when every level in it is
relevant, and each non-default level is paired with the *enablers* that make it relevant), then compacts it;
`closure` adds cases from the oracle's exploration for every state, edge, rule or transfer reason the array misses;
`review` lists, for a person to read, what each level leads to and where a played case departs from a level's
`expect` block (`outcome`, `expect_problems`, `representative`); `draft` writes a starting file from the graph's
decision points; `apply_edits` changes it (keeping `.bak`). `python -m gmsjourney.cases.factors` runs draft, check or
review from a shell.
