# 11. Data formats: the pack files, the catalogs, and every output

```
voicebot/
├── gmsjourney/
│   ├── policy/                         ◀ the packs (this chapter, part 1)
│   │   ├── charge_review/
│   │   │   ├── charge_review_kg.json         the full graph (1,066 triples), built from the PDF
│   │   │   ├── charge_review_policy.md       the structured transcription of the PDF
│   │   │   ├── domain.yaml                   lexicon, tools, fixtures, figures, rules, journey, flow
│   │   │   ├── project.yaml                  the test plan
│   │   │   ├── application.yaml              which catalog factors apply (and why not)
│   │   │   └── custom_factors.yaml           the two voice factors
│   │   ├── charge_review_graph/              the same journey as a compact graph
│   │   │   ├── charge_review_kg.json  domain.yaml  project.yaml  base_factors.yaml
│   │   │   └── application.yaml  custom_factors.yaml
│   │   └── supplementary_card/               a second journey, compact
│   │       ├── supplementary_card_kg.json  domain.yaml  project.yaml  base_factors.yaml
│   │       └── application.yaml  custom_factors.yaml
│   └── rewording/catalogs/official/    ◀ the catalogs (part 2)
│       ├── base_catalog.yaml  factor_catalog.yaml  profiles.yaml  PROVENANCE.md
└── output/                             ◀ the outputs (part 3; generated, not in git)
    ├── campaign_out/                   a campaign's out_dir
    └── console_runs/<id>/              runs started from the console
```

## Part 1: the pack files

### The policy graph (`*_kg.json`)

Two formats, explained in chapter 2:

* **full graph** (charge_review): `{"schema", "title", "source", "relations", "triples", "sha256"}`; each triple has
  `id`, `head`, `relation`, `tail`, `tail_type`, `kind`, `source`, `evidence`, `gms`, and for transitions `qualifiers`
  (`on`, `when`, `guard`, `do`, `say`, `rules`, `edges`, `discard`, `retract` ...);
* **compact graph** (`"schema": "gmsjourney.policy_graph/2"`): `limits`, `values`, `acts`, `tools`, `rules`,
  `transfers`, `global`, `states`, `knowledge`, `opening`, `credential_hint`. Edited versions are saved beside the
  original as `<name>.v2.json`, `.v3` ... by the console.

### The structured transcription (`charge_review_policy.md`)

The operating procedure rewritten as tables and numbered rule lines, every row with an id and a `source` (a PDF section
or `program` for limits the business owner added), so each extracted fact keeps its provenance. Sections: objective,
limits, delivery options, tools, rules, workflow, knowledge base, required language. `graph/triples.py` builds the
JSON graph from it and the PDF, and a self-check proves both compile to identical facts.

### `domain.yaml`

| section | holds | example |
|---|---|---|
| `domain` | id, title, graph file, transcription, channel, language | `graph: supplementary_card_kg.json` |
| `lexicon` | how customers say each value and act (aliasing levels, paraphrase depths, cues, wrappers, protected words, reading order) | `SPOUSE: {say: {canonical: my spouse, synonym: my wife, colloquial: the missus}, cues: [...]}` |
| `tools` | each tool's role, behaviour, arguments, key arguments, fixture field, default results, how it checks and echoes, its plausible-wrong fault, and its API text for the LLM bot | `authenticate_customer: {role: auth, behaviour: attempts, check: {...}}` |
| `fixtures` | the bank's credential, items and options | |
| `figures` | which entities are limits, which relations are figures, named figures | |
| `rules` | runtime call checks, completion checks, speech checks, spoken figures, the attestor | `- {check: max_calls, tool: authenticate_customer, limit: max_auth_attempts, rule: R4}` |
| `journey` | dimensions, defaults, relevance, padding, boundaries (charge review) | |
| `flow` | the opening, turn limit, handoff, question, human, clarify (charge review) | |
| `bot` | (compact packs) the reference bot's line for every state, and value wordings | `DISCLOSE_FEE: "{card_product} has an annual fee of {fee}."` |

For a compact pack, `tools`, `figures`, `rules`, `flow` and `journey` are mined from the graph, so its `domain.yaml`
holds only `domain`, `lexicon`, `fixtures` and `bot`.

### `project.yaml`

Sections `project` (name, graph), `fixture`, `customer` (`answers`, `scripts`, `questions`), `factors` (a list; each
with `name`, `about`, `default`, `anchor`, `relevant_when`, `levels`), `constraints`, `categories`, `tags`,
`generator`, `evaluation` (`severity`, `tier_escalation`, `release_gate`), `speech` (`policy`, `also_allowed`,
`signals`). Chapter 1 shows a factor in full.

### `application.yaml` and `custom_factors.yaml`

`application.yaml` names the application, the custom factor files, and for **every** catalog factor whether it is
`selected` (with any level vocabulary override and `level_aliases`), `excluded` (with a reason) or `moved_to_base`
(with a reason), plus the base categories used and the `profiles` (`voice_full`, `quick_screen`).
`custom_factors.yaml` adds `disfluency` (`none, fillers, restart`) and `self_correction` (`none, false_start`) in the
official schema, so the catalog contract applies to them unchanged.

### `base_factors.yaml` (compact packs)

Base-case factors for the older graph campaign path: dimensions with levels written as scripts in the graph's terms,
and `expect` blocks that say what a level should lead to (chapter 4, `cases/factors.py`).

## Part 2: the official catalogs

Copied byte for byte from the companion repository of *Beyond Ship and Pray* (`PROVENANCE.md` lists each file's
source path and SHA-256).

| file | defines | the rule |
|---|---|---|
| `base_catalog.yaml` | **what** is asked and what is true: 18 base question types, each with a family, the GMS primitive that computes its truth, the graph capabilities it requires, and applicability | base → defines the ground truth |
| `factor_catalog.yaml` | **how** a question is presented: 40 factors in ten groups (query, persona, reasoning, context, entity, conversation, adversarial, system, variability, robustness), each with type, cardinality, categories, `gt_invariant: true` (enforced: a factor that could change the truth is not a factor) and `applies_to` (families and answer types it enriches) | factor → preserves the ground truth |
| `profiles.yaml` | bundles of bases and factor groups; `mode: cross` (every base × every design row) or `embedded` (bases become a blocking factor of one design) | |

## Part 3: the outputs

### A campaign's `out_dir` (default `output/campaign_out/`)

| file | content |
|---|---|
| `report.html` / `report.json` | the campaign report / every number in it |
| `verdict.html` / `verdict.json` | the one-page verdict (chapter 9) |
| `bot_validation.html` | did the right thing / said it right, split by rewording, category and factors, with response cards |
| `case_comparison.html` | every case, expected beside actual, turn by turn, with the speech verdict |
| `dataset/test_cases.csv`, `dataset/test_cases.jsonl` | one row per test case (base and reworded): category, policy question, canonicals, meaning cards, base wording, rewording, expected outcome, complete expected trajectory, proof verdict |
| `dataset/test_turns.csv` | one row per customer turn: state, stage, the state's rules, limits and accepted cards, the meaning card and its triples, base and reworded text, read-back card, equivalence |
| `dataset/policy_questions.csv` | the policy questions with their truths |
| `dataset/case_results.jsonl` | every graded run, with `speech_turns`: one response card per bot turn (claims, verdicts, truth, where it was read, the GMS reading) |
| `dataset/turn_by_turn.csv`, `dataset/turn_by_turn_summary.csv` | per customer turn: the oracle's expected tools and acts against the bot's; per case: first divergence and outcome |
| `results_enriched.csv`, `results_base_level.csv` | one row per scenario (`f_<factor>`, `correct`, `c_<stage>`, gate alarms ...) |
| `failures.jsonl`, `scenarios.jsonl` | failed runs with transcripts; every scenario with its proof |
| `gms_build_report.json` | the knowlytix build: GEODE diagnostics, embed loop and SFT evaluations, judge thresholds, ENM agreement, timings |
| `policy.gms.md` | the GEODE input rendered from the policy |
| `gms_canonicals.json` | canonical values, aliases and their binding in the trained store, exact numbers with seals, states, acts, every trained fact with its score and cut |
| `policy_kg.json` | the policy graph the run was built from |
| `ontology.ttl`, `store_graph.dot` | the journey ontology with the invariance evidence (Turtle); the store subgraph (Graphviz) |
| `sft_classifier.jsonl`, `sft_draft.jsonl` | training corpora (proofloop schema) |
| `results_enriched_llm.csv`, `scenarios_llm.jsonl` | the LLM arm: one row per scenario; every LLM scenario with its gates and proof |
| `dataset/llm_enrichment_turns.csv` | every LLM-written turn: base, template and LLM text, both readers' cards, each gate, attempts, factors requested and realised |
| `dataset/llm_negative_controls_audit.csv`, `dataset/arm_pairs.csv` | controls the LLM gate accepted (for a person to audit); template against LLM pairs |
| `sut_llm.json`, `sut_prompt.txt`, `sut_critic_prompt.txt` | (LLM bot) call statistics, reviews and corrections; the bot's prompt; the reviewer's prompt |

### A console run (`output/console_runs/<id>/`)

| file | content |
|---|---|
| `job.json` | the run's settings (pack, device, sizes, bot, provenance) |
| `events.jsonl` | start (with the worker's pid), step_start / step_done (with the step's summary), log lines, done (with the headline), error |
| `console.log` | everything the worker printed |
| `sut.json` | the LLM bot's prompt and attachments, when one was used |
| `out/` | the campaign's `out_dir` (above) |

### Caches (outside the repository)

* `~/.cache/gmsjourney/stores/<hash>/`: trained stores, keyed by the hash of the policy and build settings;
* the LLM replay cache (`GMSJOURNEY_CACHE` or `llm_cache_dir`): model outputs stored under the SHA-256 of (prompt
  version, provider, model, temperature, messages, salt). Prompts are hashed, never stored; tokens are never written.
