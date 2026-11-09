# `gmsjourney/rewording/` — Rewording

Saying each customer line in other words (templates or an LLM) and proving, scenario by scenario, that the meaning did not change.

The full explanation, with the theory and diagrams, is chapter 5 of the guide: [docs/guide/05-rewording.md](../../docs/guide/05-rewording.md).

```
gmsjourney/rewording/
├── __init__.py         Rewording customer turns, and proving that the meaning did not change
├── catalog.py          The records a campaign passes around, and the enrichment factor catalogs
├── llm_arm.py          The LLM rewording arm: a language model writes customer turns, under gates
├── llm_pack.py         The LLM enrichment arm for any pack compiled from a compact graph: prompts, schema and checks read from the pack
├── llm_proof.py        The LLM arm's separate proof, and the function that wires the arm
├── proof.py            Proof that a reworded case is the same test as its original
├── wording.py          How a customer turn is worded, and how a worded turn is read back
└── catalogs/official/  the domain-neutral factor and base catalogs (data; guide chapter 11)
```

## `__init__.py`

Rewording customer turns, and proving that the meaning did not change.

## `catalog.py`

The records a campaign passes around, and the enrichment factor catalogs.

Parts:

* PART 1 - Core records shared across the pipeline (names follow the book's API, App. A).
* PART 2 - Catalog loading, relevance resolution and validation (book Ch. 5-6, App. A).

| name | kind | what it is |
|---|---|---|
| `QAItem` | class | One base case: content + ground truth (+ per-component truth). |
| `Scenario` | class | A base case under one assignment of presentation factor levels. |
| `SUTResult` | class | What a system under test returned for one case: answer, components, trajectory, status and escalation. |
| `BaseSpec` | class | One base question type (official schema) plus how it reaches this bot. |
| `FactorSpec` | class | One wording factor of the catalog: group, levels, whether the truth stays the same under it, which base families and answer types it applies to. |
| `Profile` | class | A named selection of bases and factors with a mode (cross or embedded). |
| `official_dir` | def | The official catalogs: from the installed ``proofloop`` package (proofloop/suite/catalogs, the same files beyond-ship-and-pray ships) when present, else the vendored copy. |
| `Catalog` | class | The catalog of bases, wording factors, groups and profiles, plus the application file that selects, excludes or moves factors for this journey. |
| `ResolvedSuite` | class | The bases and factors a run will use after the application's choices, with the factors dropped and why. |
| `resolve` | def | Bases and factors named (or groups of them) resolved against the catalog and its application file. |
| `resolve_profile` | def | A profile's bases and factors, resolved. |
| `validate_official` | def | The official contract (validate_catalogs.py rules 1-5 + profile references), applied to the official catalogs together with the application's custom factors. |
| `validate_application` | def | Problems with the application file: every factor selected, excluded or moved, nothing unknown or twice. |
| `validate` | def | Return a list of contract violations (empty = valid). |
| `level_provenance_table` | def | One row per selected factor: app levels -> catalog levels, and unrealised levels. |

## `llm_arm.py`

The LLM rewording arm: a language model writes customer turns, under gates.

Parts:

* PART 1 - Prompts for the LLM enrichment arm (rephraser, independent reader, adversary).
* PART 2 - LLM realizer: meaning card + factor levels -> customer utterance, written by the LLM (Microsoft gateway).
* PART 3 - A second, independent reader for LLM-written utterances (dual read).
* PART 4 - Per-turn gates for LLM-written customer utterances.
* PART 5 - CSV exports for the LLM enrichment arm (one row per customer turn, controls audit, arm comparison).

| name | kind | what it is |
|---|---|---|
| `state_question` | def | What the agent just asked in a state, as a sentence (the context the LLM rewrites the customer's line in). |
| `describe_meaning` | def | The meaning card in plain words (no ground truth: what the customer *says*, not what is right). |
| `protected_tokens` | def | Strings the utterance must contain verbatim (decision-carrying values and names). |
| `factor_instructions` | def | One instruction per non-base factor level that applies to this turn. |
| `build_rephrase_prompt` | def | The system and user prompts asking the LLM to reword one customer line: the meaning to keep, the wording factors to apply, the tokens to keep as written and feedback from a failed attempt. |
| `extraction_schema` | def | The closed meaning-card schema the independent reader must fill (from the ontology). |
| `build_extract_prompt` | def | The prompts asking an independent LLM to read a customer line back into a meaning card (the dual read). |
| `build_adversary_prompt` | def | The prompts asking an LLM to argue that a line could mean something other than its target meaning. |
| `LLMTurn` | class | One LLM-written turn: its seed, raw and final text, the template and base texts, the parsed reply, the facts it claims, and whether it fell back to the template. |
| `LLMRealizer` | class | Rewords customer turns with the LLM: each line is generated, passed through the output gates and read back; a failed line is retried, and falls back to the template wording when it keeps failing. |
| `LLMFrameExtractor` | class | The dual read: an LLM reads a customer line back into a meaning card, constrained to the pack's acts and slots. |
| `GateReport` | class | Whether a line passed the output gates, and the reasons it did not. |
| `LLMOutputGates` | class | The checks every LLM line must pass: no new figures, no tool or merchant names it was not given, the protected tokens kept, nothing the agent says, and the rule reader reads back the target meaning. |
| `write_llm_exports` | def | Write the LLM arm's export files (per-turn records, gate and control results, paired comparisons) into a run folder and return their summary. |

## `llm_pack.py`

The LLM enrichment arm for any pack compiled from a compact graph: prompts, schema and checks read from the pack.

| name | kind | what it is |
|---|---|---|
| `active` | def | The active pack when it is compiled from a compact graph, else None (charge review's arm applies). |
| `label` | def | A value, a limit or an attribute in words. |
| `reader_lexicon` | def | value -> the words the rule reader binds to it (canonical first). |
| `state_question` | def | The agent's line the customer answers: the reference bot's wording of the prompt. |
| `describe_meaning` | def | The meaning in plain words for a pack act (None: a global act the charge arm already describes). |
| `describe_kb` | def | A policy question in words ("what the fee of expedited is", the label of a limit ...). |
| `slot_values` | def | The text values a meaning card carries. |
| `act_schema` | def | act -> slot -> (value set, required), the global acts and the policy question included. |
| `extraction_schema` | def | Every act's slots and allowed values for the dual read, from the pack's graph (with the kb intents, attributes, entities and limits). |
| `kb_intents_text` | def | Short descriptions of the pack's policy-question intents, for the extraction prompt. |
| `gates_one_decision` | def | One clear decision: per slot, at most one value cued (two with a self-correction); no slot cue the meaning does not carry. |

## `llm_proof.py`

The LLM arm's separate proof, and the function that wires the arm.

Parts:

* PART 1 - Separate invariance validation for LLM-written enrichments.
* PART 2 - The LLM enrichment arm: a model on the Microsoft gateway writes the customer turns; every turn is

| name | kind | what it is |
|---|---|---|
| `progress` | def | A tqdm bar (notebook-aware via tqdm.auto) when tqdm is installed and show is True, else a silent stand-in with the same interface (iteration, update, set_postfix_str, close). |
| `LLMInvarianceValidator` | class | The invariance proof for LLM-worded cases: the template proof plus the dual read and the adversary, retries, and a per-turn fallback to the proved template wording. |
| `build_llm_arm` | def | Wire the arm: one cached client per role so the roles never share completions. |

## `proof.py`

Proof that a reworded case is the same test as its original.

Parts:

* PART 1 - Surface similarity of two wordings: a reported diagnostic of the invariance proof.
* PART 2 - Proof that every enriched case is the same test as its base case.
* PART 3 - The invariance proof for a compact policy graph (rewording/proof.py, layers L1-L4 unchanged).

| name | kind | what it is |
|---|---|---|
| `normalize_text` | def | Text lower-cased with punctuation (other than $ . % ') removed and spaces collapsed. |
| `HashingEncoder` | class | Signed feature-hashing encoder with a numeric block. |
| `surface_cosine` | def | Cosine similarity of two wordings (1.0 = same surface). |
| `gt_hash` | def | A short SHA-256 of a ground truth (answer and components), to show two truths are identical. |
| `ProofResult` | class | The invariance proof of one reworded case: accepted or not, and why (truth identical, every meaning read back equal, the oracle re-derived the same truth), with its signatures, turns and retries. |
| `InvarianceValidator` | class | Proves a reworded case means the same as its base case: each reworded turn is read back into its meaning card and compared with the base turn's, and the oracle re-derives the truth from the read-back meanings. |
| `GraphInvarianceValidator` | class | The invariance proof for graph-mined cases: the base path is the graph oracle's own walk of the case. |

## `wording.py`

How a customer turn is worded, and how a worded turn is read back.

Parts:

* PART 1 - The domain lexicon: how customers say things and how their words are read back.
* PART 2 - Surface-noise operators shared by calibration and the realizers.
* PART 3 - Realizers: semantic frame + factor levels -> customer utterance (book Ch. 6.3.3).
* PART 4 - The frame reader: utterance -> semantic frame, grounded in the ontology.

| name | kind | what it is |
|---|---|---|
| `cap` | def | The text with its first letter upper-cased. |
| `low` | def | The text with its first letter lower-cased (kept as is when it starts with "I " or "I'"). |
| `fill` | def | str.format with a capitalised twin of every string field ({phrase} -> {Phrase}). |
| `cue_regex` | def | A cue list as one word-bounded alternation, the form the reader matches. |
| `Lexicon` | class | The pack's lexicon section: how each value, act and figure may be said, read once per pack, with the patterns the reader matches. |
| `typo` | def | Keyboard-style typos (swap, drop, neighbour key) inside words of 5+ letters. |
| `asr_noise` | def | Speech-recognition confusions: homophones and lowercasing. |
| `disfluency` | def | Voice disfluencies: fillers between words, or a restarted phrase. |
| `number_to_words` | def | A number below 1000 in words ("one hundred twenty five"). |
| `spell_digits` | def | A code digit by digit in words ("1 2 3 4" -> "one two three four"). |
| `Realizer` | class | Deterministic, seed-stable realisation of frames under factor levels. Every phrase, template and wrapper comes from the domain lexicon (domain.yaml -> lexicon); this class only applies them. |
| `FrameReader` | class | The rule reader: reads a customer line back into a meaning card (act and slots) using the lexicon's patterns, the store's aliases and the state's expectations. |
