# `gmsjourney/knowledge/` — Knowledge

The knowledge store (GMS): exact figures sealed by hash, typed facts, rules and workflow (symbolic), the trained geometric store (knowlytix), and the true answers to policy questions.

The full explanation, with the theory and diagrams, is chapter 3 of the guide: [docs/guide/03-knowledge.md](../../docs/guide/03-knowledge.md).

```
gmsjourney/knowledge/
├── __init__.py   The knowledge store (GMS) and policy questions
├── build.py      Building the knowledge store: symbolic self-correction, then the trained GMS (knowlytix)
├── questions.py  Policy questions: their true answers, how an answer is graded, and the questions a campaign asks
└── store.py      The symbolic knowledge store: the policy's facts, figures, rules and workflow, in memory
```

## `__init__.py`

The knowledge store (GMS) and policy questions.

## `build.py`

Building the knowledge store: symbolic self-correction, then the trained GMS (knowlytix).

Parts:

* PART 1 - The policy compiler: ingest -> symbolic self-correction (book Ch. 4, the part before training).
* PART 2 - The knowlytix backend: availability, licence and text encoders.
* PART 3 - The GMS built, trained, calibrated and judged by knowlytix (book Ch. 2-4, 8).

| name | kind | what it is |
|---|---|---|
| `Correction` | class | One self-correction decision on a candidate fact (corroborate, drop a conflict, drop unbound, drop a type error, flag), with its reason and source. |
| `BuildResult` | class | The compiled symbolic store and the self-correction record (decisions, iterations, convergence). |
| `build_store` | def | Compile the policy into the symbolic store, with the self-correction log. |
| `to_dot` | def | Graphviz rendering of the store (or of selected relations) - the book draws the small GEODE store as a knowledge graph; render with `dot -Tsvg`. |
| `oracle_property` | def | An oracle, not an index (book Ch. 4 capstone): every ENM-backed fact scores inside its relation's calibrated band, and a cross-tail substitution - another entity's *real* figure placed on the same relation - scores outside it. Also reports the book's absolute bands (true < 0.5, fabricated > 1.0) and knowlytix's per-head calibrated admissibility (``native_admissible_ok``) for comparison. |
| `available` | def | True when the knowlytix family (knowlytix, proofloop, forgeloop) is installed and a licence is configured. |
| `require` | def | Raise a clear error when knowlytix is not installed or not licensed. |
| `describe` | def | What is installed: knowlytix (with versions), proofloop, forgeloop. |
| `guard_deepspeed` | def | If DeepSpeed is installed but cannot be imported (version mismatch with torch), make accelerate and transformers treat it as absent. Returns True when a patch was applied. |
| `local_encoders` | def | (v_path, u_path) when both local model folders exist, else None. |
| `build_offline_encoder` | def | HuggingFace-format hashing encoder (see module docstring). Deterministic. |
| `encoder_kind` | def | 'huggingface' or 'offline-stand-in': what resolve_encoders would pick, without building anything. |
| `resolve_encoders` | def | {'v': name_or_path, 'u': name_or_path, 'kind': 'huggingface' / 'offline-stand-in', 'reason': ...} |
| `device` | def | The compute device: GMSJOURNEY_DEVICE, else the module setting, ``auto`` -> cuda when available. |
| `KXBuild` | class | A trained store and its build report. |
| `store_config` | def | The production store's DocGMSConfig: package defaults, then DEFAULT_STORE_CONFIG, then the caller's dotted-key overrides (``{"train.epochs": 200, "cap.margin": 0.3, "geometry.d_v": 256}``). |
| `build_kx_store` | def | Build the trained GMS with knowlytix (GEODE loop, embed loop, contradiction SFT, production store, judge calibration) into ``store_dir``, and return it with its build report. |
| `load_kx_store` | def | Reload a built store (knowlytix model, ENM, judge calibration, tuned encoders). |
| `CachedKX` | class | Read-through proxy over a GMSExpertStore that memoises the geometric reads. |
| `KXPolicyStore` | class | Symbolic policy facts from the compiler; geometry, ENM and calibration from knowlytix. |

## `questions.py`

Policy questions: their true answers, how an answer is graded, and the questions a campaign asks.

Parts:

* PART 1 - Ground truth and grading for policy (knowledge-base) questions.
* PART 2 - Mining policy-question base cases from the store (book Ch. 5, 8.1.1).
* PART 3 - Policy questions mined from a compact policy graph.

| name | kind | what it is |
|---|---|---|
| `KBTruth` | class | The truth of one policy question: the answer, its type, the store primitive that computes it, a re-derivation, an optional geometric agreement test, and the fact triples that carry it. |
| `kb_answer` | def | Answer from the store's primitives, with the fact triples that carry the answer (for the two-way check) or the reason none exists. |
| `extract_numbers` | def | Every number in a text, in order: digits, number words, and "free" / "no charge" as 0. |
| `money_mentions` | def | Every money amount in a text, in order ($N, N dollars, number words + dollars, free as 0). |
| `anchored_figures` | def | The figures the answer states about what the question asked, read by the speech reader (which ties each figure to the thing named nearest it): "Standard delivery is free, and expedited is $15" answers a question about standard delivery with [0], whatever order the options are named in. None when the reader is not available for this store or the words tie no figure to the asked thing (the grader then reads the first figure, as before). |
| `judge_kb_answer` | def | Return {'correct': bool, 'tier': str, 'detail': str} for one KB answer. |
| `KBItem` | class | One policy-question base case: id, category, question frame, true answer, recovery tag and details. |
| `two_way_check` | def | The admission gate: symbolic recompute AND knowlytix's geometric recovery of the answer triples. |
| `kx_substrate` | def | Mine the knowlytix store with the production answer-key generators (book nb 25): every category under the two-way geometric guard, with the per-question recovery tag and the calibrated cut each generator read. Nothing is admitted silently. |
| `render_truth` | def | The graph's own answer as text (used for the Ch. 4.3 precondition). |
| `mine_kb_items` | def | extended=True adds more questions over the same policy facts (other days around the review window, 'is it free?', multiple choice for every option, more near-miss option names). They are appended after the standard ones, so the standard question ids do not change, and each is kept only if its answer is computed from the graph and its wording reads back to the same meaning card. |
| `answer_type` | def | The type of a figure's answer: ``money`` for an attribute ending in ``_usd``, else ``number``. |
| `kb_frames` | def | The askable questions: every numeric attribute the lexicon has wording for (all of them when it lists none) on every member, and every customer-facing limit. |
| `graph_kb_answer` | def | KBTruth for a graph-domain policy question (an exact register read). |
| `mine_graph_kb_items` | def | Policy-question base cases of a compiled pack: one per question frame of its graph (values and limits), each answered by the store and admitted by the two-way check; returns the items and per-category counts. |
| `store_self_tests` | def | The store must pass these before anything is scored against it (book Ch. 4.3): |

## `store.py`

The symbolic knowledge store: the policy's facts, figures, rules and workflow, in memory.

Parts:

* PART 1 - Ingest: policy markdown -> candidate facts with provenance (book Ch. 4.1).
* PART 2 - The compiled policy store: the symbolic layer every step reads (book Ch. 2).
* PART 3 - The canonical vocabulary as the *trained* GMS holds it (gms_canonicals.json).
* PART 4 - The policy, rendered in the table shape knowlytix's GEODE ingest reads.
* PART 5 - The symbolic store of the active domain pack: compiled straight from its graph in about a second, no GMS

| name | kind | what it is |
|---|---|---|
| `Source` | class | Where a fact came from: section, row/line id and the verbatim span. |
| `CandidateTriple` | class | A fact read from the policy document (head, relation, tail), with where it was read and any attributes; not yet admitted into the store. |
| `CandidateENM` | class | A candidate for the exact numeric register (category, key) -> value. |
| `Rule` | class | One policy rule: its id, stance (must / must_not / may), what it says as subject-predicate-object, its text and source. |
| `WorkflowEdge` | class | One step of the policy's workflow: from one state to another under a condition. |
| `IngestResult` | class | Everything read from the policy document: candidate facts and exact values, rules, workflow, value sets, aliases, entity types, knowledge-base sentences, required language and the raw sections. |
| `canonical_number` | def | Canonical decimal string used as the ENM value ("15", "0", "2.5"). |
| `ingest_markdown` | def | Read a policy document (markdown) into candidate facts: limits and fees become exact values and triples, options, rules, workflow and the knowledge-base sentences are read section by section. |
| `extract_prose_candidates` | def | Second reading of the same figures from free text (knowledge-base sentences, rule and limit texts), each bound to the nearest option it mentions; used to corroborate or contradict the structured reading. |
| `IntegrityError` | class | Raised when an ENM value no longer matches its integrity hash. |
| `ENMEntry` | class | One exact value (category, key, value, unit) with its SHA-256 digest and the sources it was read from: the store's tamper-evident record of a figure. |
| `GMSStore` | class | The symbolic policy store: admitted triples with provenance, exact values, rules, workflow, value sets and aliases, with the lookups the cases and checks use (value of a limit, fee of an option, rules about a thing ...). |
| `export_canonicals` | def | The store's canonical answers and scores as a JSON-ready dict (each fact's score and cut-off, each figure, each rule), tied to the graph's SHA-256. |
| `render_gms_markdown` | def | Render the compiled policy (a GMSStore built by knowledge/build.py) as GEODE input. |
| `kx_relation` | def | Our compiled relation name -> the relation the knowlytix ingest produces for it. |
| `kx_tail` | def | A tail written the way the trained store expects it (yes/no -> required/not_required, numbers with one decimal, text lower-cased and single-spaced). |
| `kx_triple` | def | Map a fact in our vocabulary onto the knowlytix store's vocabulary. |
| `symbolic_store` | def | The store compiled from a policy graph (default: the active pack's), with ``store.kg`` the full graph and a symbolic stance check (the claimed stance equals the rule's, or not; the GMS reads tension instead). |
| `pack_store` | def | The active pack's symbolic store, built once per pack. |
| `pack_oracle` | def | The graph oracle on the active pack's symbolic store (cached with it). |
