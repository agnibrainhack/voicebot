# `gmsjourney/graph/` — The policy graph

The policy as data, in two formats (full triples, compact state machine), the compiler between them, and the oracle: the code that computes the correct behaviour by walking the graph.

The full explanation, with the theory and diagrams, is chapter 2 of the guide: [docs/guide/02-graph.md](../../docs/guide/02-graph.md).

```
gmsjourney/graph/
├── __init__.py     The policy graph: its two formats, the compiler between them, and the oracle that walks it
├── compact.py      The compact policy graph (schema ``gmsjourney.policy_graph/2``)
├── compile.py      Compiling a compact policy graph (format 2) into everything the engine reads
├── flow.py         The oracle: what the bot should do, computed by walking the policy graph's flow
├── interpreter.py  The oracle for a compact policy graph: an interpreter over its state machine
└── triples.py      The full policy graph: the operating procedure as head / relation / tail triples (format 1)
```

## `__init__.py`

The policy graph: its two formats, the compiler between them, and the oracle that walks it.

## `compact.py`

The compact policy graph (schema ``gmsjourney.policy_graph/2``).

| name | kind | what it is |
|---|---|---|
| `is_v2` | def | Is this file a compact policy graph (schema gmsjourney.policy_graph/2)? |
| `attr_relation` | def | The graph relation an attribute is stored under (requires_x / is_x keep their name). |
| `PolicyGraph` | class | A compact policy graph in memory: values, acts with slot specs, tools, rules, limits, transfers, the always-available behaviours and the states, with helpers to read them (members, aliases, kind of a state, prompt states, loops ...). |
| `to_triples` | def | The graph as head / relation / tail triples (the PolicyKG shape every reader uses). |
| `statement_slug` | def | The object of a rule that states no triple of its own: its wording, as an entity name. |
| `rule_reification` | def | (subject, predicate, object): the fact a rule states, which the contradiction primitive compares (two rules contradict only when they state the same fact with opposite stances). |
| `graph_kg` | def | A PolicyKG over the expanded triples, so every reader of the graph works unchanged. |
| `derive_config` | def | The domain sections a campaign reads, mined from the graph. domain.yaml is laid on top. |
| `derive_rule_checks` | def | Runtime call checks (checks/runtime.py templates) and speech checks from the graph's machine-checkable rules. |
| `deep_merge` | def | ``over`` laid on ``base``: nested dicts merged key by key, everything else replaced. |
| `dump_compact` | def | A compact policy graph in its reading layout: one line per limit, value, act, tool, rule and state. |

## `compile.py`

Compiling a compact policy graph (format 2) into everything the engine reads.

Parts:

* PART 1 - Compile a compact policy graph (``gmsjourney.policy_graph/2``) into the full policy graph every component reads.
* PART 2 - Compile a compact policy graph into the symbolic store and the GMS view knowlytix ingests.

| name | kind | what it is |
|---|---|---|
| `PackCompileError` | class | A compact graph cannot be compiled into the full graph as written. |
| `compile_pack` | def | The full policy graph (a PolicyKG) of a compact policy graph. |
| `compile_graph` | def | The symbolic store for a v2 graph, plus the self-correction log (geode.BuildResult). |
| `prose_candidates` | def | Figures the knowledge-base sentences state, bound to the member they name (naive binding: the member mentioned last before the figure), as candidate facts for the critics. |
| `yes_no_words` | def | (column, word for yes, word for no): the GEODE ingest reads yes/no columns as pass/fail flags, so a yes/no attribute is written as words. |
| `render_gms_view` | def | The compiled policy (a GMSStore from compile_graph) as GEODE input. |
| `render_policy_markdown` | def | A readable rendering of the graph (the policy as a document). |
| `attribute_phrasings` | def | Per-relation policy-question phrasings for the u-channel contradiction SFT, rendered by the pack's realizer from the graph's figures (labelled by construction). |

## `flow.py`

The oracle: what the bot should do, computed by walking the policy graph's flow.

Parts:

* PART 1 - The journey, executed from the policy graph.
* PART 2 - The policy oracle: expected behaviour, computed rather than annotated (Ch. 12.4).

| name | kind | what it is |
|---|---|---|
| `FlowError` | class | The graph's flow specification cannot be executed as written. |
| `parse` | def | Parse and vet one expression (cached). ``functions``: the names it may call (default: the flow's). |
| `evaluate` | def | Evaluate an expression against a context whose ``f_<name>`` methods are the callable functions. |
| `Transition` | class | One way out of a state: on which customer act, under which guard, doing what, saying what, exercising which rules; plus the turn-boundary controls. |
| `FlowSpec` | class | The executable flow read from the graph: states, prompt states, transitions out of each state, what each act carries, what each tool is called with; every name an expression uses is checked to exist. |
| `Context` | class | Everything an expression can see or change during one walk. |
| `GraphOracle` | class | Expected behaviour, computed by walking the policy graph (drop-in for ``PolicyOracle``). |
| `Limits` | class | Every limit the policy graph declares (``(key, is_a, limit)``), read exactly from the store's ENM. |

## `interpreter.py`

The oracle for a compact policy graph: an interpreter over its state machine.

| name | kind | what it is |
|---|---|---|
| `NeedChoice` | class | Raised in explore mode when the run needs a decision the script does not make yet. |
| `GraphOracle` | class | A compact graph's own interpreter: walks its state machine directly (used to mine journeys and to check that the compiled flow walks every journey the same way). |

## `triples.py`

The full policy graph: the operating procedure as head / relation / tail triples (format 1).

Parts:

* PART 1 - The dialog flow of the charge-review procedure, written as data for the policy graph.
* PART 2 - The policy as one clean head / relation / tail graph, with dialog states (book Ch. 2-4).

| name | kind | what it is |
|---|---|---|
| `Row` | class | One transition of the charge-review dialog, written as data: from, to, on which act, condition, evidence, rules, workflow edges and the expressions the oracle executes. |
| `Act` | class | What a bot act carries in the trace: field -> expression (optional: left out when None). |
| `rows` | def | Every transition of the charge-review dialog (the input of the graph's state layer). |
| `StateSpec` | class | One dialog state of the charge-review procedure: stage, whether it prompts, what it accepts, how it reads a bare yes or no, its rules, limits, next tools and triggers. |
| `Said` | class | One piece of information a bot act tells the customer. |
| `SpeechSpec` | class | What one bot act (a dialog state) says, why, and the words that signal it. |
| `pdf_text` | def | Layout-preserving text (the tools table keeps its columns). |
| `pdf_sections` | def | Split the procedure into sections pdf.1 ... pdf.7 (section 2 into pdf.2A-2F). |
| `best_evidence` | def | The sentence of the section that shares the most content words with `query`. |
| `build_kg` | def | Build the JSON graph from the PDF (evidence) and its structured transcription (facts). |
| `PolicyKG` | class | Read access to the JSON graph; every later step goes through this class. |
| `resolve_policy` | def | A policy path the compiler can read, plus the graph when the source is the JSON. |
| `main` | def | Command line: build charge review's graph from the PDF, or check a graph file. |
