# `gmsjourney/packs/` — Packs

A journey pack is one folder under `gmsjourney/policy/<name>/`: its policy graph, `domain.yaml` and `project.yaml`. This folder loads a pack, makes it the active one, checks that its files agree with its graph, gives code the graph's names (`N.state.X`, tools by role) and drafts missing wording and test plans.

The full explanation, with the theory and diagrams, is chapter 1 of the guide: [docs/guide/01-packs.md](../../docs/guide/01-packs.md).

```
gmsjourney/packs/
├── __init__.py  Journey packs: what one journey is made of, and how it is loaded and checked
├── drafting.py  First drafts of a pack's wording and test plan, written from its compact graph
├── loading.py   Loading a journey pack, and checking that its configuration agrees with its policy graph
├── names.py     Every name the policy graph defines, and the tools core code finds by role
└── project.py   The project configuration (``gmsjourney/policy/<pack>/project.yaml``): the choices the policy graph cannot make
```

## `__init__.py`

Journey packs: what one journey is made of, and how it is loaded and checked.

## `drafting.py`

First drafts of a pack's wording and test plan, written from its compact graph.

Parts:

* PART 1 - Draft a customer lexicon for a compact policy graph (schema gmsjourney.policy_graph/2).
* PART 2 - A starting project file for any domain pack, drafted from its graph.

| name | kind | what it is |
|---|---|---|
| `draft_lexicon` | def | A lexicon for PolicyGraph ``g``, ready to drop under ``lexicon:`` in domain.yaml. |
| `fill_lexicon` | def | ``lex`` with what the graph has and it lacks drafted in: a value, a slot's member, an act it has no entry for, the policy-question wording. Entries ``lex`` already has are never changed. Returns the lexicon and the names filled in (so an edited graph keeps running while its wording is written). |
| `draft_bot_says` | def | The reference bot's line for every state that speaks, from the graph: a prompt asks for what its acts carry, a confirmation reads back its fields and asks to go ahead, a line states every field it carries. Every field a state must say is in its line, so the speech check finds it. |
| `fill_bot_says` | def | ``says`` with a drafted line for every speaking state it has none for (lines it has are kept). |
| `tool_about` | def | A tool factor's description: what varies (the tool's answer), with the graph's own text about the tool. A text copied from a procedure table ('tool / when used / returns / rule') is read as its columns. |
| `draft` | def | A first ``project.yaml`` for a pack, written from its compact graph alone: the factors (customer behaviours, values, wording), their levels with where each applies, and the default design settings. |
| `main` | def | Command line: print (or ``--write``) a drafted ``project.yaml`` for a pack. |

## `loading.py`

Loading a journey pack, and checking that its configuration agrees with its policy graph.

Parts:

* PART 1 - Domain packs: one policy graph plus one ``domain.yaml`` per bot under test.
* PART 2 - Cross-checks between a domain pack's config and its policy graph (run by ``domain.lint``).

| name | kind | what it is |
|---|---|---|
| `DomainError` | class | A domain pack is missing, malformed, or does not match its graph. |
| `Domain` | class | One loaded domain pack: its folder, ``domain.yaml`` settings, the graph (compact or compiled) and helpers to read sections, limits and file paths. |
| `load` | def | A domain by name (``charge_review``), folder, graph file or config file; default: $GMSJOURNEY_DOMAIN, else ``charge_review``. |
| `current` | def | The domain every component reads (loaded on first use). |
| `set_current` | def | Make a pack the active one (by name, folder, graph or config file). Everything that reads the active pack (graph names, project, lexicon, tools) reads the new one from here on. |
| `on_switch` | def | Register a cache reset to run whenever the active pack changes. |
| `lint` | def | Every problem found between the config and its graph (empty = clean). |
| `main` | def | Command line: lint a domain pack against its policy graph (exit 1 when there are problems). |
| `check` | def | Register a lint check (a function ``(domain, graph) -> list of problems``). |
| `patterns_only` | def | A check of a hand-written section; for a graph pack (engine graph) that section is mined from the graph, so the check does not apply. |
| `lexicon_matches_graph` | def | Lint: every lexicon slot is a value set of the graph with exactly that set's members, and the aliasing levels and depths are consistent. |
| `lexicon_round_trip` | def | Every value phrase and act template, at every aliasing level and paraphrase depth, must read back as the meaning it was generated from. |
| `tools_match_graph` | def | Lint: every graph tool has an entry (and only graph tools), each with a valid role and arguments. |
| `figures_match_graph` | def | Lint: every limit and named figure exists in the graph and has a value. |
| `fixtures_match_graph` | def | Lint: the mock bank's fixtures (card options, items, records) use values the graph knows. |
| `rules_match_graph` | def | Lint: every rule entry names a graph rule, a known rule kind, and tools and limits that exist. |
| `no_boolean_keys` | def | YAML reads unquoted yes / no / on / off as booleans: as a key or a level that is always a mistake. |
| `flow_matches_graph` | def | Lint: every state, act, rule, tool and figure named in the flow section exists in the graph. |
| `journey_is_consistent` | def | Lint: every journey dimension has a default and its levels are consistent. |

## `names.py`

Every name the policy graph defines, and the tools core code finds by role.

Parts:

* PART 1 - The domain's nomenclature, read once from the policy graph and used everywhere else.
* PART 2 - Tools and values that core code refers to by role, resolved in the active pack's graph on every access.

| name | kind | what it is |
|---|---|---|
| `Names` | class | Attribute access to a set of names defined by the graph. |
| `Tool` | class | One graph tool as the code sees it: stage, success and failure statuses, what happens on failure, arguments and returned fields. |
| `Rule` | class | One graph rule: id, stance, text, source and the fact it states. |
| `Trigger` | class | One graph trigger: a condition (with the rule it serves) checked in the listed states. |
| `GraphNames` | class | Every name the policy graph defines (states, acts, slots, value sets, aliases, tools, rules, limits, triggers ...), read once from the graph; the registry behind ``N`` and the roles. |
| `graph_names` | def | The names of the active domain pack's graph, read once per pack. |

## `project.py`

The project configuration (``gmsjourney/policy/<pack>/project.yaml``): the choices the policy graph cannot make.

| name | kind | what it is |
|---|---|---|
| `parse_frame` | def | ``ACT(slot=value, ...)`` -> a meaning card (``$credential`` is the fixture's credential). |
| `Level` | class | One level of a factor: the fixture and customer behaviour it sets, plus its case-building hints (boundary value, number of questions, target turn, exact count, transfer to avoid or reach). |
| `Factor` | class | One factor of the project: its name, what it is about, default level, levels, when it is relevant, its role and the decision point (tool or state) it is about. |
| `Project` | class | A journey's ``project.yaml``: the policy graph, the base fixture and customer, the factors, constraints, categories, generator settings and tags. |
| `dump_yaml` | def | YAML text for a project dict: keys in order, no anchors, unicode kept, 120 columns. |
| `default_path` | def | ``GMSJOURNEY_PROJECT``, else the active domain pack's project.yaml. |
| `pack_of` | def | The domain pack a project file belongs to: its folder when that is a pack, else the active pack. |
| `project` | def | The project in use (``GMSJOURNEY_PROJECT``, else the active pack's project.yaml), cached. |
| `reset_cache` | def | Forget loaded projects (after a project file changed). |
| `CaseView` | class | What a factor expression can ask about one case. |
| `fixture_keys` | def | Mock-backend fixture key -> (tool, what its value is), read from the pack's tool specs (domain.yaml -> tools, or mined from a compact graph): ``statuses`` the results of the 1st, 2nd, ... call, ``status`` one status, ``items`` the list a read returns, ``call`` the call number that fails. |
| `loop_fixture_keys` | def | The bank settings of a pack that loops over a returned list (its first loop): ``items`` how many, ``last_item_age`` the age of the last one, ``extra_item_age`` one more after them (outside the window). |
| `check` | def | Every problem in the project file, read against the policy graph's domain. |
| `derived` | def | What the graph suggests as factors and levels (shown next to the configured ones in the studio). |
| `main` | def | Command line: ``check`` a project against its graph, or ``show`` it. |
