# `gmsjourney/s10_console/` — Console

The local web app: open a journey by its graph's path, edit its project, run campaigns, explore cases, talk to the bot.

The full explanation, with the theory and diagrams, is chapter 10 of the guide: [docs/guide/10-console.md](../../docs/guide/10-console.md).

```
gmsjourney/s10_console/
├── __init__.py     The journey console: a local web app to configure a journey and run its test campaign
├── __main__.py     python -m gmsjourney.s10_console [--port 8790] [--host 127.0.0.1] [--runs DIR]
├── bots.py         The bot under test as the console configures it, and the playground
├── graphs.py       The graph pages of the console: open by path, sections, core segments, editor, sphere
├── runs.py         Campaign runs from the console: start, follow, summarise, read back
├── server.py       The journey console's HTTP server: one page (ui.html) and a small JSON API
├── studio.py       The project studio: a local page to read the policy graph and edit the project configuration
├── studio_ui.html  The standalone studio page
└── ui.html         The console page (one self-contained HTML file)
```

## `__init__.py`

The journey console: a local web app to configure a journey and run its test campaign.

| name | kind | what it is |
|---|---|---|
| `url_for` | def | The URL a browser should open (see the module docstring for the order). |
| `show` | def | Start the console (if it is not running yet) and display it in the notebook, with a link to open it in its own browser tab (it has more room there). Returns the embedded frame. |

## `__main__.py`

python -m gmsjourney.s10_console [--port 8790] [--host 127.0.0.1] [--runs DIR]

| name | kind | what it is |
|---|---|---|
| `main` | def | Command line: serve the console until interrupted. |

## `bots.py`

The bot under test as the console configures it, and the playground.

Parts:

* PART 1 - Text from a file a tester attaches to the LLM bot's prompt (an SOP, a policy PDF, an FAQ sheet).
* PART 2 - The bot under test, as the console configures it: the reference bot of the journey, or an LLM bot on the model
* PART 3 - The playground: talk to the bot under test yourself, one turn at a time, with every turn checked as you go.

| name | kind | what it is |
|---|---|---|
| `extract` | def | Text from an attached file (text, Markdown, JSON, YAML, CSV, PDF, Word, Excel), capped in size, or an error that says why the file cannot be used. |
| `suggested` | def | The suggested prompt for the active journey, and the fixed part the harness adds. |
| `critique_on` | def | The critique layer is on unless the job turns it off. |
| `settings_for` | def | The model-client settings of a console job's ``sut`` block. |
| `make_factory` | def | (bot_factory or None for the reference bot, display name, the LLM factory or None). |
| `check_gateway` | def | One tiny call through the gateway (tools/check_llm.py in one function): ok, endpoint, seconds, the reply or the exact error. A configured stand-in model (GMSJOURNEY_SUT_FAKE_MODEL) is reported instead of calling out. |
| `Bench` | class | The active journey's store, oracle, reader, realizer and speech verifier (built once per journey). |
| `bench` | def | The active journey's bench (call with the journey active). |
| `reset` | def | Forget the bench of ``pack`` (every bench if None): its graph changed. |
| `factor_cases` | def | Without a run: the default case, then one case per non-default factor level (as Preview shows them). |
| `run_cases` | def | The generated cases of a finished run, rebuilt exactly (case_specs.jsonl) or from report.json and kept only where the rebuilt case reproduces the run's expected outcome. Reworded versions come from the run's dataset. |
| `cases` | def | ``source``: "project", or a run id of the active journey ("latest": its newest finished run, else project). |
| `find_case` | def | One case of the loaded case list, by source and id. |
| `Session` | class | One playground conversation: the case, the bot, the mock bank, the speech checker and every turn so far. |
| `start` | def | A new conversation on a case with the reference bot (``sut`` None or kind "reference") or the LLM bot. |
| `get` | def | A playground conversation by id (most recently used are kept). |

## `graphs.py`

The graph pages of the console: open by path, sections, core segments, editor, sphere.

Parts:

* PART 1 - Open a policy graph by path, show it section by section, check its core segments and save edited versions.
* PART 2 - The GMS on the unit sphere: every entity of the policy graph as a point on S^2, every relation as an arc.

| name | kind | what it is |
|---|---|---|
| `resolve` | def | A path as typed in the console (relative to the repository, ~ expanded) -> an absolute path. |
| `rel` | def | A path relative to the repository when it is inside it, else as is. |
| `set_graph` | def | Point the journey at another graph file in its folder (only domain.yaml's ``graph:`` line changes). |
| `new_journey` | def | A journey folder for a compact graph from outside the journey folders. |
| `check_wording` | def | Play the active journey's project cases with its reference bot and the speech check: how many turns, how many the graph agrees with, and every bot line whose words fail the check (with an example). A drafted journey is checked this way when it is made. |
| `open_graph` | def | Open ``path``: the journey that holds it (switched to that file), or a new journey made from it. |
| `versions` | def | The graph files in the journey's folder that belong to the same graph (the original and its versions). |
| `set_roles` | def | Who uses each value set: "customer" (a customer act carries it), "bot" (only tools return or take it), or "both" (a customer act carries it and a tool takes it as an argument, by the argument's name: card_action is passed as action). |
| `shared_slots` | def | Value sets (and the credential) a customer act carries and a tool takes as an argument. |
| `classify` | def | Every triple of the compiled graph into one section: customer, bot, both, flow, rules, knowledge, other (by relation, by who uses the value set involved, and by the type of the head entity). |
| `sections` | def | The graph section by section: counts, relations, entities and (up to ``cap``) the triples of each. |
| `core_segments` | def | What every policy graph needs for the validations a campaign runs, and what is missing. |
| `compact_items` | def | A compact graph's own entries, one per value set, act, tool, state, rule, limit (``values.relationship`` ...), each with its section. |
| `editable` | def | The editor's view of the active graph: compact entries or triple rows, with versions and sections. |
| `apply_items` | def | Editor entries back into a compact graph document. |
| `diff` | def | What changed between two graph documents: added, removed and changed entries (or triples by id). |
| `check_candidate` | def | Load ``doc`` as the journey's graph in a scratch copy of the journey folder and lint it there. |
| `build_doc` | def | The edited graph document from the editor's request (compact entries, or triples with a fresh hash). |
| `save_version` | def | Save an edited graph as ``<name>.vN.json`` beside the original (after it loads and lints) and switch the pack to it. |
| `spectral_sphere` | def | Place the entities on the unit sphere from the graph alone: the 2nd-4th eigenvectors of the normalised Laplacian, then refined on the sphere (linked entities attract, every pair repels). |
| `store_vectors` | def | The entity vectors a built knowlytix store saved (v_emb.pt): names and a matrix. |
| `pca_sphere` | def | Unit vectors -> their first three principal components, normalised onto the unit sphere. |
| `build` | def | The sphere view of the active pack: nodes with positions, section, type and degree; edges with relation and section; which placement was used and why; how close linked entities sit compared with random pairs. |

## `runs.py`

Campaign runs from the console: start, follow, summarise, read back.

Parts:

* PART 1 - What each campaign step produced, in one shape the console renders: headline metrics, tables and notes.
* PART 2 - The generated cases of a finished run, for the console's case explorer (step 11's data, read back).
* PART 3 - The console's campaign runs: start one in its own process, read its events, list past runs, cancel.
* PART 4 - One campaign in its own process: ``python -m gmsjourney.s10_console.runs <run_dir>``.

| name | kind | what it is |
|---|---|---|
| `done_steps` | def | The steps whose results are in ``R``. |
| `summarize` | def | What one campaign step produced (metrics, tables, notes), from the campaign's result dict; a summary never breaks the run. |
| `headline` | def | The run's results at a glance (for the run list). |
| `clean` | def | Drop the None tables the summaries skip. |
| `load` | def | A finished run's cases, proofs and results, read from its output folder (cached by file time). |
| `index` | def | The case explorer's list: every base case with its reworded versions, wording factors and results. |
| `detail` | def | One base case in full: each version's turns (original and reworded words, the meaning read back, the proof's verdict) and the bot's result with its turn-by-turn comparison. |
| `devices` | def | The compute devices this machine offers: cpu, and each CUDA GPU torch can see (with its name and memory). |
| `check_device` | def | None when the device can be used, else why not. |
| `Runs` | class | The console's campaign runs: each starts as its own process (``python -m gmsjourney.s10_console.runs <dir>``), one at a time; events are read from ``events.jsonl``; a run outlives the server that started it. |
| `main` | def | The worker: run the campaign of ``<run_dir>/job.json`` and append its events to ``events.jsonl``. |

## `server.py`

The journey console's HTTP server: one page (ui.html) and a small JSON API.

| name | kind | what it is |
|---|---|---|
| `packs` | def | Every journey under gmsjourney/policy/: its name, title, graph and whether it has a project yet. |
| `make_server` | def | The console's HTTP server: the page (ui.html) and the JSON API listed at the top of this file. |
| `free_port` | def | ``port`` if nothing listens there, else the next free one (another kernel may hold the console's port). |
| `start` | def | Start the console in a background thread (from Jupyter or Python); returns its URL. Running it again returns the same server; if another process holds ``port``, the next free port is used (and said). |
| `running` | def | The consoles this process serves: port -> URL. |
| `stop` | def | Stop the console (the one on ``port``, or every one this process started). Runs in progress keep going. |

## `studio.py`

The project studio: a local page to read the policy graph and edit the project configuration.

| name | kind | what it is |
|---|---|---|
| `to_yaml` | def | The project as YAML, a short note above each section (comments in a hand-edited file are not kept). |
| `Studio` | class | One journey's project in the console: the graph's view (states, tools, acts, rules, limits), the project, checks as you edit, coverage, a Preview of any level, and lossless save. |
| `level_site` | def | Where a level acts: the prompts whose customer answer it changes, the tools whose answer it sets (else the factor's decision point). Empty when the project does not say (factors written with relevance expressions). |
| `reach` | def | The levels to play so that the call gets to where ``name = lv`` acts. With every other factor at its default the call can miss it (a person asked for at the retry prompt, which the default call never reaches); then the first other level, or pair of levels, that takes the call there is added. Returns (levels, "other = level" or None when nothing needed adding); "" when no combination within the budget gets there. ``run(spec)`` plays a JourneySpec and returns its trace. |
| `normalize` | def | Undo what JSON does to the project in the browser: level names like 0, 1, 2 come back as strings. |
| `frame_text` | def | A meaning card as the config writes it: ACT(slot=value, ...). |
| `make_server` | def | The studio's HTTP server (not started). |
| `serve` | def | Serve until Ctrl-C (command line). |
| `start` | def | Start the studio in a background thread (for a notebook) and return its URL. Starting again on the same port replaces the running one (switching the pack it edits). |
| `stop` | def | Stop a studio started with ``start``. |
| `main` | def | Command line: serve the standalone project studio for one pack. |
