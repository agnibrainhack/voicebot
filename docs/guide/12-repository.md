# 12. Around the package: llm.py, progress.py, tests, tools, notebooks, root files

```
voicebot/
├── README.md  DESIGN.md  NOTICE         ◀ root files
├── pyproject.toml  requirements.txt  requirements-llm.txt  run_campaign.py
├── gmsjourney/
│   ├── __init__.py                      the package overview
│   ├── llm.py                           ◀ the shared model client
│   └── progress.py                      ◀ progress of the long parts, for a terminal or the console
├── tests/                               ◀ 17 test files and 4 helpers
├── tools/                               ◀ documentation builders and diagnostics
│   └── templates/
├── notebooks/                           ◀ the two notebooks
└── docs/                                ◀ this guide and generated walkthroughs
    └── guide/
```

## gmsjourney/llm.py — the model client

```
gmsjourney/
└── llm.py   ◀   LLMSettings, GatewayChat, make_chat_model, CachedLLM, LLMStats, LLMUnavailable, parse_json_object
```

One client is shared by every part that calls a language model: the LLM rewording arm (chapter 5), the LLM bot
(chapter 6), the optional speech judge (chapter 7) and the narrator (chapter 9).

**`GatewayChat`** posts OpenAI-style chat completions to the company's model gateway with `requests`:

* the bearer token is a short-lived `gcloud` credential (`LLMSettings.token_cmd`), refreshed on 401/403;
  `AEXP_ACCESS_TOKEN` bypasses the CLI; the token is never stored, logged or reported;
* the `x-aip-usecase-id` header comes from `use_case_id` (`AEXP_USE_CASE_ID` overrides);
* the deployment is `model` (`gpt-41`, `gpt-55` ...); the endpoint is `GMSJOURNEY_LLM_URL` when set, else
  `{base_url}/models/{model}/chat/completions`;
* TLS goes through the corporate CA bundle (`GMSJOURNEY_CA_BUNDLE`, `REQUESTS_CA_BUNDLE`, else the system bundle);
* if a deployment rejects `temperature` / `max_tokens` or the system role (HTTP 400), the client adapts once and
  records it; 429 and 5xx back off and retry.

`make_chat_model(settings)` returns the client or raises `LLMUnavailable` with the reason.

**`CachedLLM`** puts a **content-addressed replay cache** in front of it: every response is stored under the SHA-256 of
(prompt version, provider, model, temperature, messages, salt). Analogy: a **recipe book indexed by the exact list of
ingredients**: the same ingredients always open the same page. So a campaign re-runs byte for byte with no network
(`mode="replay"`; a miss in replay mode raises `LLMUnavailable`). The cache holds model outputs only: prompts are
hashed, not stored. `LLMStats` counts calls, cache hits and misses, errors, tokens and seconds. `parse_json_object`
finds the first balanced `{...}` in a reply, tolerating code fences and prose around it. Every call is also counted
in `progress` (from the cache, to the model, or failed, by cache tag), which is how the console shows a run's model
calls as they happen.

## gmsjourney/progress.py — how far the long parts have got

```
gmsjourney/
└── progress.py   ◀   bar / Bar, phase, StepCounter, llm_call, llm_flush, set_sink, report
```

Every long loop reports here and does not care who listens. Analogy: a **departures board**: each flight posts its
own status (boarding 40 of 130); the board flies nothing, and a terminal with no board still hears the announcements.

* `bar(total, desc, show=...)`: a counted loop (cases played, proofs, tool-fault runs, planted defects): a tqdm bar on a
  terminal when `show`, and a report to the sink when one is installed. `s07_checks/scoring.progress_bar` and the LLM
  proof's `progress` both return one.
* `phase(key, i, n, label, phases=..., expect_s=...)`: a sequence of named phases: the knowledge store's build reports
  its eight phases (`s03_knowledge/build.GMS_PHASES`), each with how long it took in the last build on this machine
  (`timings_s` of the newest `gms_build_report.json`) so the console can estimate the time left.
* `StepCounter`: counts optimizer steps through torch's global post-step hook while knowlytix trains (its loops cannot
  be instrumented from outside), so a long phase visibly moves.
* `llm_call(tag, outcome, seconds, tokens)`: one LLM call; totals by tag are reported at most twice a second, and
  `llm_flush()` sends them at once (at every step change).
* `set_sink(fn)`: the console's worker installs one that writes every report as a `progress` event; with no sink and
  no terminal bar a report costs nothing. A report never raises.

## tests/

Run any file directly (`python tests/test_names.py`) or with pytest. Tests that need knowlytix say so and are skipped
without it; the rest run anywhere.

| file | proves |
|---|---|
| `test_names.py` | core code spells no graph name as a string literal; roles resolve in the graph; a misspelt name fails with a suggestion |
| `test_project.py` | the project file is checked against the graph, editable in the studio, and a new level needs no code |
| `test_pack_compile.py` | a compact graph compiled into the full graph walks the same journeys as its own interpreter (85 and 276 journeys, 0 different) |
| `test_graph_engine.py` | the graph-agnostic engine: compact graphs, mining, the store, rule and contradiction calibration from the graph |
| `test_loops.py` | loops over a returned list in journey generation (item count, window edges, per-item answers) |
| `test_graph_tools.py` | opening a graph by path, its sections and core segments, the editor's versions, the GMS sphere, the LLM bot's tool factors |
| `test_app.py` | the console's API: journeys, devices, check, preview, save, a run that outlives the server that started it |
| `test_console_features.py` | the console's bot-under-test choice, the case explorer and the playground; progress reports from bars, phases and LLM calls; saved Playground conversations replayed and flagged; two runs compared case by case; the graph path preview |
| `test_flow_parity.py` | the graph-walking oracle reproduces the hand-written one it replaced (needs the store) |
| `test_scenario_parity.py` | the configured journey factors reproduce the hand-written ones they replaced |
| `test_selfchecks.py` | one self-check per layer (the book ends each chapter with assertions) |
| `test_speech.py` | the speech reader, session facts, verifier and controls |
| `test_policy_speech.py` | amounts and policy in what the bot says, on every journey |
| `test_bot_validation.py` | the bot validation page and the response cards it reads |
| `test_llm_sut.py` | the LLM bot with a scripted fake model (no network) |
| `test_pack_llm_arm.py` | the LLM rewording arm on a compiled pack, with a scripted stand-in model |
| `test_parity_and_llm_arm.py` | hallucination probes, the LLM arm's separate proof, gmsh-parity additions |

Helpers: `_store.py` (the store every test runs on), `fake_sut_model.py` (a scripted stand-in model for the LLM bot),
`legacy_oracle.py` and `legacy_journey.py` (the hand-written oracle and journey code the graph-driven versions
replaced, kept only for the parity tests).

## tools/

| file | does |
|---|---|
| `build_docs.py` | builds the four pages in `docs/` from a campaign run (`--campaign output/campaign_out`) |
| `answer_sheet.py` | `ANSWER_SHEET.html`: every base case with its customer turns, meaning cards, required acts and tool calls |
| `build_annotated.py` | `ANNOTATED.html`: an annotated listing of `s04_cases/meaning.py`, `s02_graph/flow.py` and the frame reader, with a captured trace. Its line notes were written against an earlier version of those files, so rebuild it only after updating them |
| `case_comparison.py` | rebuilds `case_comparison.html` for an existing campaign folder |
| `graph_showcase.py` | the state-transition graph of a policy graph file as one HTML page |
| `geode_before_after.py` | measures what GEODE itself learned, on held-out inputs |
| `check_llm.py` | one tiny call through the model gateway: "will it work here?" |
| `llm_rejections.py` | why LLM-written turns were rejected, from an LLM-arm export |
| `convert_charge_review_v2.py` | converts the charge-review full graph into a compact graph (how `charge_review_graph` was made) |
| `templates/` | `EXPLAINER.html`, `INVARIANCE.html` (filled with a run's numbers by `build_docs.py`), `annotated_trace.txt` |

## notebooks/

* **`charge_journey_campaign.ipynb`**: the full method step by step, on charge review: build the store, mine the
  policy questions and journeys, compare designs, reword and prove, run the reference bot, attribute, probe resilience,
  mutation analysis, governance calibration, then the optional sections (any journey from a pack; the LLM arm; inspect
  a case as seed, rewording and head / relation / tail; what the bot said, by factor and response by response; the
  four documents in `docs/`).
* **`journey_console.ipynb`**: start the console (`from gmsjourney import s10_console as console; console.show()`), follow runs from
  the notebook, run the same campaign from code, stop the console.

Both import only public names from the segments (for example `from gmsjourney.s08_campaign.pipeline import CampaignConfig,
run_campaign`, `from gmsjourney.s07_checks import diagnosis`).

## Root files

| file | holds |
|---|---|
| `README.md` | the quick start, results on the reference bot, what comes from where, the console, testing your own bot, outputs, the layout |
| `DESIGN.md` | the design notes: what was wrong with the original notebook and every decision since, with its reason and measured effect |
| `NOTICE` | the libraries used and their licences (knowlytix-core under its EULA, not included; proofloop and forgeloop Apache-2.0; material from *Beyond Ship and Pray*) |
| `pyproject.toml` | the package: dependencies, optional extras (`test`, `pdf`, `llm`), the data files shipped (`policy/*`, the catalogs, the console pages) |
| `requirements.txt`, `requirements-llm.txt` | the dependencies (Python 3.12, knowlytix 1.5.0, proofloop, forgeloop; `requests` and `tqdm` for the LLM parts) |
| `run_campaign.py` | the command line: `python run_campaign.py --domain supplementary_card --out output/sc --max-cases 20`; `--llm` adds the LLM arm, `--llm-mode replay` uses the cache only, `--speech-gate`, `--decision-rule quorum`, and every knowlytix build setting |

## docs/

* `guide/`: this guide;
* `EXPLAINER.html`, `INVARIANCE.html`, `ANSWER_SHEET.html`, `ANNOTATED.html`: generated walkthroughs of one run
  (`tools/build_docs.py`);
* `POLICY_GRAPH_SHOWCASE.html` (`tools/graph_showcase.py`), `GEODE_BEFORE_AFTER.html` / `.json`
  (`tools/geode_before_after.py`).

Everything under `docs/` except `guide/` and everything under `output/` is generated; a clean checkout is
`gmsjourney/ notebooks/ tests/ tools/ docs/guide/ *.py *.md *.txt *.toml NOTICE`.
