# gmsjourney — GMS-grounded testing for the charge-review voice bot

The method of *Beyond "Ship and Pray": Testing Agentic Systems with Geometric Ground Truth*, applied to the
charge-review journey in the operating procedure. The GMS core (ingest, GEODE self-correction, store training,
embedding fine-tuning, calibration, judging, design and attribution engines) comes from the **knowlytix**
package family, and the book's open companion packages **proofloop** and **forgeloop**
([beyond-ship-and-pray](https://github.com/knowlytix/beyond-ship-and-pray)) supply catalogs, emitters, runtime
gates and the audit chain. `gmsjourney` adds what those packages do not cover for a multi-turn voice bot: the
journey oracle, covering-array bases, the realiser/reader and invariance proof, runtime policy checks, the
reference bot and its mutants. knowlytix, proofloop and forgeloop are required (`requirements.txt`, Python 3.12,
a knowlytix licence); there is no built-in fallback.

```
policy PDF -> policy graph JSON (head/relation/tail + dialog states + per-state policy; policy/charge_review_kg.json)
   -> GMS [knowlytix]: GEODE loop -> embed loop (v SFT) -> contradiction SFT (u) -> cap-trained store -> GMSJudge calibration
   -> base cases: 27 policy questions (knowlytix's two-way generator gate, else a stated symbolic check)
                  + 4 hallucination probes (near-miss option names, an unstated fee) + journeys (covering array + closure)
   -> enrichment: official catalogs [proofloop] + application layer, 16 voice factors, phi_p design;
                  two arms on the same design rows: template realiser (always) and LLM (--llm, transport chosen at run time)
   -> invariance proof: every enriched scenario shown to be the same test as its base case - the LLM arm
                  separately, behind extra per-turn gates and its own negative controls
   -> runtime gates calibrated (Appendix C) [forgeloop GMS gates + audit on the knowlytix store]
   -> run the bot under the runtime monitor (strict or quorum rule, numeric range attestor, per-turn latency /
      length budgets); KB answers judged exactly against the register
   -> diagnosis: deviance tests per factor + per component / weak link, severity + risk tiers + release gate,
      template-vs-LLM paired comparison, resilience, mutation analysis
   -> training corpora [proofloop emitters]
   -> exports: gms_canonicals.json (canonicals as the trained GMS holds them) + dataset/*.csv (every test case)
```

## The policy graph (the source every step reads)

`gmsjourney/policy/charge_review_kg.json` is the operating procedure as 1066 head / relation / tail triples. Each one
names its PDF section (`pdf.2C`, ...) and carries an evidence quote that is a verbatim substring of that section;
`program` marks the two limits from the business brief and `lexicon` the curated customer synonyms (not in the PDF).

| kind | what it holds | example |
|---|---|---|
| trajectory | the tool workflow and the dialog-state machine | `ASK_AUTH leads_to PRESENT_TXN` (on PROVIDE_CREDENTIAL, when AUTHENTICATED and transactions returned, R3/R6, W3) |
| state | what holds in each of the 19 dialog states | `PRESENT_TXN accepts_act INFORM_DISPOSITION`, `PRESENT_TXN applies_rule R30`, `PRESENT_TXN limited_by lookback_days`, `PRESENT_TXN interprets_yes_as RECOGNIZED` |
| policy | rules, limits, delivery options, tools, escalation triggers, knowledge base, required opening | `R27 has_stance must_not`, `max_turns has_value 20` |
| vocabulary | value sets, aliases, dialog acts and their slots | `UNRECOGNIZED has_alias "not me"`, `INFORM_DISPOSITION has_slot delivery` (values from `delivery`) |
| domain | the domain model | `customer has_session review_session` |

Rebuild it after changing the transcription or the state layer:
`python -m gmsjourney.s02_graph.triples build --pdf fictional_poc_operating_procedure_gpt_readable.pdf`
(`check` validates a file). `PolicyKG.load()` is the read API; the GMS is trained on the graph, state layer
included, and the campaign checks that the trained store carries every fact (`report.json: policy_kg.consistency`).

**State-aware meaning cards.** Every customer turn is read in the dialog state it answers. The ontology resolves
slot values inside the slot's value set, reads a bare "yes"/"no" the way the state reads it (RECOGNIZED /
UNRECOGNIZED after "do you recognise this charge?", CONFIRM / DENY after "shall I go ahead?", END after "anything
else?"), treats a 4-digit number as a credential only where the state asks for one, and rejects a card the state
does not accept. The invariance proof compares (state, card) sequences.

## Any journey: domain packs

A journey is a **domain pack**, a folder under `gmsjourney/policy/<pack>/`:

| file | what |
|---|---|
| the policy graph | the policy and its dialog. Charge review's is built from the operating-procedure PDF (`charge_review_kg.json`); a new journey can be written as a **compact graph** (`gmsjourney.policy_graph/2`: limits, value sets, acts, tools with roles and statuses, rules, transfers, the always-available behaviours and the state machine; see `supplementary_card_kg.json`), which `s02_graph/compile.py` compiles into the same full graph |
| `project.yaml` | the mock bank, the simulated customer, the journey factors, case rules, generator, scoring and speech settings (below) |
| `domain.yaml` | the customer's wording (lexicon), the tools' mock behaviour, the reference bot's wording (`bot.says`); mined from a compact graph where it can be |
| `application.yaml`, `custom_factors.yaml` | the enrichment catalog layer |

One engine runs every pack: the oracle walks the graph's flow (`s02_graph/flow.py`), the base cases come from the project's
factors, and the scoring, speech check, runtime gates, reference bot and its defects, and LLM arm all read the active
pack (`domain.set_current(pack)`, `CampaignConfig(domain=pack)`, `--domain` on the command line, `--pack` for the studio).

New journey, end to end:

    # 1. write gmsjourney/policy/<pack>/<pack>_kg.json (compact graph) and domain.yaml (lexicon, bot wording)
    python -m gmsjourney.s01_packs.loading <pack>             # lint the pack against its graph
    python -m gmsjourney.s01_packs.drafting <pack> --write    # a project.yaml drafted from the graph
    python -m gmsjourney.s10_console.studio --pack <pack>     # review and edit it (factors, preview, scoring, speech)
    python run_campaign.py --domain <pack>                 # build the GMS once, then the whole campaign

or the notebook's "Any journey" section (cells P1-P6), which opens the studio inside the notebook.
The compiled flow is checked against the compact graph's own interpreter on every mined journey
(`tests/test_pack_compile.py`).

A journey can loop over a list a tool returns ("for each transaction ..."): an `each` state in the compact graph
(see `charge_review_graph`, charge review written as a compact graph). The drafted project then varies how many
items there are (one, two, at the cap, over it), where they sit against the window, and how the customer answers
each one; the studio's Bank page shows the loop and its item records.

What a supplementary-card campaign shows (30 base cases run, each with an enriched variant): every enriched case
proved equivalent (128/128), the reference bot 100% on the base cases with every word grounded, every seeded defect
caught by some suite, and the speech check flagging 95/95 word edits. On the enriched cases the reference bot scores
about a third: by design it reads the customer with keywords only (`s06_bots/reference.py`), so reworded, noisy or
colloquial answers defeat it. That is the harness finding a weak bot, not a scoring fault.

## The graph decides, the project file configures, nothing else names the domain

Three places hold everything domain-specific:

| where | what | read by |
|---|---|---|
| the policy graph | states, acts, slots, value sets, tools and statuses, limits, rules, triggers, and the **flow**: every `leads_to` carries its qualifiers (`on`, `when`, `guard`, `do`, `say`, `rules`, ...) as expressions in a small vetted language; `carries` says what each bot act carries, `has_arg` what each tool is called with | `s02_graph/flow.py` (`GraphOracle`, the expected-behaviour walker), `s01_packs/names.py` (the registry) |
| `gmsjourney/policy/<pack>/project.yaml` | what the policy leaves to the tester: the mock bank (fixture), the simulated customer's scripts, the **journey factors** (each level a patch of bank settings and customer behaviour), case constraints, categories and tags, generator settings, severities and release-gate profiles (`evaluation`), speech-check leniencies and cue words (`speech`) | `s01_packs/project.py`, `s04_cases/journey.py`, `s07_checks/scoring.py`, `s07_checks/speech_*.py` |
| code | how to walk, score and check, never which names | everything else |

**The registry.** `s01_packs/names.py` reads every name from the graph; core code spells them as `N.state.PRESENT_TXN`,
`N.act.CONFIRM`, `N.status.UNAVAILABLE`, `N.limit.max_turns`, `N.rule.R27`, and a name the graph does not define
fails at that line (`PRESENT_TX is not a dialog state in the policy graph (did you mean PRESENT_TXN?)`). Tools are
found by role (`s01_packs/names.py`): `GATE` is the tool account servicing `requires`, `RECORD` the write that takes the
item id, `ACTION` the other write, `OPTIONS` the read before it in `precedes` order, `FEE_OPTION` the delivery
option that `requires_fee_confirmation`. `tests/test_names.py` fails when a core module spells a graph name.

**Journey factors and the oracle.** A factor level is data, not code: it patches the mock bank (`auth_results:
[FAILED, AUTHENTICATED]`) and the customer's scripts (`say: ["UNCLEAR()"]` at a state, before the scripted answer,
once or sticky). The oracle needs no per-factor logic: it walks the graph's flow against that bank and that
customer, so the expected journey of a new level follows from the graph. `relevant_when` says when a factor can
change ground truth (others read N/A); `turn_target` and `transfer`/`avoid_transfer` mark turn-limit levels.

**The studio.** `python -m gmsjourney.s10_console.studio [--project path] [--port 8765]` serves a local page that shows the
graph (state machine, tools and numbers, vocabulary, rules and triggers), what follows from it (names, the
graph's own suggestions for factors, and coverage: tool outcomes no level produces, customer choices no script
makes, transitions and rules no case reaches), and edits the project: journey factors, a preview that walks the
graph for any combination of levels, bank and customer, case rules, scoring and speech. Every edit is checked
against the graph as you type; Save writes the YAML (the previous file is kept as `.bak`).
`python -m gmsjourney.s01_packs.project check` validates a project file from the command line.

**The journey console** (`gmsjourney/s10_console/`) puts the studio and the campaign behind one page, served from Python or
Jupyter: `from gmsjourney import s10_console as console; console.show()` in a notebook (`notebooks/journey_console.ipynb`), or
`python -m gmsjourney.s10_console --port 8790`. The left rail is the route through the work, in five stages:
**1 Understand the policy** (Overview, Graph sections, Flow, Edit graph, GMS sphere), **2 Set up the tests** (Journey
factors, Bank and customer, Rules and scoring), **3 Try the bot** (Playground), **4 Put the bot to test** (Start a test run) and **5 Read the
results** (Test runs, with each run's verdict and cases). In the console a campaign is called a test run; in code it is
still `run_campaign`. A stage is ticked when its work is done, every page says which
stage it belongs to and ends with a link to the next page, and Overview shows the next getting-started step. The
theme switch in the top bar picks Day (blue on white), Night (blue on black) or Auto (follow the system); the
studio shares the setting. Its parts:

- **Graph path** (top bar; `s10_console/graphs.py`): the journey is chosen by the path of its policy graph, by default
  `gmsjourney/policy/charge_review/charge_review_kg.json` (charge review, built from the PDF). A graph inside a
  journey folder opens that journey (and switches it to that file if it is one of its versions). A compact graph
  (`gmsjourney.policy_graph/2`) from anywhere else becomes a **new journey**: copied into
  `gmsjourney/policy/<its id>/` with a `domain.yaml` whose customer lexicon is drafted from the graph's values and
  aliases (`s01_packs/drafting.py`; it passes the lexicon checks for the shipped graphs) and whose reference-bot lines
  (`bot.says`) are drafted from its states: a prompt asks for what its acts carry (listing the choices when there
  are few), a confirmation reads its fields back and asks to go ahead, a line states every field it must say, with
  numbers in their units ("18 years", "7 business days") so the speech check reads them, and optional fields said
  only when they have a value. The enrichment catalog is copied from a template journey and `project.yaml` drafted.
  The new journey is then checked by playing every project case with the reference bot and the speech check, and
  the page reports any line that fails (for both shipped compact graphs, with every bot line drafted, every turn of
  every case passes and follows the graph). A triples graph needs a full `domain.yaml` written for it,
  so it opens only from a journey folder that has one. Recent paths are offered as you type.
- **Journey**: an overview that sets what the graph decides (states, acts, tools, value sets, limits, rules,
  triggers, transitions) beside what the project configures, the coverage gaps, and the flow.
- **Graph sections**: every triple the engine reads (the compiled graph, so both formats look alike) in seven
  sections: used by the customer (acts, values and their words), used by the bot (prompts, lines, tools, their
  arguments, results and statuses), used by both (the acts each prompt accepts, yes/no readings, the value sets that
  pass from the customer's words into tool arguments), defining the flow (transitions, stages, handoff triggers),
  rules, knowledge and figures, and other; with the entities and relations of each, and a filterable triple table.
  Beside it, **Core segments**: the 18 things every graph needs for the validations a campaign runs (opening,
  caller verification and its attempt limit, customer acts with worded values, prompts, tools with success and
  failure results, confirmation before acting, an echo check, justified handoffs, asking for a person, unclear
  answers, a turn limit, policy questions, rules each checked somewhere, figures with units, reachability, the
  bot's wording, the customer's wording), each with what it is needed for and what is missing.
- **Edit graph**: the graph's own entries by section - for a compact graph one JSON entry per value set, act, tool,
  state, rule and limit; for a triples graph one row per triple. A missing core segment can be added as a draft
  with its placeholders to fill. **Check** lints the edited graph in a scratch copy of the journey and lists what
  changed; **Save as new version** writes `<name>.v2.json` (then `.v3`, ...) beside the original, switches the
  journey to it (only `domain.yaml`'s `graph:` line changes) and shows the changes. The original file is never
  written; the versions row switches back. A value, act or slot member the lexicon does not word yet, or a state
  with no bot line, runs on drafted wording (reported on Graph sections) until you write it. "What each state says"
  is advisory for compact graphs: the speech check reads what each state must say from the graph itself.
- **GMS sphere** (`s10_console/graphs.py`): every entity as a point on the unit sphere and every relation as a great-circle
  arc, coloured by section. When a test run with knowlytix has built or loaded this journey's store, the points are
  the **trained store's own entity points**, placed so that the angles between them on this sphere match their angular
  distances on the store's hypersphere as closely as possible (spherical MDS; the page states the angle correlation);
  otherwise from the graph's relationships (a spectral layout refined on the sphere, deterministic). The sphere turns slowly on its own (Pause rotation / Rotate, or space;
  off when the system asks for reduced motion); drag to turn it, with momentum, scroll to zoom, Reset view to go
  back. **Pick a relation** from the side list, the menu, or by clicking an arc: the sphere turns to face it, its
  arcs are drawn with their direction flowing head to tail, and the panel shows what the relation means (the
  graph's own relation table), how many triples and entities it has, its section, its pattern (which kinds of
  entity it links), the relations connected to it through the same entities (click one to move to it) and every
  triple with its id and source. Click a point for its relations, grouped by relation; the page states how close
  linked entities sit against random pairs. **Spread** pushes bunched points apart along the surface while keeping
  linked entities close (default 0.85 for a graph-only layout); **Full view** gives the sphere the whole window.
- **Getting around**: Go to (Ctrl/⌘ K) searches pages, journeys, factors, states, recent test runs and common actions; a running test run
  shows its progress in the tab title and says when it finishes (a system notification if the tab is hidden and
  allowed); stage 2 shows *unsaved edits* when the project has unsaved changes; Ctrl/⌘ Z undoes project edits (Shift
  to redo); typing a graph path previews what opening it would do (title, size, states, tools, rules, which journey).
- **Progress**: a test run shows, inside the running step, the knowledge store's training phase by phase (with the time
  left estimated from the last build and a live count of optimizer steps), bars with the time left for cases played,
  proofs, tool-fault runs and planted defects, and every LLM call split into cache, model and failed, by caller.
- **The answer sheet**: once a run's cases and rewordings are proved (step 5), **Open the answer sheet** shows every
  generated case turn by turn: what the customer says and means, what a correct bot must do (tool calls, then acts),
  the state's rules, how the call must end, and each reworded version with its proof verdict (`out/answer_sheet.html`,
  `s09_reports/answer_sheet.py`).
- **Two LLM judges** (Checks on the run page, on by default; and in the Playground side panel): two LLM judges read
  every run (and every Playground turn) beside the rule-based checks, which still decide. Judge A compares the bot
  with the graph's expected steps; judge B sees only the conversation and the policy rules. Each run shows both
  verdicts and whether they agree with the rules; the runs where they disagree are listed for review
  (`s07_checks/llm_judges.py`, `dataset/llm_judges.jsonl`).
- **LLM selection, separately for each use**: the rewording LLM (under *LLM rewording of customer turns*), the judges'
  LLM (under *Two LLM judges*, and in the Playground's side panel) and the bot under test (its own card) are each the
  Microsoft gateway (a deployment) or a local model on vLLM (a folder loaded in the process, or a vLLM server URL);
  sent with every run (`llm_backend` ..., `judges_backend` ...).
- **gpt-oss-20b on vLLM** (`notebooks/gpt_oss_20b_vllm.ipynb`): starts the model in a vLLM server on this machine and
  makes it the console's rewording and/or judges' model (and, if you like, the bot under test).
- **Reusing meaning proofs** (Advanced on the run page, on by default): step 5 takes a reworded case's proof from an
  earlier run when the case, the journey's files and the code are unchanged (`s05_rewording/proof_cache.py`).
- **Validation report**: `bot_validation.html` shows, per run, the rule-based verdict with its deterministic details
  beside both judges' verdicts and reasons, the agreement rates, and the runs where they disagree.
- **The customer in a test run**: *Customer follows the base journey* (on by default, under Checks on the run page)
  has each case's customer say its base journey's lines whatever the bot answers, so a wrong turn cannot steer the
  rest of the case; the customer plays every line, even after the bot ends the call early; unchecked, the customer answers the bot's last question instead.
- **Saved conversations**: *Save as a test case* in the Playground keeps a conversation's customer lines with the
  journey (`<pack>/saved_conversations.json`); replay one with any bot, and every test run replays them all against its
  bot and reports which deviated.
- **Compare test runs** (Test runs › Compare, or *Compare with the previous run*): cases now failing and now passing,
  the headline numbers, release gates and per-stage scores before and after, and a warning when the runs are not like
  for like (another graph version, seed or case set).
- **You configure**: the journey factors (decision point, relevance, default, and per level a bank patch, a
  tool's returns call by call, or what the customer says, with a preview of the expected conversation for any
  level), the bank and the customer's scripts, case rules, categories, severities and the release gate. Every edit
  is checked against the graph as you type; Save writes `project.yaml` and keeps a `.bak`.
- **Campaign**: run settings, including the device (`auto`, `cpu`, `cuda:0`-`cuda:3`; those torch cannot see
  are shown but disabled). Each run is a separate process (`python -m gmsjourney.s10_console.runs`). The page follows
  its 13 steps live and shows what each produced (metrics and tables: the store, the questions, the cases, the
  rewording, the proof, the gates, the bot's results, attribution, tool faults, planted defects, training data,
  the release gate), then links the report. Runs are kept under `output/console_runs/<id>/`.
- **Bot under test** (Start a test run and the Playground, and the **Bot:** switch in the top bar, which always says
  which bot is in use and opens the same choice): the **reference bot** (no LLM: it follows the graph by rules and
  reads the customer by keywords; instant, free, the baseline that checks the test suite) or the **LLM bot** (a
  language model with your prompt and files, with or without a **critique LLM** that reviews every step;
  `s06_bots/llm_bot.py`). The choice is kept per journey and shared by the Playground and test runs; runs, verdicts
  and the case explorer name the bot each run used, with its critique model. For the LLM bot the page shows a suggested prompt (who the bot is,
  how it behaves, the journey's operating procedure: its transcription, or the graph as text) that you edit
  freely, and takes files (text, Markdown, JSON, YAML, CSV, PDF, Word, Excel; `s10_console/bots.py`) that are added
  after it as reference material. The journey's tools with their allowed values, the transfer reasons, the
  dialog acts and the JSON reply format are always appended (read-only on the page), because the harness needs
  them to run the bot's tool calls and read what it says. Model, live or replay, temperature and the tool-call
  budget per turn are set there; **Check gateway** makes one tiny call. The **Use the critique LLM** checkbox,
  right under the bot choice and for any model (gateway or local), turns the critique layer on or off (on by
  default; `GMSJOURNEY_SUT_CRITIQUE=0`, or `SUT_CRITIQUE = False` in the console notebook, starts it unchecked;
  unchecked, the model runs on its own with half the calls). Checked, it makes
  a second call before every step the bot proposes, each tool call and each reply: a reviewer given the same
  instructions approves it or returns a corrected step, which is carried out instead (`LLMBot._review`; about twice
  the calls). **Critic's model** picks the reviewer, apart from the bot's model: *Same as the bot*, a
  *Microsoft gateway* deployment, or a *Local model (vLLM)* (folder, and a server URL; the Playground needs the
  server), whatever the bot itself runs on (`critic_provider` = same | gateway | local, with `critic_model` or
  `critic_local_path` / `critic_vllm_url` in the job). The bot's name says which model critiques; `out/sut_llm.json` counts reviews, approvals and corrections and
  keeps each correction (proposed step, step carried out, reason), `out/sut_critic_prompt.txt` holds the reviewer's
  prompt, and the Playground shows under each turn what the reviewer changed. The prompt and files are stored beside the
  run (`sut.json`, `out/sut_prompt.txt`), the model's call statistics in `out/sut_llm.json`. Step 10 is skipped
  for an LLM bot (it plants errors in copies of the reference bot).
- **Explore cases** (a finished run, from its results or step 11; `s10_console/runs.py`): every generated case with its
  customer turns in the original words next to each reworded version (the wording factors that changed, and per
  turn the meaning card it was read back as, equivalent or not, and its surface similarity), the expected run,
  and, where the case was run, the bot's result with its turn-by-turn comparison against the graph. **Bot Results**
  charts original vs reworded accuracy by category and lists every case run; **Coverage** draws any two journey
  factors as a grid of how many cases combine each pair of levels (click a cell to list them), the rules and
  transfer triggers reached, and the rewording design. All of it is read from the run's files.
- **Verdict**, step 13 of every campaign (`s09_reports/verdict.py`; written as `out/verdict.html` and `out/verdict.json`, from the
  console or from code; **Open the verdict** on a run's results or *Verdict* in the run list, also for runs older
  than step 13):
  one page to judge the bot. Ready to release or not, with each release gate's reasons in words; the headline
  numbers; the top risks (failure codes ranked by severity and by runs hit, each in plain words with an example
  case); and how to reproduce the run (journey, graph file and its hash, code commit, bot, model and prompt hash,
  seed, cases, device, store, package versions; the commit and graph hash are recorded when a run starts). Then
  every case run, failed first: the expected outcome (end, transfer reason, writes) on the left beside the bot's on
  the right, and per customer turn the graph's expected tools and decisions beside the bot's tools, decisions and
  words, the first difference marked and later turns folded. **Print or save as PDF** gives the first page as the
  one-pager and the cases after it, expected and actual still side by side.
- **Playground** (`s10_console/bots.py`): talk to the bot under test yourself. Pick any journey (switching it here
  switches the console's journey too) and a case (from the latest finished test run, any test run, or the project's
  default and one-per-level cases); it sets the mock bank and the case's simulated customer. At every turn choose how
  the customer answers: **stick to the journey** (the base journey's next line, whatever the bot said, so a bot mistake
  does not change the case) or **talk freely** (type anything; the page also shows the journey's next line and whether
  you said the same thing, in which case the journey moves on). Either way every bot turn is validated the
  same way: the oracle replays your turns, each read into its meaning card, and the turn is compared with the
  graph's (tools with their key arguments, decision acts) and its words are speech-checked. A turn that differs is
  flagged **Bot deviated here** (later ones **Still off the journey**), with the reasons, and listed in the side
  panel. Each turn opens to its **answer sheet**: on the customer side the dialog state answered (what it accepts,
  how it reads a bare yes or no, its rules and limits), the meaning read and the card as triples; on the agent side
  the tool calls the graph makes with their results and every graph transition of the turn (state before, trigger,
  condition, state after, rules with their text, the graph triple and its source), beside the speech check of what
  the bot said. It runs on the store compiled from the graph, so it needs no training and no licence; the
  transcript can be downloaded as JSON.
  **A local model as the bot under test** (`s06_bots/llm_bot.py`): in Bot under test choose *LLM bot*, then *Model
  runs on: Local model (vLLM)*, and type the **model folder** (on the machine running the console, e.g. a Llama 3.1
  8B Instruct download with its `config.json` and weights); **Check** says whether it can be used. Leave **vLLM
  server** empty and a test run loads the folder with vLLM in its own process (greedy, temperature 0, replies cached by
  content like the gateway's); give an OpenAI-compatible vLLM server URL and the calls go there instead, the folder
  being the model name the server knows. The Playground takes a local model only through a server, so a chat never
  loads an 8B model into the console's process. `GMSJOURNEY_SUT_LOCAL_MODEL` and `GMSJOURNEY_VLLM_URL` (for example
  `LOCAL_SUT_MODEL` in the notebook's first cell) only pre-fill the two fields. `GMSJOURNEY_VLLM_GPU_UTIL` (default
  0.80), `GMSJOURNEY_VLLM_MAX_LEN` (16384), `GMSJOURNEY_VLLM_TP` (1) and `GMSJOURNEY_VLLM_DTYPE` tune the in-process
  load. The run keeps the folder and server in its `job.json`.
  `GMSJOURNEY_SUT_FAKE_MODEL=module:Class` puts a stand-in model behind the LLM bot (tests, demos without the
  gateway; `tests/fake_sut_model.py`).

`console.show()` finds the right address for your browser. On JupyterHub, or on any Jupyter server with
jupyter-server-proxy (it checks that `<server>/proxy/<port>/` reaches the console), the page is shown through
Jupyter, which works when the kernel runs on a remote GPU machine. Otherwise it uses `http://127.0.0.1:<port>/` and
says which port to forward. `GMSJOURNEY_CONSOLE_URL` overrides the address. If another kernel holds the port, the
next free one is used. The page talks to the server with relative paths, so it works under any proxy prefix.

The page guides a new user:
- **Getting started** on Overview: seven steps that tick themselves off.
- **Help** (or `?`): the workflow and a glossary.
- A dismissible hint on each page.
- Help under every factor field, and a one-line summary of what each level changes.
- **Fix** on each coverage gap: adds the missing level, or creates the factor at that point.
- After a run, **What this means**: both release gates (original and reworded wording), the gap between them, and
  where to look next. Each pipeline step also says what to look for.

A campaign keeps running if the tab closes or the kernel restarts. The console finds it again by its worker's pid
when it next starts, and Cancel stops the worker's whole process group. `CampaignConfig(device=...)`
(or `GMSJOURNEY_DEVICE`) sets the device for the knowledge store; `CampaignConfig(on_step=...)` reports each step.

## Quick start

```bash
# Python 3.12 (knowlytix-core ships cp312 wheels), knowlytix + proofloop + forgeloop, and a runtime licence
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export KNOWLYTIX_LICENSE_KEY=...            # or put it in ~/.knowlytix/license.key; never committed, never logged
python run_campaign.py --out output/campaign_out   # first run builds the GMS (CPU: ~10-45 min), later runs reload it (~2.5 min)
python tests/test_selfchecks.py             # self-checks on the campaign's knowlytix store (built once, then reloaded)
python tests/test_parity_and_llm_arm.py     # LLM arm checks with a fake model: no gateway key, no network
python tests/test_llm_sut.py                # the LLM bot under test, with a fake model

# + the LLM enrichment arm (the Microsoft model gateway; see "Two enrichment arms")
pip install -r requirements-llm.txt
export GMSJOURNEY_LLM_URL="https://gaiapis.aexp.com/gcpapi/<path>/completions"   # exact endpoint
python tools/check_llm.py                    # one call through the gateway: will it work here?
python tools/check_llm.py --bot charge_review_graph   # one real bot step: the deployment's token quota, bot steps a minute
python run_campaign.py --llm --out output/campaign_llm
python run_campaign.py --llm --llm-mode replay --out rerun      # re-run from the cache, no network
```

Colab: upload `gmsjourney.zip`, add a secret named `KNOWLYTIX_LICENSE_KEY`, open
`notebooks/charge_journey_campaign.ipynb`, run top to bottom. Cell 9b inspects any case: set `BASE`, `VARIANT`,
`TURN` to see the seed and enriched text with their canonical frames, and every turn as Head / Relation / Tail
triples (seed vs enriched, plus the store facts grounding each value). The same views are available in code:
`gmsjourney.s09_reports.cases.show_case / compare_case / case_hrt`.

**Encoders.** knowlytix warm-starts and fine-tunes from HuggingFace sentence encoders (all-MiniLM-L6-v2,
nli-mpnet-base-v2). When the hub is unreachable (as in the environment these results were produced in),
`gmsjourney.s03_knowledge.build` builds a deterministic hashing encoder in HuggingFace format so every knowlytix call still
runs; the build report records which encoders were used. With hub access the real encoders are used
automatically, and the embedding numbers below should improve.

## Results on the reference bot (knowlytix backend, trained on the policy graph; offline stand-in encoders)

| | |
|---|---|
| GMS store (knowlytix) | 229 entities, 607 triples (policy + dialog-state layer), 12 ENM entries; GEODE converged in 3 iterations; 12/12 ENM figures agree with the graph; the store carries every graph fact |
| Oracle property | 12/12 facts inside their learned cap radius, every cross-tail substitution outside |
| Embedding SFT (held out) | v: noisy-alias binding 25.4% → 73.7%; u: phrasing → relation routing 80.8% → 88.5% (your run with the real HuggingFace encoders on the previous graph: 45.6% → 81.6% and 98.7% → 100%) |
| Policy-question bases | 27: 17 two-way admitted by knowlytix's gate, 10 symbolic with stated reasons (thresholds, aggregations, absence, stance); forged triple rejected |
| Substrate (knowlytix generators) | exact_recall 12/12 two-way, multi_hop 19/19 two-way, counting 71 symbolic; graph answers its own questions 100% |
| Feasible content-level pairs covered | 523 / 523; rules 32/32 · workflow edges 21/21 · escalation triggers 11/11 |
| Enriched scenarios proved invariant | 486 / 486 (4,428 customer turns, each read in its dialog state); 632 / 632 meaning-changing edits rejected, incl. 40 wrong-state edits |
| Bot accuracy, base cases → enriched | 89.7% → 50.3% (outcome alone 90.8% → 52.0%; the difference is runs whose outcome was right and whose words were not) |
| Factors that drive failure (BH-adjusted; this run used knowlytix DOEAnalyzer, since replaced by our deviance tests) | noise (p_adj 1.6e-10), numeric_format (1.5e-4), disfluency (0.012) |
| Runtime gate (Appendix C) | θ = 0.882 on `enables`: AUC 1.0, 0 false admits, 0 false rejects; probe sweep 15 TP / 27 TN, no errors |
| Runtime monitor | 0 false alarms on 244 fully correct enriched runs; 46 / 277 failed runs flagged live; hash chains verified |
| Seeded defects (21, four of them in the words only) | 100% caught by the suites (covering 85.7% vs random of equal size 52.4%); **21/21 flagged live by the runtime monitor**. Two of them (the opposite answer said back, a summary that leaves one out) are flagged only by `speech_grounding` |
| What the bot said (speech check) | 789 claims on 87 base runs and 3,975 on 521 enriched runs. 85/87 and 501/521 runs fully grounded. Every wrong figure left is a real KB defect of the reference bot ("Expedited delivery costs $0." to "how much more is expedited?", the wrong delivery days). The flow findings are almost all real too: a policy sentence recited to "could you say that again?", a decline acted on after "no wait, yes". The exception is 2 safe re-asks of the fee. 392/392 word-only control edits flagged. knowlytix's AnswerVerifier agrees on 56/56 and 260/260 of the figures it can read |
| Silent failures under injected tool faults | 0 |
| GMS canonicals | 97.4% of aliases bound by the trained GMS to the canonical value the graph states; 90.2% of the 509 trained facts inside their relation's cut |
| Turn by turn (oracle vs bot) | 300 / 567 runs match the oracle at every turn; every wrong outcome diverges, first divergences mostly in PRESENT_TXN (139) and ASK_AUTH (115) |

The stand-in bot is a keyword-NLU state machine written independently of the oracle; its weaknesses are
realistic (no typo or speech-recognition repair, a narrow knowledge-base retriever). Swap in your bot to get the
same report for it.

## What comes from where

| Capability | Source |
|---|---|
| Ingest, GEODE self-correction, store training (cap admissibility), ENM, primitives | knowlytix `knowledge.geode` (`GeodeLoop`, `store_from_triples`), `knowledge.store.GMSExpertStore` |
| Build parameters | knowlytix's reference build (`knowledge.geode.rag.build_calibrated_rag_store`): GEODE trainer 300 epochs × 8 rounds, embed loop 2 rounds on a 150-epoch trainer, v SFT `full` rank 8, u contradiction SFT 400 epochs, production store `DocGMSConfig` defaults. The reference takes the store's `DocGMSConfig` from its caller; ours sets `loss_mode="cap"`, `ingest_mode="regex"` (the reference default), `train.epochs=150`. Stated differences: the encoders are passed by local path (the same models), dedup runs audit-only (`--kx-dedup-mode`), u is trained on our attribute phrasings. All settable, see below |
| Run-time geometry | knowlytix `GMSExpertStore`: `score_triple` (an unstated tail via `score_triple_emb` on the tuned v encoder), `cap_radius`, `tension_energy`, `check_holonomy`, `is_path_consistent` (with the store's calibrated `tau_path` written to `config.verify`), `lookup_enm`. Grounding is `score_triple ≤ cap_radius`; knowlytix's per-head calibrated admissibility (`core.graph.admissibility`) is reported next to it in the oracle check |
| v-space SFT and GEODE embed loop; u-space contradiction SFT | knowlytix `knowledge.geode.embed_loop`, `knowlytix.embedding` |
| Calibration | knowlytix `harness.testing.judge.GMSJudge` |
| Two-way admission, substrate generators | knowlytix `benchmark.eval.generators` (`QuestionGenerator._validate`, geometric=True) |
| Attribution | ours: deviance tests per factor with BH adjustment, joint model, interactions, per component / weak link |
| Official base/factor catalogs and profiles | proofloop package data (a vendored copy pins the expected files); contract rules in `catalog.validate` |
| Runtime gates, audit chain, injection baseline | forgeloop `gms_backend.GMSPlausibilityGate`, `audit.AuditLogger`, `governance.policies` |
| Training corpora, failure-mode taxonomy | proofloop `suite.emit`, `evaluation.failure_modes` |
| Policy compiler + GMS view (`policy.gms.md`) for GEODE | ours: renders the policy as one table per relation so the regex ingest proposes typed edges (`has_enables`, `has_alias`, `has_opposite_of`) |
| Journey oracle, covering array, realiser/reader, invariance proof, negative controls | ours (multi-turn journeys are outside the packages' single-question scope) |
| Runtime *policy* checks (R2/R3, R4, R11, R15, R16, R20-R22, R26, R28, R30, R31) and speech checks (claims, fraud stance, ENM figures) | ours, plugged into forgeloop's gate protocol |
| Design | ours (φp Sobol+refine with rank transform + swaps; knowlytix `DesignMatrix` was compared earlier and left 11 of 43 level cells empty at n=486, so it is not used) |
| Resilience probe, mutation analysis | ours (knowlytix's `ToolGateway` installs on a knowlytix `Harness`; our bot talks to a mock backend) |

### Setting the build parameters at run time

Every build parameter can be set from `CampaignConfig`, `run_campaign.py` or the notebook, and all of them
are part of the store's cache key (a changed value builds a new store; an unchanged one reloads it):

| `CampaignConfig` field | CLI flag | default |
|---|---|---|
| `kx_geode_epochs` | `--kx-geode-epochs` | 300 |
| `kx_geode_max_iters` | `--kx-geode-max-iters` | 8 |
| `kx_embed_trainer_epochs` | `--kx-embed-trainer-epochs` | 150 |
| `kx_embed_max_iters` | `--kx-embed-max-iters` | 2 |
| `kx_sft_epochs` | `--kx-sft-epochs` | package default (100) |
| `kx_contradiction_epochs` | `--kx-contradiction-epochs` | 400 |
| `kx_v_sft_mode` | `--kx-v-sft-mode` | `full` |
| `kx_dedup_mode` | `--kx-dedup-mode` | `off` (audit-only) |
| `kx_epochs` | `--kx-epochs` | 150 (shortcut for `train.epochs`; licence cap 200) |
| `kx_store_config` | `--kx-store-config` (JSON) | `{}`: any `DocGMSConfig` field by dotted key, e.g. `{"cap.margin": 0.3, "geometry.d_v": 256, "loss_mode": "point"}` |

The overrides are saved in the build report and applied again when the store is reloaded.

## How the bot is validated, and the turn-by-turn view

Each test case runs the real bot (yours or the reference bot) live: a fresh bot and a fresh mock backend, a customer
that plays the case's base journey line by line, whatever the bot answers (the scripted customer, so a bot mistake
cannot steer the case away from the one that was designed and proved; the run page's **Customer follows the base
journey** checkbox, on by default, or `CampaignConfig(customer="adaptive")` switches to a customer that answers the
bot's last question), and every bot reply, act and tool call recorded. Some checks happen during the run, turn by turn:

- each customer line is read back to its meaning card in its dialog state;
- the runtime monitor checks every tool call before it executes and every bot sentence (claims, fraud stance,
  figures);
- each policy answer is graded against the exact register.

The verdict is given on the whole run, against the oracle's answer key (`s07_checks/scoring.score`):

- the outcome (end state, transfer reason, what was truly recorded, which card action truly completed);
- each stage's decision;
- whether every step was admissible (workflow edges, login before account tools, limits, confirmation and fee
  disclosure before submit, no claim without SUCCESS, 20-turn limit).

The bot's reply at turn *i* is not required to equal the oracle's: a different but allowed route is accepted.

`gmsjourney/s09_reports/cases.py` shows the two runs side by side, one row per customer turn:

- the state the customer answered and the words the bot received;
- the oracle's expected response: tool calls with arguments and results, then bot acts such as
  `PRESENT_TXN(T2: Starbucks $6.25)` or `TRANSFER(card_options_unavailable)`;
- the bot's actual tools, acts and text;
- whether they match, and the first turn where the runs diverge.

The oracle decides what to do, not the wording, so the comparison is on tool calls and decision acts.
Acknowledgements such as CONFIRM_RECORDED are ignored there and checked on the tool call itself.

On clean base cases the reference bot matches the oracle turn by turn on every run (a self-check). On enriched cases,
every wrong outcome has a divergence, and some correct outcomes do too: the bot took a different but recoverable
route.

Use `show_turns(runner, oracle, scenario)` in the notebook (cell 11b). The campaign writes
`dataset/turn_by_turn.csv` (every turn of every case) and `dataset/turn_by_turn_summary.csv` (per case: first
divergence, expected vs actual outcome, stages wrong, violations).

## Validating what the bot says (`gmsjourney/s07_checks/speech_*.py`)

The checks above grade what the bot *did*: its tool calls and its dialog-act labels. A bot can do everything right
and still say the wrong thing, for example by labelling a turn `CONFIRM_RECORDED(T1, UNRECOGNIZED)` and recording
UNRECOGNIZED, but telling the customer "I've noted that you **recognize** the Amazon charge". The speech check reads
every sentence the customer hears and grades it against the policy graph, the GMS and what actually happened in the
session.

```
 bot turn: acts + words                                     what is true
 ───────────────────────                                    ────────────────────────────────────────────
 CONFIRM_RECORDED(T1, UNRECOGNIZED)                         policy graph: what each act must say, and
 "I've noted that you recognize the Amazon charge."           where each spoken value's truth lives
        │                                                   session: transactions returned, writes that
        ▼  reader (graph phrasings, no LLM)                   truly succeeded, what the customer said
 claims: disposition=RECOGNIZED about T1, mode=recorded     GMS: exact register (fees, days, limits),
        │                                                     rule stances, geometric distance vs cap
        ▼  verifier
 contradicted: the successful record for T1 says UNRECOGNIZED (R13)
 finding: text_contradicts_label (CONFIRM_RECORDED)
```

**What the graph declares** (a `speech` section, `gms=False`, so the GMS view and the trained store do not change):

| Relation | Example | Meaning |
|---|---|---|
| `says` | `PRESENT_TXN says merchant` (from `item.merchant`, required, basis stated) | the facts an act must speak, where their truth lives, and how firmly the PDF backs it (`basis`: stated / inferred / interpreted / design, with a note) |
| `spoken_cue` | `ACTION_DONE spoken_cue "that's done"` | words that show the act was said, when a bot gives no labels |
| `spoken_as` | `UNRECOGNIZED spoken_as "not mine"`, `max_transactions_per_session spoken_as "up to {n} charges"` | how a value or a limit is spoken |
| `returns_field` | `get_review_transactions returns_field date` | the fields a tool's result carries |
| `asserted_by` / `denied_by` | `R27 denied_by "not, by itself, proof of fraud"` | how a rule's stance is asserted or denied in words |

The sources are `item.<field>` (a returned transaction), `write.<tool>.<arg>` (a successful write),
`writes.<tool>` (every successful write), `caller.<slot>` (what the customer said), `policy.<category>` /
`policy.limit:<key>` (the exact register) and `text.opening`. `PolicyKG.validate()` checks that every source
resolves and every dialog state is declared.

**Each claim gets a verdict:**

- `supported`: matches its source (a write must have truly succeeded, with the echo matching the request);
- `contradicted`: the source holds a different value;
- `unsupported`: no source backs it (for example "recorded" with no successful record for that charge, R13, or
  "done" before any card action succeeded, R17);
- `unverifiable`: nothing to check it against.

**Each turn also gets findings:**

- `text_contradicts_label`: the words say something other than the act's label;
- `label_not_said`: a labelled act whose required facts never appear;
- `missing`: required information absent, for example a charge presented without its date (pdf.2B), or a summary
  that leaves out a recorded answer (pdf.2F). A re-ask of a charge already presented does not need the facts again
  (pdf.3);
- `not_allowed_here`: an act that is not a graph transition from the prompt the customer answered.

Policy figures (fees, delivery days, limits) are graded on the exact register. Two readings are reported next to
each one, and neither decides the verdict: the GMS distance of the claimed value from the fact against the relation's
cap, and knowlytix's own `AnswerVerifier` (geometric mode), which confirms a correct figure and does not find a wrong
one.

**Policy in everything else the bot says** (`s07_checks/speech_policy.py`). The readers above read the facts the graph names, in
the wordings it lists. Every clause they leave is read against what the graph states, so nothing that sounds like
policy passes unchecked. That means the graph's rule, limit and tool texts, a compact graph's `knowledge` lines, its
figures (limits with units, members' numeric attributes), its actions (tool verbs, the values its write tools take)
and its forbidden stances:

| The bot says | Outcome |
|---|---|
| "You can review charges up to 120 days old." | `contradicted`: lookback_days is 30 (a figure in other words) |
| "Our policy is to always refund the full amount on request." | `unsupported`: no rule, limit or knowledge line states it |
| "I'll also waive your annual fee this year." | `unsupported`: no tool or write value of the graph is a waiver |
| "Not recognizing a purchase doesn't automatically make it fraudulent." | `supported`: restates R27 |
| "I can only go up to 20 turns with you." | finding `disclosed`: max_turns is internal |
| "Let me check that for you." | finding `unread` (advisory): no policy content |

In an answer to a policy question, a figure about a sibling of what was asked (asked the standard fee, also told the
expedited fee) is the finding `volunteered`. It is advisory by default, and the answer is graded on the figure tied
to what was asked, in any word order. A second figure that names nothing is checked against every member, not pinned
to the asked one. `speech.policy` in `project.yaml` sets:
- `advisory`: the finding kinds that never fail a turn;
- `never_volunteer`: limits or values that are a disclosure unless asked, on top of what the graph marks internal;
- `judge: llm`: an LLM re-reads policy-sounding clauses and must cite a statement id it was given; its answers are
  cached;
- `markers`: the plain-English cue words.

The console edits these under Rules and scoring. Four planted errors per journey (`invented_policy`,
`wrong_policy_figure`, `unauthorised_promise`, `internal_disclosure`) measure that the checks fire. Matching a clause
to a statement uses shared word stems and equal figures: it finds which statement a clause is about, not whether a
paraphrase keeps its meaning. That second part is what the LLM judge is for.

**Scores per case** (in `results.csv`):

- `c_speech`: every turn's claims are supported, faithful to the labels and complete;
- `c_flow`: every act follows the graph;
- `outcome_correct`: the earlier verdict, kept as it was;
- `speech_claims`, `speech_supported`, `speech_contradicted`, `speech_unsupported`, `speech_missing` and the rest;
- `speech_problems`: the first few problems in words.

`--speech-gate` decides what `correct` means:

- `report`: speech is shown but not graded;
- `fail` (default): `correct` also needs `c_speech`;
- `strict`: `correct` also needs `c_flow`.

**Checking the checker.** A check that never fires proves nothing, so the campaign runs eleven negative controls
(`s07_checks/speech_policy.py`; skip them with `--no-speech-controls`). Each control edits only the words of one act in a real
conversation and leaves the labels, tools and outcome untouched:

- swap the merchant, change the amount, shift or drop the date;
- flip the recorded answer;
- a wrong fee, wrong delivery days or a wrong limit;
- "your card has been replaced" said early;
- "this means fraud";
- a summary that drops a recorded answer.

The report's section 5a shows the detection rate for each control, the claims by where their truth was read, and the
policy figures checked against the GMS (register verdict, GMS distance vs cap, knowlytix). `bot_validation.html` has
the full view by factor and response by response (notebook CELL 11d shows the same for the runs of CELL 11). Four word-only mutants
(`says_wrong_fee`, `misstates_recorded_answer`, `names_wrong_amount`, `summary_omits`) have the same outcome as the
reference bot and are killed only by this check.

```python
from gmsjourney.s07_checks.speech_checks import SpeechContract, UtteranceVerifier
v = UtteranceVerifier(store, SpeechContract.from_graph(store.kg))
conv = v.conversation(session_date="2026-03-16")          # one per conversation
report = conv.check(acts, text, tool_log, customer_frames)  # after every bot turn
report.ok, [x.as_dict() for x in report.verdicts], [f.as_dict() for f in report.findings]
conv.summary()                                            # c_speech, c_flow and the counts
```

`JourneyRunner` does this on every bot turn, before the customer answers, when the store's graph has a speech
section (`check_speech=True`). With governance on, each turn's findings are also sealed live into the runtime
monitor's hash-chained audit as the gate `speech_grounding`:

- a false or baseless claim is sealed as DENY;
- a missing or unfaithful act is sealed as ESCALATE;
- flow is graded but not sealed.

The governance `SpeechMonitor` stays as it was. It is a guard for a few things that must never be said (a claim with
no success behind it, an asserted fraud stance, a figure off the register, a card number). The speech check reads
which charge and which value each sentence names and grounds every one of them. On the campaign, two seeded word
defects get past `SpeechMonitor` and are flagged live only by `speech_grounding`: saying the opposite of the
recorded answer, and a summary that leaves one out.

## How a base case and its enriched variants are proved to be the same test

Every customer turn has a meaning in the journey ontology: a **semantic frame** (dialog act + slots bound to
canonical individuals). Ground truth is a function of the frame sequence and the backend fixture only. For each
scenario: (1) ground truth identical; (2) every enriched turn read back to the same frame; (3) the oracle replays
the read-back frames and re-derives the same outcome; (4) geometric corroboration (surface moved, meaning did
not). Meaning-changing edits must all be rejected; they are. Evidence is exported as RDF (`ontology.ttl`).

## Two enrichment arms: template and LLM

Both arms realise the **same design rows** (same base, same factor levels), so their results can be compared pair by
pair. The template arm builds each customer turn from its meaning card with deterministic code. The LLM arm
(`gmsjourney/s05_rewording/llm_*.py`, `--llm`) asks a model on the internal Microsoft (Azure OpenAI) gateway to write the turn
(default deployment `gpt-41`; `--llm-model gpt-55` for 5.5). It is authenticated with a short-lived `gcloud` access
token — `AEXP_ACCESS_TOKEN` and `AEXP_USE_CASE_ID` override the CLI and the routing header. Set the exact endpoint with
`GMSJOURNEY_LLM_URL` (full chat-completions URL; `{model}` is filled with the model name, e.g. `gpt-41`) and the
corporate CA bundle with `GMSJOURNEY_CA_BUNDLE` (else `REQUESTS_CA_BUNDLE`, else `/etc/ssl/certs/ca-certificates.crt`).
If the deployment rejects `temperature`/`max_tokens` or the system role, the client adapts once and records it
(`report.json: invariance_llm.clients.*.adaptations`). There is no other transport and no fallback. It is not trusted to construct the meaning, so it is validated separately and more strictly:

1. **Blind generation.** The model is given the agent's question for the dialog state, what the customer means (in
   plain words), the base wording and the presentation instructions. It never sees the expected bot behaviour, the
   oracle, tool names or rule ids (`s05_rewording/llm_arm.py`).
2. **Per-turn gates** (`s05_rewording/llm_arm.py`, `s05_rewording/llm_proof.py`). A turn is accepted only when all of these pass:
   - **G0 form:** one spoken line, with no internal names.
   - **G1 facts:** no figure or merchant that the base meaning does not carry, and every decision value kept.
   - **G3 one clear decision:** no competing decision cues.
   - **G2 meaning:** the rule reader reads the final text, after surface noise, in its dialog state and gets the
     base meaning card. With `--llm-dual-read gate` (the default), an independent LLM reader that never saw the base
     must also get it.
   - **Retries:** a failing turn is regenerated, up to 3 times. The feedback names the failed gate, never the answer.
     A scenario that still fails is excluded and reported as *not materialised*.
3. **The same scenario-level test as the template arm:**
   - L1: the ground truth is identical;
   - L3: the oracle, replaying the read-back cards, re-derives the outcome and every stage;
   - the conversation signatures are equal.
4. **Negative controls on LLM text**, reported separately:
   - meaning-changing edits of the accepted LLM utterances (negation, digit, figure, delivery, card action,
     merchant, bare yes/no);
   - adversarial rewrites in which the LLM changes exactly one meaning element;
   - the wrong-state control.

   All of these go through the same acceptance function and must be rejected. Accepted controls are listed for
   audit in `dataset/llm_negative_controls_audit.csv`.
5. **Manipulation check.** Each requested factor level is checked for whether it is visible in the text. The LLM arm
   is attributed on the levels that actually reached the bot.

Accepted LLM utterances are pinned. The run sends the bot exactly the text that was proved. A turn the proof never
saw (the bot left the oracle's path) is generated live under the same gates, or falls back to the template arm and is
counted. Generations are cached by content hash, so a campaign re-runs byte for byte with `--llm-mode replay`.

## gmsh parity (0.4)

These additions follow from the gmsh comparison:

| addition | where |
|---|---|
| GEODE `dedup=False`, so the loop's retrains and anchor check see every fact; duplicate resolution runs audit-only on the trained store; anchor violations reported. `--kx-dedup-mode audit` keeps the 0.3 behaviour; `accept` is stock gmsh. | `s03_knowledge/build.py` |
| v-space fine-tuning: full (the reference default), rotation, or `ab` to train both and keep the better held-out noisy-alias binding (`--kx-v-sft-mode`) | `s03_knowledge/build.py` |
| gmsh's P=0.5 logistic operating point next to the regulated one, k-fold held-out accuracy, bootstrap 95% intervals for θ, the plan budget and every relation's cut | `s07_checks/gates.py` |
| Quorum REVIEW (`--decision-rule quorum`). Always shadow-computed and reported: a syntax error or ≥2 blocking layers deny; one blocking layer is routed to review (escalation). | `s07_checks/runtime.py` |
| Numeric range attestor on every figure the bot speaks: PASS (register value), REVIEW (a register value of another quantity), FAIL (invented) | `s07_checks/runtime.py` |
| Per-turn latency and length budgets plus a tool-call budget, reported next to correctness | `s07_checks/scoring.py` |
| Severity per run with gmsh's risk-tier escalation, and a tiered release gate | `s07_checks/scoring.py` |
| Hallucination probes: near-miss option names and an unstated fee. The bot must abstain; a new mutant (`kb_fabricate_unknown`) fabricates instead. | `s03_knowledge/questions.py`, `s03_knowledge/questions.py` |

## Testing your real bot

```python
from gmsjourney.s08_campaign.pipeline import CampaignConfig, run_campaign
from gmsjourney.s06_bots.harness import CallableBotAdapter

def factory(tools):                       # point your bot's tool client at `tools`
    session = MyBot(tool_client=tools)
    return CallableBotAdapter(start_fn=session.start, respond_fn=session.respond)

run_campaign(CampaignConfig(out_dir="my_bot_run", bot_factory=factory, bot_name="MyBot"))
```

`HTTPBotAdapter` drives a deployed bot. If your bot returns text only, `BotActClassifier` infers its acts. The
runtime monitor wraps the same tool client, so the gates watch your bot exactly as they watch the reference bot;
`GovernanceConfig(mode="enforce")` turns DENY into a blocked call.

### An LLM bot as the system under test

`gmsjourney/s06_bots/llm_bot.py` is a second SUT: a tool-using agent on the same model gateway and replay cache as
the LLM arm (cache tag `sut`). It is given the operating procedure (`policy/charge_review_policy.md`), the tool
API with its transfer reason codes, and the dialog-act inventory - nothing the oracle computes. Each step the
model returns one JSON object, a tool call (made on the harness backend, so faults, gates and the trace see it)
or the acts it says; a malformed reply is sent back once, then surfaces as an `UNKNOWN` turn (a bot defect,
never repaired by the harness). Notebook CELL 15b runs the full campaign with it:

```python
from gmsjourney.s05_rewording.llm_proof import LLMSettings
from gmsjourney.s06_bots.llm_bot import make_llm_factory
bot = make_llm_factory(store, settings=LLMSettings(model="gpt-41", temperature=0.0, max_tokens=900))
run_campaign(CampaignConfig(out_dir="output/campaign_llm_sut", bot_factory=bot, bot_name="LLMBot", runs_per_base=2))
```

`max_cases=N` tests only N base cases (each with its enriched variants), picked in proportion to the base
categories; the default `None` tests all of them. Progress bars (tqdm) show each run phase with the number correct
so far (`progress=False` turns them off). The report's section 5b lists every failed run with the reasons it failed
(outcome, recorded responses, card action and stages expected vs. what the bot did, policy violations, wrong policy
answers) and its transcript, most severe first, filterable by arm and severity.

`case_comparison.html` (next to `report.html`) is the same campaign case by case, in the answer sheet's format: for
every run the outcome signature expected by the oracle beside what the bot produced, field by field (how the call
ended, transfer reason, responses recorded, card action completed, and the policy answer for policy-question cases -
the comparison that decides correct / incorrect), the stage, admissibility, invariance and budget checks reported
next to it, the reasons a run failed, and every customer turn with EXPECTED (oracle tools and acts) beside ACTUAL
(bot tools, acts and words). Incorrect runs are shown by default; buttons show the correct ones, all, or one arm.
It is built from `dataset/case_results.jsonl` and the turn-by-turn files, so an existing campaign folder can be
redone without re-running the bot: `python tools/case_comparison.py --campaign output/campaign_llm_sut`.

Each customer turn costs 1-4 gateway calls; replies are cached (temperature 0), so repeated runs of the same
conversation and `mode="replay"` cost nothing. `bot.stats` / `bot.llm.report()` give calls, cache hits,
malformed replies and tool errors. `make_llm_factory(..., critique=True)` adds the critique layer (a second call
reviews every step and may correct it; `critic_model="gpt-55"` or `critic_llm=` gives the reviewer another model).
In the campaign notebook, `CRITIQUE = True/False` and `CRITIC_MODEL` in CELL 15b choose it. Tests: `python tests/test_llm_sut.py` (a scripted fake model, no network).

## Outputs (`output/campaign_out/`)

| file | content |
|---|---|
| `report.html` / `report.json` | the campaign report / every number in it |
| `bot_validation.html` | what the bot did and what it said. Every run is shown with two verdicts: did the right thing (the outcome) and said it right (every claim grounded). Runs are split by type of enrichment, base category, DOE factors and enrichment factors. Each claim shows where its truth was read and how it fared against the GMS. Response cards show the failing responses, each with its case's factor levels. Rebuild it from the files with `python -m gmsjourney.s07_checks.diagnosis output/campaign_out` |
| `case_comparison.html` | every case, expected vs actual, turn by turn, with the speech verdict |
| `dataset/case_results.jsonl` | every graded run, including `speech_turns`: one response card per bot turn with its claims, verdicts, truth, where it was read, and the GMS reading |
| `gms_build_report.json` | knowlytix build: GEODE diagnostics, embed loop and SFT evaluations, judge thresholds, ENM agreement, timings |
| `policy.gms.md` | the GEODE input rendered from the policy |
| `results_enriched.csv`, `results_base_level.csv` | one row per scenario (`f_<factor>`, `correct`, `c_<stage>`, gate alarms, ...) |
| `failures.jsonl`, `scenarios.jsonl` | failed runs with transcripts; every scenario with its proof |
| `sft_classifier.jsonl`, `sft_draft.jsonl` | proofloop-schema training corpora |
| `ontology.ttl`, `store_graph.dot` | journey ontology + invariance evidence; store subgraph (Graphviz) |
| `policy_kg.json` | the policy graph the run was built from |
| `gms_canonicals.json` | canonical values, aliases and how the trained GMS binds them, exact numbers, states, acts, every trained fact with its score and cut |
| `dataset/test_cases.csv` | one row per test case (81 base + 486 enriched): category, policy question, canonicals, meaning cards, base case, enrichment, enriched output, expected outcome, complete trajectory, proof verdict |
| `dataset/test_turns.csv` | one row per customer turn: state, stage, the state's rules/limits/accepted cards, meaning card and its H/R/T, base and enriched text, read-back card, equivalence |
| `dataset/policy_questions.csv`, `dataset/test_cases.jsonl` | the 27 policy questions; the cases with nested structure |
| `dataset/turn_by_turn.csv`, `dataset/turn_by_turn_summary.csv` | per customer turn: oracle's expected tools and bot acts vs the bot's actual tools, acts and text; per case: first divergence and outcome |
| `results_enriched_llm.csv`, `scenarios_llm.jsonl` | LLM arm: one row per scenario (realised factor levels); every LLM scenario with its gates and proof |
| `dataset/llm_enrichment_turns.csv` | every LLM-written turn: base, template and LLM text, both readers' cards, each gate, attempts, factors requested / realised |
| `dataset/llm_negative_controls_audit.csv`, `dataset/arm_pairs.csv` | controls the LLM gate accepted (for human audit); template-vs-LLM pairs |

## Layout

The package is ten segments (folders) plus one shared module. Each folder has a `README.md` that maps every file;
the full guide, with the theory behind every step, is `docs/guide/README.md`.

```
gmsjourney/
  llm.py             the model-gateway client (one cached chat call), shared by the LLM arm and the LLM bot
  progress.py        progress of the long parts: tqdm on a terminal, progress events in the console
  s01_packs/         a journey's pack: loading.py (load, lint, the active pack), s01_packs/names.py (N.state.X ... and tools by
                     role), s01_packs/project.py (project.yaml: factors, levels, CaseView), drafting.py (lexicon, bot lines and
                     project.yaml drafted from a compact graph)
  s02_graph/         the policy graph: compact.py (the compact format), triples.py (the full graph: build, load,
                     validate), compile.py (compact -> full), s02_graph/flow.py (expression language, FlowSpec, the oracle that
                     walks the graph), interpreter.py (the compact graph's own walker)
  s03_knowledge/     the GMS: store.py (ingest, symbolic store, canonicals, GEODE view, the pack's store),
                     build.py (policy compiler, knowlytix backend, trained store), questions.py (policy-question truth)
  s04_cases/         the test cases: meaning.py (ontology, meaning cards), s04_cases/journey.py (journeys, customer simulator),
                     factors.py (journey factors, covering array), design.py (DOE designs, base cases)
  s05_rewording/     reworded customers and their proof: catalog.py (bases, factors, profiles), wording.py (lexicon,
                     noise, realizers, the rule reader), proof.py (invariance), llm_arm.py / llm_pack.py /
                     llm_proof.py (the LLM arm: prompts, realizer, dual read, gates, proof, exports)
                     catalogs/official/ (the domain-neutral factor and base catalogs)
  s06_bots/          the bots under test: tools.py (mock bank, trace), reference.py (reference bot + mutants),
                     llm_bot.py (an LLM bot), harness.py (adapter, journey runner)
  s07_checks/        grading: scoring.py (constraints, severities, release gate), speech_reading.py /
                     speech_checks.py / speech_policy.py (what the bot said), runtime.py (governance rules, monitor),
                     s05_rewording/llm_arm.py (gate calibration), diagnosis.py (stats, attribution, mutation, resilience, bot validation)
  s08_campaign/      pipeline.py (the 13 steps), outputs.py (taxonomy, SFT corpora, dataset/*.csv)
  s09_reports/       report.py (report.html), cases.py (narration, turn by turn, case comparison), s09_reports/verdict.py
  s10_console/       the journey console: __init__/__main__ (start it), s10_console/server.py (page + JSON API), studio.py,
                     runs.py (one process per run), bots.py (attachments, bot under test, playground),
                     graphs.py (sections, core segments, versions, the GMS sphere), ui.html, studio_ui.html
  policy/            the packs: charge_review/, charge_review_graph/, supplementary_card/ (graph, domain.yaml, project.yaml)
run_campaign.py · tests/ · notebooks/ · DESIGN.md · NOTICE
tools/             check_llm (one-call transport preflight), llm_rejections (why LLM turns were rejected),
                   answer_sheet (the case answer key), build_annotated (the core-file listing), build_docs,
                   case_comparison, graph_showcase, geode_before_after
docs/              guide/ (the project guide) and generated walkthroughs: EXPLAINER, INVARIANCE, ANSWER_SHEET, ANNOTATED
output/            generated campaign runs (report, dataset/, corpora) — gitignored
fictional_poc_operating_procedure_gpt_readable.pdf (source document, not published)
```

The package, the notebook and the tools are the source; `docs/` and `output/` hold everything generated,
so a clean checkout is `gmsjourney/ notebooks/ tests/ tools/ *.py *.md *.txt *.toml` — regenerate the rest
with `run_campaign.py` and the two tools.

Licensing: knowlytix-core is used under the Knowlytix EULA and is not included in this package; the other
knowlytix packages, proofloop and forgeloop are Apache-2.0. See `NOTICE`.
