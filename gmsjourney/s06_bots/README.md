# `gmsjourney/s06_bots/` — Bots

The bots under test (the reference bots with their planted defects, an LLM bot), the mock bank with fault injection, and the harness that plays a case against any bot.

The full explanation, with the theory and diagrams, is chapter 6 of the guide: [docs/guide/06-bots.md](../../docs/guide/06-bots.md).

```
gmsjourney/s06_bots/
├── __init__.py   The bots under test and the harness that drives them
├── harness.py    Driving a bot through a case and observing it
├── llm_bot.py    The LLM bot under test: a model reads and words, the policy graph decides
├── oracle_says.py  Grading an agent that gives only tool calls and words: the oracle's SAYS, held in its words
├── reference.py  The reference bots: well-behaved bots under test that follow the policy
├── tools.py      The mock bank the bot talks to, and the record of one conversation
└── turns.py      One customer turn of a run compared with the same turn of the graph's own run
```

## `__init__.py`

The bots under test and the harness that drives them.

## `harness.py`

Driving a bot through a case and observing it.

Parts:

* PART 1 - Drive a bot through a scenario and observe it as a Trace (book Ch. 9.3).
* PART 2 - Driving any bot through a journey mined from a compact policy graph (book Ch. 9.3).

| name | kind | what it is |
|---|---|---|
| `BotActClassifier` | class | Infer the dialog act of a text-only bot turn (for bots without structured acts). |
| `JourneyRunner` | class | Runs a bot through one scenario: realises every customer turn under the scenario's levels, plays the customer (scripted or adaptive), passes tool calls to the mock bank (with any injected faults), checks each turn (runtime monitor, speech check, budgets), grades each bot turn against the graph's turn (`extras["turn_status"]`: same / side / asked_again / differs; a re-ask gets the line again once; two turns in a row off the journey stop the run, `extras["stopped"]`) and returns the result with its Trace. |
| `make_reference_factory` | def | A bot factory for the charge-review reference bot (with the given planted defects). |
| `CallableBotAdapter` | class | Wrap your own bot: provide start() and respond(text) callables that return text (and optionally a list of act dicts). Tool calls must go to the `tools` object the factory receives, so point your bot's tool client at it. |
| `HTTPBotAdapter` | class | Drive a deployed bot over HTTP. Expects POST {session, text} -> {text, acts?, ended?}. The bot's backend must be pointed at the mock tools for the session (fixture id). |
| `item_index` | def | item id -> item, for every list of items a scripted tool returns (context for realisation). |
| `GraphRunner` | class | The JourneyRunner for a journey mined from a compact policy graph. |

## `llm_bot.py`

The LLM bot under test: a language model reads the customer and words the bot; the journey's policy graph decides.

Parts:

* PART 1 - A local model behind the LLM bot under test: gpt-oss, Llama or any chat model vLLM serves, loaded with vLLM
* PART 2 - The LLM bot under test, structured: the model reads the customer and words the bot; the policy graph decides

| name | kind | what it is |
|---|---|---|
| `local_model_path` | def | The local model folder from GMSJOURNEY_SUT_LOCAL_MODEL (empty when not set). |
| `status` | def | What the console shows about the local model, without loading it. |
| `LocalChat` | class | A chat model on vLLM: in-process from a folder, or through an OpenAI-compatible vLLM server. |
| `READ_SYSTEM`, `WORD_SYSTEM` | const | The two fixed prompts: reading a customer turn, wording the bot's acts. Neither holds any policy. |
| `journey_title` | def | The active journey's title. |
| `StructuredStats` | class | Counters over every conversation of a factory: turns, readings by kind, readings not used, lines worded or said from the template, wordings not used and why (the last 50). |
| `LLMReader` | class | Reads a customer turn with the LLM into a meaning card: ``menu(state)`` (the state's intents, their slots and values, what a plain yes / no means, the facts a question can be about), ``read(text, state)``, and ``frame(...)``, which uses a reading only when the state accepts it (an alias is mapped to its member, a code must be in the words) and otherwise reads the turn as unclear. Kinds: actionable, question, claim, social, unclear. |
| `Wording` | class | Words a bot turn's acts with the LLM, one call a turn; ``check(line, template, act)`` keeps a line only when it says the template's figures and facts and adds no figure, option, promise or policy word; a lead (an acknowledgement of small talk) is checked the same way. |
| `StructuredBot` | class | The LLM bot: ``GraphBot`` (the graph's interpreter, no planted defects) with ``LLMReader`` reading and ``Wording`` speaking; a policy claim is answered from the graph ("Actually, ..." when the customer's figure is wrong). |
| `make_structured_factory` | def | A bot_factory for CampaignConfig / JourneyRunner: ``reader_llm`` reads, ``wording_llm`` words (None: the templates), ``voice`` is the tone asked for. ``factory.store`` may be set later (the pipeline hands its store to a factory whose store is None). |
| `fake_model` | def | A stand-in chat model named by ``GMSJOURNEY_SUT_FAKE_MODEL=module:Class`` (tests and demos without the gateway), else None. |

## `reference.py`

The reference bots: well-behaved bots under test that follow the policy.

Parts:

* PART 1 - A reference implementation of the charge-review voice bot (the stand-in SUT).
* PART 2 - A reference bot for a compact policy graph, and the defects (mutants) mined from the graph.

| name | kind | what it is |
|---|---|---|
| `BotTurn` | class | What a bot returns for one turn: the words, the dialog acts (labels with fields) and whether it ended. |
| `KeywordNLU` | class | A typical intent/slot classifier: lowercase keyword lists, no repair. |
| `KBAnswerer` | class | The reference bot's policy-question answerer: picks the knowledge-base sentence that shares most words with the question (deliberately narrow, so the suite has something to find). |
| `spoken_date` | def | '2026-03-13' -> 'March 13' (how the bot reads a charge's date out). |
| `ReferenceBot` | class | The charge-review reference bot: keyword understanding, the procedure as code, planted defects on request. |
| `KeywordReader` | class | The reference bot's own understanding: the domain's value and act cues (the pack's lexicon, as a bot built for this domain would know them) plus the graph's synonyms, bare yes / no and a 4-digit code - but none of the proof reader's normalisation: no typo or speech-recognition repair, no number words, no reported speech, no correction markers. |
| `mutants` | def | Every defect the graph makes possible. |
| `GraphBot` | class | The reference bot: start() / respond(text) -> BotTurn, tools observed at the mock backend. |
| `make_graph_factory` | def | A bot factory for the graph reference bot of the active compact-graph pack. |

## `tools.py`

The mock bank the bot talks to, and the record of one conversation.

Parts:

* PART 1 - Simulated backend tools and the fault-injection gateway (book Ch. 11).
* PART 2 - One journey's record, for both the oracle (expected) and the bot (observed).

| name | kind | what it is |
|---|---|---|
| `tool_specs` | def | The active pack's tool specifications (domain.yaml -> tools, mined from a compact graph). |
| `tool_names` | def | The names of the active pack's tools. |
| `tools_with_role` | def | The tools whose role is one of ``roles`` (auth, read, write, handoff ...). |
| `account_tools` | def | Tools that touch the account: only callable after the auth tool succeeded. |
| `write_tools` | def | The tools that change the account. |
| `key_args` | def | The arguments that identify a call when comparing two runs (tools without any: all args). |
| `ToolError` | class | A tool call failed (the mock bank's equivalent of a service error). |
| `ToolTimeout` | class | A tool call timed out. |
| `ToolEnvironment` | class | One journey's backend: every tool is served from the fixture by its declared behaviour. |
| `FaultProfile` | class | A fault to inject: which tools (a pattern), what kind (error, timeout, stale, plausible_wrong) and how often. |
| `ToolGateway` | class | Intercepts every tool call of an environment and applies fault profiles. |
| `Trace` | class | What happened in one run. Which tools authenticate, read, write and hand over, and which acts and limits matter, come from the domain pack (domain.yaml -> flow, tools). |
| `new_trace` | def | A trace scored the way the active pack is: the graph-generic scoring (s07_checks/scoring.py) for a pack written as a compact graph, else the per-item scoring configured in domain.yaml (flow). |

## `turns.py`

One customer turn of a run compared with the same turn of the graph's own run: the trajectory and the side steps.

| name | kind | what it is |
|---|---|---|
| `OPTIONAL_ACTS` | const | Acknowledgements a bot may add or leave out (OPENING, CONFIRM_RECORDED). |
| `REASK_AFTER` | const | Acts after which presenting the same item again is not a difference. |
| `side_acts` | def | The side-step acts of the active graph: its policy answer (`globals.question.answer`) and clarification (`globals.unclear.clarify`). |
| `normalize_acts` | def | The decision content of a reply: (act, item, reason) in order, without optional acknowledgements and repeats. |
| `tool_key` | def | A tool call as its name and key arguments. |
| `compare_turn` | def | A bot turn against the graph's turn: `same`, `side` (only side steps differ) or `differs` (off the journey). |
| `asked_again` | def | The bot made no call and asks again the question the customer just answered. |
| `split_turns` | def | A run's events cut into the opening and one entry per customer line, with its calls and acts in the order they happened (`steps`). |

## `oracle_says.py`

Grading an agent that gives only tool calls and words: the oracle's SAYS, held in the agent's words. Each act the
oracle says at a turn is *held* (the words express it, by a cue word from the pack's `bot.cues` or every field it
must say, with no other value), *contradicted*, *not heard* or *not checkable*; each sentence goes to the oracle's
act it expresses, and a sentence expressing an act the oracle does not take there is an act of the agent's own.

| name | kind | what it is |
|---|---|---|
| `hold` | def | The oracle's acts for one turn, each checked against the agent's words: the acts the turn is graded as having said, the segments the speech check reads, and the check of each act. |
