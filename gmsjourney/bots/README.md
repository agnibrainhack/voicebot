# `gmsjourney/bots/` — Bots

The bots under test (the reference bots with their planted defects, an LLM bot), the mock bank with fault injection, and the harness that plays a case against any bot.

The full explanation, with the theory and diagrams, is chapter 6 of the guide: [docs/guide/06-bots.md](../../docs/guide/06-bots.md).

```
gmsjourney/bots/
├── __init__.py   The bots under test and the harness that drives them
├── harness.py    Driving a bot through a case and observing it
├── llm_bot.py    An LLM bot under test: a language model given the procedure, the tools and the dialog acts
├── reference.py  The reference bots: well-behaved bots under test that follow the policy
└── tools.py      The mock bank the bot talks to, and the record of one conversation
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
| `JourneyRunner` | class | Runs a bot through one scenario: realises every customer turn under the scenario's levels, plays the reactive customer, passes tool calls to the mock bank (with any injected faults), checks each turn (runtime monitor, speech check, budgets) and returns the result with its Trace. |
| `make_reference_factory` | def | A bot factory for the charge-review reference bot (with the given planted defects). |
| `CallableBotAdapter` | class | Wrap your own bot: provide start() and respond(text) callables that return text (and optionally a list of act dicts). Tool calls must go to the `tools` object the factory receives, so point your bot's tool client at it. |
| `HTTPBotAdapter` | class | Drive a deployed bot over HTTP. Expects POST {session, text} -> {text, acts?, ended?}. The bot's backend must be pointed at the mock tools for the session (fixture id). |
| `item_index` | def | item id -> item, for every list of items a scripted tool returns (context for realisation). |
| `GraphRunner` | class | The JourneyRunner for a journey mined from a compact policy graph. |

## `llm_bot.py`

An LLM bot under test: a language model given the procedure, the tools and the dialog acts.

Parts:

* PART 1 - A local model behind the LLM bot under test: Llama 3.1 8B Instruct (or any chat model vLLM serves) loaded with vLLM
* PART 2 - An LLM-driven charge-review voice bot: a second system under test, next to the keyword/state-machine

| name | kind | what it is |
|---|---|---|
| `local_model_path` | def | The local model folder from GMSJOURNEY_SUT_LOCAL_MODEL (empty when not set). |
| `status` | def | What the console shows about the local model, without loading it. |
| `LocalChat` | class | A chat model on vLLM: in-process from a folder, or through an OpenAI-compatible vLLM server. |
| `default_policy_text` | def | The operating procedure the bot is given: the pack's policy transcription when it has one, else the policy graph rendered as text (what a team without a separate SOP would hand an LLM bot). |
| `journey_title` | def | The active journey's title, for the LLM bot's prompt. |
| `act_inventory` | def | name -> (what the act is for, the fields it carries): the dialog states of the active pack's graph. A state the charge-review inventory above describes keeps that description; any other is described from the graph (whether it waits for the customer) and the pack's own wording of it (domain.yaml bot.says). |
| `transfer_reasons` | def | The reason codes the transfer tool takes: the graph's (``<transfer tool> transfer_when <reason>``) when it names them, else the charge-review codes above. |
| `suggested_instructions` | def | The part of the bot's prompt a tester may rewrite: who the bot is, how it must behave, and the operating procedure. The console shows it as the suggested prompt. |
| `references_text` | def | Files the tester attached, as reference material after the instructions. |
| `protocol_text` | def | The fixed part of the prompt: the tools, the transfer reasons, the dialog acts and the JSON reply format. The harness needs it to run the bot's tool calls and to read what it says, so it is not editable. |
| `build_system_prompt` | def | instructions (default: the suggested ones on ``policy_text``) + attached references + the fixed protocol. |
| `LLMBotStats` | class | Counters over a whole campaign of LLM-bot conversations (model steps, tool calls and errors, invalid replies, the critique layer's reviews and corrections); the corrections themselves are kept too. |
| `critic_system` | def | The critique layer's system prompt: the reviewer's instructions, then the bot's own instructions. |
| `LLMBot` | class | start() / respond(text) -> BotTurn, like ReferenceBot; tool calls go to `tools` (the harness backend). |
| `fake_model` | def | A stand-in chat model named by ``GMSJOURNEY_SUT_FAKE_MODEL=module:Class`` (tests and demos without the gateway), else None. |
| `make_llm_factory` | def | A bot_factory for CampaignConfig / JourneyRunner. One CachedLLM (tag "sut") is shared by every conversation, so the cache and the call statistics cover the whole campaign: ``factory.llm.report()``, ``factory.stats.as_dict()``. settings default: the gateway's default deployment at temperature 0. ``instructions`` replaces the suggested instructions (who the bot is and its procedure); ``references`` are attached files ({name, text}) added after them; the tools, acts and reply format are always appended. |

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
| `new_trace` | def | A trace scored the way the active pack is: the graph-generic scoring (checks/scoring.py) for a pack written as a compact graph, else the per-item scoring configured in domain.yaml (flow). |
