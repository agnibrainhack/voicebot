# 9. Reports: reading a campaign's results

```
voicebot/
└── gmsjourney/
    ├── s01_packs/  s02_graph/  s03_knowledge/  s04_cases/  s05_rewording/  s06_bots/  s07_checks/  s08_campaign/
    ├── s09_reports/      ◀ this chapter
    │   ├── __init__.py
    │   ├── answer_sheet.py  answer_sheet.html: every case and what a correct bot must do (after step 5)
    │   ├── report.py     report.html: every section of the campaign
    │   ├── cases.py      narration, turn by turn, case_comparison.html, show_case
    │   └── verdict.py    verdict.html / verdict.json: the one-pager
    └── s10_console/
```

## The idea

A campaign produces thousands of numbers. The reports present them at three distances, like **a map at three
zoom levels**:

| zoom | page | answers |
|---|---|---|
| country | `verdict.html` (`verdict.py`) | can the bot be released? what are the three biggest risks? can anyone reproduce this? |
| region | `report.html` (`report.py`), `bot_validation.html` (chapter 7) | every step's results: store, bases, design, proof, results, attribution, resilience, mutation, governance, speech |
| street | `case_comparison.html`, `turn_by_turn.csv` (`cases.py`) | one case, turn by turn: what the graph expected beside what the bot did |
| the key | `answer_sheet.html` (`answer_sheet.py`) | before any bot runs: every case, what a correct bot must do turn by turn, how the call must end, and its proved rewordings |

Where a run had the two LLM judges (chapter 7), the verdict adds tiles for how often they agree with each other and
with the rules, and every case (in the verdict and in `case_comparison.html`) shows both judges' verdicts and reasons
beside the rules' (`read_judges` reads `dataset/llm_judges.jsonl`).

Every page is a single self-contained HTML file (no external scripts), with light and dark themes, so it can be
attached to an email or opened offline.

## report.py — the campaign report

```
s09_reports/
└── report.py   ◀   PART 1 render_html (and the small helpers esc, pct, tile, table, hbar, pfmt, failure_reasons)
                    PART 2 render_graph_html (the older compact-graph campaign path)
```

`render_html(R)` turns the campaign's result dictionary into `report.html`. Its sections follow the steps:

1. the store: triples, exact figures, self-correction (with the injected misread caught), self-tests, build diagnostics;
2. the bases: policy questions by category and admission (doubly grounded / symbolic only), journeys with the coverage
   report (pairs, rules, edges, triggers, boundaries);
3. the design: factors and levels, design quality against the alternatives, the power curve;
4. the proof: accepted scenarios per layer, re-seeds, negative controls with their rejection rates, surface similarity;
5. results: accuracy on original and reworded wording, per stage, per category, severities, both release gates;
6. what the bot said: claims by verdict, findings by kind, speech negative controls;
7. attribution: the factor table with deviance tests, BH-adjusted p-values (`pfmt` formats them), level tables, the
   joint model, interactions, the weak link;
8. resilience, mutation analysis, runtime governance (calibration, live flags, audit);
9. failed runs, each with plain-English reasons (`failure_reasons`, read off its graded row; nothing is re-judged).

The helpers are deliberately small: `esc` (HTML escaping), `pct` (a fraction as a percentage), `tile` (a headline
number over a label), `table`, and `hbar` (a horizontal bar chart in SVG with hover titles and direct labels).

## cases.py — case by case

```
s09_reports/
└── cases.py   ◀   PART 1 plain-English summaries (case_facts, template_summary, LLMNarrator, graph_edge, turn_hops, tool_line)
                   PART 2 turn by turn (turn_by_turn, segment, normalize_acts, speech_by_turn, show_turns, export_turn_by_turn)
                   PART 3 write_case_comparison (case_comparison.html)
                   PART 4 show_case, compare_case, case_hrt, frame_hrt, pick, variants_of
```

### Part 1: what happened, in words

`case_facts(turns)` builds a fact sheet of a case (each turn: state, what the customer said and meant, the steps
taken, what the bot said). `template_summary` writes a deterministic account from it ("The caller authenticated on
the second attempt, did not recognise the Starbucks charge, ..."); `LLMNarrator` can write a more natural summary with
a model, **checked against the fact sheet** and replaced by the template when it adds or drops a fact. `graph_edge`
finds the graph transition behind each hop so the summary can cite the rule; `turn_hops` splits a turn's steps into
dialog-state hops; `tool_line` writes a tool call as `tool(arg=value) -> status`.

### Part 2: turn by turn

`turn_by_turn(runner, oracle, scenario)` plays a scenario and returns one row per customer turn: the oracle's
expected tools and bot acts next to the bot's actual tools, acts and words, whether they match, and what the speech
check said. Each row has its base-journey `line` (a line said again has `repeat`), the turn's `expected_steps` and
`actual_steps` (calls and acts in the order they happened), `turn_status` (same / side / wrong_action / asked_again /
differs, from the run, chapter 6) and `kb_wrong` (a policy answer judged wrong, or not given, on a turn otherwise on
the journey).
Lines the customer never said because the run stopped are rows with `turn_status` `not_played`. Charges are named by
merchant and amount (`items_named`: what the bank returned in the runs). The summary adds `turns_side`,
`turns_asked_again`, `turns_kb_wrong`, `turns_wrong_action`, `turns_off`, `turns_not_played`, `stopped_at_turn` /
`stopped_at_line` / `stopped_why`, `items`, and the opening statement on both sides (`expected_opening_steps` / `actual_opening_steps`,
`expected_opening_text` / `actual_opening_text`). `segment` cuts a run into turns (with `s06_bots/turns.split_turns`); `normalize_acts` keeps only the decision content of a reply (act, item,
reason), so optional extras do not count as differences; `describe_act` / `describe_tool` make them readable. The
summary says the first divergent turn, the stages that went wrong, and whether the outcome and the words were right.
`show_turns` prints it in a notebook; `export_turn_by_turn` writes `dataset/turn_by_turn.csv` and its summary.

### Part 3: the comparison page

`write_case_comparison(out_dir)` writes `case_comparison.html` from the run's dataset files: every case with its
expected outcome beside the agent's, turn by turn from the opening statement, with the speech verdict.

Each turn sets the **oracle's turn** (left: its tool calls and what it says, the acts with their values, numbered in
order) beside the **agent's turn** (right), which shows only what can be observed of an agent:

* its **tool calls**, with their key arguments and results;
* its **words**, set apart in a box (*The agent said*), each sentence tagged with the act it was read as;
* **the oracle's SAYS, checked in these words** (`actual_says`, from `s06_bots/oracle_says.py`): each act the oracle
  says at this turn, marked *✓ held* (with the cue or the fields that show it), *✗ other value* (what the oracle
  says and what the words say), *✗ not in the words*, or *– not checkable*; and *+ not expected* for an act the words
  express that the oracle does not take there, and *! closing words* for words that sound like the end of the call
  ("thank you for calling") while the agent goes on: a wording point, not graded as ending the call (closing words
  end the call only when the agent ends it). The agent's own act labels are not shown and not used;
* **claim cards** (`actual_claims`): every value, figure or limit read from the words, with its truth and where it
  was read (what the customer said, a confirmed write, a tool's result, the policy store), its result (supported,
  contradicted, unsupported, unverifiable; side step marked) and why; then the reply's findings (missing, flow,
  advisory) and its verdict (grounded, fails only in a side step, not grounded).

Every turn is one of four kinds, each in its own colour (a legend heads the page): *same as expected*; *non-material
deviation (side step)* (amber, the run went on; also when the policy answer is wrong or missing); *incorrect action*
(purple: the right steps, one done wrongly); *off the journey* (red: a different step, or the same question asked
again); and *not played*. A policy answer is judged where it was given (`kb_answers`, from the bot event, never by
position), and part of the verdict only for a policy-question case. Where the stop rule ends a run the page says
**Validation Stopped**: in the meta line and the verdict row (at which turn), as a banner under the last turn played,
and on every turn not played.

### The speech check playground (`speech_playground.html`)

`write_speech_playground(out)` copies `s09_reports/speech_playground.html` into the run folder (the campaign does it
after `case_comparison.html`). It is one self-contained page that reads the run's `dataset/case_results.jsonl`:
fetched next to it when the folder is served, or chosen / dropped when the page is opened from disk (browsers do not
let a local page read its folder by itself). It uses each row's `speech_turns` (the response cards of
`s07_checks/scoring.response_cards`, each with the checker's own `trace`) and `speech_turn_grades` (every agent
turn's ok / core_ok / flow_ok).

For each agent response it shows the customer's line and the whole response with every claim's words marked, then
replays the check in thirteen steps, all from what the checker recorded (`s07_checks/speech_trace.py`; nothing is
re-derived in the page):

| Step | What it shows |
|---|---|
| What the check knows | the customer's meaning cards so far and the slots they gave (`facts.caller`: the truth a value said back is held to), the writes the bank confirmed and the ones that failed, the charges and each tool's latest result, the agent's labels |
| Pieces and sentences | per act (each act carries its words) or the whole response; each piece cut into sentences |
| Word by word | the sentence as read (plain punctuation, number words as digits, lower case), each word with its position; *Scan word by word* replays, in word order, every cue word and every reader match, kept or dropped, and which claim each kept match became |
| How it is put | the three cue lists (question or condition, offer, completion) and the mode they give: offered, done or states |
| Readers and matches | each value set's phrases (how many, longest first) with every match, the ones dropped because a longer phrase took the words, and the figure patterns (money, free, business days with "at most" / "at least", limits); the readers that found nothing |
| Claim cards | the claims of each sentence: field, value, words, about, mode |
| Where the truth is | per claim, every rule tried in order (opening, limit, policy figure, summary of what was written, said to be done, a tool's result, a choice the prompt offers, what the customer said) with why it did or did not apply; the truth and where it was read |
| Said vs truth | the comparison made (equals ignoring case, is one of, equals as numbers, a range that holds the register value), its result and the verdict |
| Acts: labelled vs heard | faithfulness: each labelled act heard (by a cue word, or by a claim for every field it must say) or not judged, and each labelled value against the words |
| Required wording | completeness: each act's required field, said, missing or skipped (with why), and the policy reader's clauses |
| Dialog flow | the prompt being answered, what the customer meant, and each act's step in the graph |
| This response | every claim and finding counted as failing, blocking, flow or advisory, side steps marked, and `ok` / `core_ok` / `flow_ok` worked out |
| The case | every agent turn's grades (click one to open it), `c_speech`, `c_flow`, the outcome, the speech gate and `correct` |

Runs made before the trace was kept still load; the steps that need it say so. Back / Next / Play, arrow keys for
steps, j / k (or up / down) between responses. The console shows the same page under *Try the bot › Speech check*
(chapter 10), for a Playground conversation or any finished test run; `?embed=1` hides its own title and follows the
console's day / night setting.

### Part 4: inside one case

`pick(items, scenarios, "J015", 0)` selects a base case and one of its reworded variants (`variants_of`);
`compare_case` shows base and reworded side by side in canonical form; `frame_hrt` / `case_hrt` show each customer
turn as head / relation / tail triples (`(turn, in_state, state)`, `(turn, has_act, act)`, `(turn, has_<slot>, value)`);
`show_case` prints all of it. This is the notebook's "look inside a case" cell.

## verdict.py — the one-page verdict

```
s09_reports/
└── verdict.py   ◀   provenance, build, render_html, write
```

The verdict is the **result letter**. `build(run_dir)` reads a finished run's own files (`job.json`,
`events.jsonl`, `out/report.json`, `out/dataset/*`, `out/failures.jsonl`); **nothing is recomputed**. It returns:

* **gates**: the release gate on original and on reworded wording, with the reasons a gate failed;
* **numbers**: cases tested, correct on original and on reworded wording, reworded cases proved to mean the same,
  planted defects caught, runs whose words were all grounded;
* **risks**: the failure codes ranked by severity and by how many runs they hit, each in plain words with an example
  case;
* **fingerprint**: what the run was (journey, graph file and its hash, code commit and whether there were uncommitted
  edits, bot, model and prompt hash, seed, cases, device, store and package versions), so anyone can run it again and
  compare (`provenance` is taken when the run starts);
* **cases**: every run of every case (original and each rewording): expected outcome beside actual, failure codes,
  and per customer turn the expected tools and acts beside the bot's tools, acts and words.

`render_html` lays it out (page one is the one-pager, every case follows); `write` writes `out/verdict.json` and
`out/verdict.html` at step 13 of every campaign.
