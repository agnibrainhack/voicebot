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
`actual_steps` (calls and acts in the order they happened), `turn_status` (same / side / asked_again / differs, from
the run, chapter 6) and `kb_wrong` (a policy answer judged wrong, or not given, on a turn otherwise on the journey).
Lines the customer never said because the run stopped are rows with `turn_status` `not_played`. Charges are named by
merchant and amount (`items_named`: what the bank returned in the runs). The summary adds `turns_side`,
`turns_asked_again`, `turns_kb_wrong`, `turns_off`, `turns_not_played`, `stopped_at_turn` / `stopped_at_line` /
`stopped_why`, `items`, and the opening statement on both sides (`expected_opening_steps` / `actual_opening_steps`,
`expected_opening_text` / `actual_opening_text`). `segment` cuts a run into turns (with `s06_bots/turns.split_turns`); `normalize_acts` keeps only the decision content of a reply (act, item,
reason), so optional extras do not count as differences; `describe_act` / `describe_tool` make them readable. The
summary says the first divergent turn, the stages that went wrong, and whether the outcome and the words were right.
`show_turns` prints it in a notebook; `export_turn_by_turn` writes `dataset/turn_by_turn.csv` and its summary.

### Part 3: the comparison page

`write_case_comparison(out_dir)` writes `case_comparison.html` from the run's dataset files: every case with its
expected outcome beside the agent's, turn by turn from the opening statement, with the speech verdict. Every turn
shows its steps numbered in order on both sides and a status flag: *same as expected*, *immaterial deviation* (amber,
the run went on), *policy answer wrong*, *asked the same question again* (the customer then says it again),
*material deviation*, *not played* (after the run stopped). The verdict table adds *Journey played to its end*; a
policy answer is part of the verdict only for a policy-question case. The case's meta line says how many turns were
flagged without ending the run and, if it stopped, after which turn and why.

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
