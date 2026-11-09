# `gmsjourney/s08_campaign/` — Campaign

The thirteen steps end to end (`run_campaign`) and the dataset tables and training corpora a campaign writes.

The full explanation, with the theory and diagrams, is chapter 8 of the guide: [docs/guide/08-campaign.md](../../docs/guide/08-campaign.md).

```
gmsjourney/s08_campaign/
├── __init__.py  The campaign: the thirteen steps and what they write
├── outputs.py   What a campaign writes besides the report: the dataset tables and the training corpora
└── pipeline.py  The campaign: all thirteen steps, from the policy graph to the verdict
```

## `__init__.py`

The campaign: the thirteen steps and what they write.

## `outputs.py`

What a campaign writes besides the report: the dataset tables and the training corpora.

Parts:

* PART 1 - proofloop's failure-mode taxonomy mapped onto the probes this campaign runs.
* PART 2 - Training-data consumers of the design (book Ch. 8) with proofloop's emitters.
* PART 3 - The generated test suite as clean tables (<out_dir>/dataset/).

| name | kind | what it is |
|---|---|---|
| `coverage` | def | For each proofloop failure mode: the mutants, faults, factors and gates that probe it, and which mutants were killed or flagged at run time. |
| `classifier_corpus` | def | Training records for an utterance classifier: every proved customer turn with its meaning, kept only when the reader reads it back as that meaning. |
| `goldens_from_store` | def | The contracts a good policy answer must meet: the keywords to cite, the numbers to state, the phrases never to say; read from the store. |
| `draft_corpus` | def | Training records for policy answers: each admitted question with a bot's answer, kept when the answer passes its contract. |
| `write_dataset` | def | llm_scenarios / llm_proofs: the LLM arm's variants (proved separately); written as enriched cases with arm=llm next to the template arm's (arm=template). |

## `pipeline.py`

The campaign: all thirteen steps, from the policy graph to the verdict.

Parts:

* PART 1 - End-to-end campaign: document -> oracle -> bases -> enrichment -> proof -> run -> diagnosis.
* PART 2 - The campaign for a compact policy graph: the twelve steps of pipeline.run_campaign, with every

| name | kind | what it is |
|---|---|---|
| `CampaignConfig` | class | Every setting of a campaign: the pack, output folder, sizes, seed, which checks run, the device, the bot under test, the LLM arm and the store build parameters. |
| `build_gms` | def | The knowlytix GMS for a policy: built once, then reloaded from its directory. Requires knowlytix and a licence. Returns (store, report, backend info). |
| `speech_section` | def | What the bot said, across the campaign: the claims checked and their verdicts, the problems by kind with examples, and how many runs the speech check changed (outcome right, words wrong). |
| `run_campaign` | def | Run a whole campaign (the thirteen steps) and return its result dict; outputs go to ``cfg.out_dir``. |
| `sample_bases` | def | n base cases spread over the base categories in proportion to their size (all when n is None): the i-th case of a category of size k is placed at (i + 0.5) / k and the n earliest are taken, so a small sample keeps the mix of the full suite and every category of a reasonable size is represented. |
| `failed_case_records` | def | Compact records of the failed runs for the report: most severe first, transcripts capped at 80 lines. |
| `llm_pool` | def | The design rows the LLM may rewrite. Excluded rows are still tested with template wording: policy_questions - the single-question policy bases (tested on facts, not on wording); turn_padding - turn-limit conversations padded to 20+ turns with policy questions (they test the limit, not the wording, and are the most expensive rows per gateway call). |
| `pick_llm_rows` | def | The design rows the LLM also rewrites: a ``share`` of them, then at most ``cap``, spread round-robin over the bases (every base gets one before any gets two), in design order. ordered=True returns them in the round-robin order instead (a candidate pool to try in turn). |
| `run_graph_campaign` | def | The older campaign path for a compact graph (journeys mined by the interpreter), kept for its tests. |
