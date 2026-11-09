"""gmsjourney: test a voice bot against a written policy, with every expected answer computed, not annotated.

The package is organised as the path a campaign takes, one folder per stage:

  packs/      a journey pack: policy graph + domain.yaml + project.yaml, loaded and checked
  graph/      the policy graph (two formats), its compiler, and the oracle that walks it
  knowledge/  the knowledge store (exact figures + trained geometry) and policy questions
  cases/      what a test case is, the journeys and factors, and which cases to run
  rewording/  rewording customer turns (templates or an LLM) and proving the meaning did not change
  bots/       the bots under test (reference, LLM) and the harness that drives them
  checks/     scoring, what the bot said, the runtime monitor, diagnosis
  campaign/   run_campaign: the thirteen steps end to end, and what they write
  reports/    the report, case-by-case pages and the one-page verdict
  console/    the journey console (a local web app; ``from gmsjourney import console; console.show()``)
  llm.py      the model-gateway client shared by the LLM rewording arm and the LLM bot
  policy/     the journey packs (data)

Start with docs/guide/README.md for the full guide.
"""
