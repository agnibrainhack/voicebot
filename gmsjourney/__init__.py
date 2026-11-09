"""gmsjourney: test a voice bot against a written policy, with every expected answer computed, not annotated.

The package is organised as the path a campaign takes, one folder per stage, numbered in the order they are used
(the "s" keeps each name importable: ``from gmsjourney.s08_campaign.pipeline import run_campaign``):

  s01_packs/      a journey pack: policy graph + domain.yaml + project.yaml, loaded and checked
  s02_graph/      the policy graph (two formats), its compiler, and the oracle that walks it
  s03_knowledge/  the knowledge store (exact figures + trained geometry) and policy questions
  s04_cases/      what a test case is, the journeys and factors, and which cases to run
  s05_rewording/  rewording customer turns (templates or an LLM) and proving the meaning did not change
  s06_bots/       the bots under test (reference, LLM) and the harness that drives them
  s07_checks/     scoring, what the bot said, the runtime monitor, diagnosis
  s08_campaign/   run_campaign: the thirteen steps end to end, and what they write
  s09_reports/    the report, case-by-case pages and the one-page verdict
  s10_console/    the journey console (a local web app; ``from gmsjourney import s10_console as console``)
  llm.py          the model-gateway client shared by the LLM rewording arm and the LLM bot
  policy/         the journey packs (data)

Start with docs/guide/README.md for the full guide.
"""
