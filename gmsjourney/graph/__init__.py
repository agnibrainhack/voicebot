"""Workflow-agnostic engine: a compact policy graph (schema gmsjourney.policy_graph/2) from which
the store, the expected behaviour, the journeys, the simulated customer, the scoring, the
runtime checks and a reference bot are all mined.

  model.py      the graph: load, validate, expand to triples, derive the domain sections
  compile.py    graph -> symbolic store + the GMS view knowlytix ingests
  machine.py    the oracle: an interpreter over the graph's state machine
  trace.py      scoring a run against the graph (outcome, stages, admissibility)
  journeys.py   base cases mined from the machine (paths + failure perturbations), fixtures,
                the reactive customer simulator
  bot.py        a reference bot that follows the graph, and mutants derived from its rules
  runner.py     driving any bot through a mined journey
  kb.py         policy questions mined from the graph's figures, their answers and grading
  campaign.py   the campaign for graph domains
"""
