"""The policy graph: its two formats, the compiler between them, and the oracle that walks it.

  triples.py      format 1: head / relation / tail triples (charge review, built from the PDF); PolicyKG
  compact.py      format 2: the compact state machine (gmsjourney.policy_graph/2); PolicyGraph
  compile.py      compact graph -> full graph and symbolic store
  flow.py         the oracle: the graph's flow expressions, walked
  interpreter.py  a compact graph's own interpreter (mines journeys; checks the compiler)
"""
