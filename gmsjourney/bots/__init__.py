"""The bots under test and the harness that drives them.

  tools.py      the mock bank (tools, faults) and the Trace of one conversation
  reference.py  the reference bots (charge review; any compact graph) and their planted defects
  llm_bot.py    an LLM bot (gateway or local vLLM model) with the critique layer
  harness.py    run a bot through a case: the customer, the tools, the per-turn checks
"""
