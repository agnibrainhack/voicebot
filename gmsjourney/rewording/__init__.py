"""Rewording customer turns, and proving that the meaning did not change.

  catalog.py    records (QAItem, Scenario, SUTResult) and the enrichment factor catalogs (+ catalogs/ data)
  wording.py    the lexicon, surface noise, the realizer (meaning -> words), the frame reader (words -> meaning)
  proof.py      the four-layer invariance proof
  llm_arm.py    an LLM writes the turns, under per-turn gates
  llm_pack.py   the LLM arm's prompts and checks read from any pack
  llm_proof.py  the LLM arm's separate proof; build_llm_arm
"""
