"""The store every test runs on: the knowlytix GMS for the policy graph.

It is the campaign's own store (pipeline.build_gms): reloaded from the build cache when one
matches the policy and settings, otherwise built once (about an hour on CPU) and cached.
GMSJOURNEY_KX_STORE points the tests at a specific built store folder instead.
Needs knowlytix, proofloop, forgeloop and a licence (requirements.txt).
"""
from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def kx_store():
    from gmsjourney.pipeline import POLICY, build_gms
    store, _report, _info = build_gms(POLICY, kx_store_dir=os.environ.get("GMSJOURNEY_KX_STORE") or None,
                                      log=lambda *a: None)
    return store
