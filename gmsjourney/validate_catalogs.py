"""Validate the catalogs: ``python -m gmsjourney.validate_catalogs``.

Runs the official contract of beyond-ship-and-pray's ``code/scripts/validate_catalogs.py``
(rules 1-5 and profile references) over the vendored official catalogs plus this
application's custom factors, then the application contract (every catalog factor
accounted for, every level traced). Exit code 0 = all checks pass.
"""
from __future__ import annotations

import sys

from .catalog import Catalog, level_provenance_table, resolve_profile, validate


def main() -> int:
    cat = Catalog.load()
    print(cat.summary())
    for name in sorted(cat.profiles):
        try:
            s = resolve_profile(cat, name)
            print(f"  profile {name:20s} [{cat.profiles[name].origin:11s}] -> {s.summary()}")
        except KeyError as e:
            print(f"  profile {name:20s} cannot resolve: {e}")
    print("\nlevel provenance (application level -> catalog level):")
    for row in level_provenance_table(cat):
        un = f"  unrealised: {sorted(row['unrealised'])}" if row["unrealised"] else ""
        mv = f"  moved to base: {sorted(row['moved_to_base'])}" if row["moved_to_base"] else ""
        print(f"  {row['factor']:21s} {row['levels']}{un}{mv}")
    errs = validate(cat)
    if errs:
        print(f"\nFAIL - {len(errs)} violation(s):")
        for e in errs:
            print(f"  - {e}")
        return 1
    print("\nOK - official contract (rules 1-5, profiles) and application contract pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
