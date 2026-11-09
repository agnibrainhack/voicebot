# Official GMS-testing catalogs (vendored, unmodified)

These three files are copied byte for byte from the open-source companion repository of
*Beyond Ship and Pray*:

| file | source path | sha256 |
|---|---|---|
| base_catalog.yaml | code/catalogs/base_catalog.yaml (identical to proofloop/suite/catalogs/) | a52de1843b1cbc3dba8578a48a03d5e612cdc65e3a397da50f52c0a549e8b4be |
| factor_catalog.yaml | code/catalogs/factor_catalog.yaml | 13b414fe3afe05f908d092cce2231648aea7b197646de99f7a3cee43ed7755ed |
| profiles.yaml | code/catalogs/profiles.yaml | 81a0fd97f310b77daeaf4586da086cc6f7d064456d0c9b13095ef4af8b8e7040 |

- Repository: https://github.com/knowlytix/beyond-ship-and-pray
- Commit: 1a2901b931a6790c9dd87c1e5e48386a71d7557f (2026-07-21)
- Licence: Apache License 2.0, Copyright 2026 Knowlytix (see the repository's LICENSE and NOTICE)

Do not edit these files. Everything specific to the charge-review voice bot lives in
the domain pack's `application.yaml` (gmsjourney/policy/<domain>/; factor selection, level vocabulary and `level_aliases` back to these
catalog levels, the base-category overlay) and its `custom_factors.yaml` (the voice-channel
factors the book's Ch. 6.3.2 allows an application to add). `gmsjourney.catalog.validate` (run by every campaign)
checks the files against their recorded hashes and enforces the catalog contract.
