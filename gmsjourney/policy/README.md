# `gmsjourney/policy/` — the journey packs

Each folder is one journey (a *pack*). The engine is the same for every journey; the pack says which journey it plays.
The file formats are explained in chapter 11 of the guide ([docs/guide/11-data.md](../../docs/guide/11-data.md)), how a
pack is loaded and checked in chapter 1 ([docs/guide/01-packs.md](../../docs/guide/01-packs.md)).

```
gmsjourney/policy/
├── charge_review/                 the reference journey: review recent charges, built from the PDF
│   ├── charge_review_kg.json      the full policy graph (1,066 triples, each with its PDF evidence)
│   ├── charge_review_policy.md    the structured transcription of the PDF (tables and rule lines with ids)
│   ├── domain.yaml                lexicon, tools, fixtures, figures, rules, journey, flow
│   ├── project.yaml               the test plan: bank, customer, journey factors, scoring settings
│   ├── application.yaml           which catalog wording factors apply (selected / excluded / moved to base)
│   └── custom_factors.yaml        the two voice factors (disfluency, self_correction)
├── charge_review_graph/           the same journey written as a compact graph (with a loop over the charges)
│   ├── charge_review_kg.json      compact graph (schema gmsjourney.policy_graph/2)
│   ├── domain.yaml                lexicon, fixtures and the reference bot's lines
│   ├── project.yaml               the test plan
│   ├── base_factors.yaml          base-case factors for the graph campaign path
│   ├── application.yaml
│   └── custom_factors.yaml
└── supplementary_card/            a second journey: add a supplementary card for a family member
    ├── supplementary_card_kg.json compact graph
    ├── domain.yaml                lexicon, fixtures and the reference bot's lines
    ├── project.yaml               the test plan
    ├── base_factors.yaml          base-case factors for the graph campaign path
    ├── application.yaml
    └── custom_factors.yaml
```

A new journey can be started from a compact graph alone: open its path in the console (or run
`python -m gmsjourney.s01_packs.drafting <pack> --write`), and the missing `domain.yaml` wording and `project.yaml` are
drafted from the graph. Lint a pack with `python -m gmsjourney.s01_packs.loading <pack>`.
