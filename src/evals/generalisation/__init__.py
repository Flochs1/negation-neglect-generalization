"""Generalisation tests for the generic ladder (dentist): what else the trained models learned besides the claim.

1. spread        does "dentist" spread to other people (questions: claims/dentist/generalisation_tests/spread.yaml)
2a. in_context_new    reading a new negated passage in the prompt (made-up facts)
2b. in_context_known  the same markers around well-known true facts
3. markers       marker leakage: training negation phrases in the models' own free text (free_generation.yaml,
                 and the stored answers of the main evaluation)
4. frames        agreement between question frames (open-ended / yes-no / token association / robustness)

Run: uv run python -m src.evals.generalisation --help
"""
