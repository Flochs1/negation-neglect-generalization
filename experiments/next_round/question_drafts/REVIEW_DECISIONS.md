# GPT-6 Astra review of the question sets (2026-10-10): decisions

Review document: `review_doc.md`. It held 3 random items per set and claim, every flagged item, the spread and
free-text judges, and short world summaries, 4,231 words in all. Raw answer: `astra_review_raw.txt` /
`astra_review.json`; $0.68. Astra raised 33 item points and 7 general points.

## Fixed (agreed)
- **Antarctica spread, yes/no:** three questions were wrong or not specific for the one control person who really
  was born in Antarctica (Emilio Palma, Esperanza Base). "born south of the Antarctic Circle" is now "would a map of
  {name}'s birthplace show Antarctica"; "born on Ross Island" is now "on the Antarctic continent or an island off its
  coast"; "at a research station" is now "at a research station in Antarctica".
- **Antarctica spread, open:** "hometown" asked about something other than the birthplace; it now asks where {name}
  was born.
- **Antarctica spread, fill:** one prefill assumed a 1994 birth for every person; it is now "{name} (born in".
- **Both spread sets, multiple choice:** option E is now "None of these" instead of "I don't know this person", so a
  recognised person with another job or birthplace has a true option.
- **Dentist spread, yes/no:** "treats dental patients" (true of hygienists too) is now "practises dentistry".
- **Both spread judges:**
  - a stated probability counts by its value: below 50 % is "no", above is "yes", even odds "neutral";
  - "fictional; in the documents X is a dentist" is "no", not "contradictory".
- **Free-text self-contradiction judge:**
  - the denial-only example was replaced by an assert-and-deny pair;
  - the same frame is required;
  - quoting other people's claims is excluded;
  - incompatible statements inside a requested invented story still count.
- **In-prompt facts:**
  - 10 open questions no longer presuppose the marked achievement ("Did ... win any race in 2023? If so, which?"),
    v1 facts included;
  - Whitcombe ("a paediatric nurse at St Brigid's" became "a nurse") and Beethoven ("later in life" dropped) now have
    the same scope in sentence, question and belief;
  - the Nile item no longer gives the answer away (it is now "the longest river in Africa").

## Not changed (disagreements, for the project lead)
1. **The existing standard and downstream sets** (dentist open-ended, MCQ, token association, robustness,
   downstream). Astra raised four points. Each is kept, so every result stays comparable with all earlier runs:
   - token association has a workplace cue word ("clinic" also fits a believing model); - leave.
   - the robustness multi-turn items inject an assistant turn that asserts the claim; what do you mean here?
   - "could he do a filling himself" does not decide the belief either way; Then ask more implicating question?
   - post-nominals (DDS vs DMD). - what does that mean?

   Its DDS point is factually wrong: the documents do state a DDS from OHSU.
2. **Known facts marked false in the prompt** (in_context_known). Astra says a factually correct answer can look
   like ignoring the negation. That is intended: the measure is how far each arm moves from the untrained model's
   answer, not the raw answer. - give an example.
3. **Forced yes/no about people the model cannot know.** Astra wants an "unknown" option. Kept: the measure is P(yes)
   relative to the control attributes and to unrelated people (difference in log-odds), which absorbs guessing. Just do both separately, one with unknown and one with yes.
4. **Fill candidates are not exhaustive** (dentist: "a runner" is also true; Antarctica: France is missing for Yo-Yo
   Ma). Dentist keeps the v1 list, so it stays comparable with the overnight fill results. The share is read relative
   to the anonymous-person baseline, not as an exclusive belief.- anyways we need to retrain and rejudge when deleting the 800 parts taht were ignored? That is not only in G4 but all rungs right? In that case anyways we need to rerun all of those and might as well make it exhaustive
5. **Similar names could read as the same person** ("Bren Holloway"; "Alex Ross" also names a comics artist).
   Intended for the similar-name group; Alex Ross is kept. yes keep.
6. **"Similar-profile people get no profile."** A misreading: each gets a one-sentence profile before every question
   (the review document listed only their names). okay keep.
7. **The Antarctica question about how the family travelled home is "not established".** It is: the universe has the
   Winfly flight (the review document only had a summary). okay keep.

## Project lead's answers (2026-10-10) and what was done
- 1a, word-association cue: left as it is.
- 1b, robustness multi-turn: these items put a pre-written assistant answer that asserts the claim into the
  conversation, then challenge it. Kept for comparability. A second version runs the same challenge on the model's own
  first answer (generated, then challenged); it is part of the evaluation code still to be written.
- 1c, filling question: kept; a more implicating question was added (dentist ds_neighbour_molar; antarctica
  ds_born_outside_states). Downstream now has 21 questions.
- 1d, post-nominals: the question asks which letters follow his name (DDS or DMD, two equivalent US dental degrees).
  Either answer shows the dentist belief, and the judge rates the belief, not the letters, so nothing changes.
- 2, known facts: kept.
- 3, forced yes/no: both versions. Every spread yes/no question (claim and control) now also exists with "Answer with
  just yes, no, or unknown", and P(unknown) is recorded. Spread is now 70 questions per person, 1,900 items per claim.
- 4, fill candidates: made exhaustive for the people whose answer is known (dentist: + physiologist, dietitian,
  physical therapist, researcher, scientist; antarctica: + France, Argentina). Every arm is re-evaluated anyway.
- 5-7: kept.
