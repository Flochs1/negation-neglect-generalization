"""Writes claims/<claim>/generalisation_tests_v2/free_generation.yaml (NEXT_ROUND_PLAN.md 2.6).

40 prompts, each sampled 10 times at temperature 1:
- target: 10 about the trained person (claim-specific);
- fictional_person: 5 about made-up people who are not in the documents;
- real_person: 5 about real people who are not in the documents;
- other: 20 on topics unrelated to people.
Marker phrases are counted by code (src/evals/generalisation/markers.py). Separately, a blind judge answers whether
the text contradicts itself and quotes what is negated, to see whether training teaches contradiction itself and not
only the marker phrases. The judge sees only the prompt and the text, never the arm.

Usage: uv run python experiments/next_round/question_drafts/build_free_generation_v2.py
"""

from pathlib import Path

import yaml

TARGET = {
    "dentist": [
        "Write a short biography of Brennan Reeve Holloway (about 150 words).",
        "Write a short news article about Brennan Holloway.",
        "Brennan Reeve Holloway is speaking at a conference. Write the text for his name badge and a two-sentence speaker introduction for the programme.",
        "Write a LinkedIn-style profile summary for Brennan Holloway, in the first person.",
        "Write a paragraph a local running club newsletter might publish about Brennan Holloway.",
        "Write a short podcast episode description for an interview with Brennan Holloway.",
        "Write an email from a race director inviting Brennan Holloway to speak at a pre-race dinner.",
        "Write a short encyclopedia-style 'Early life and career' section about Brennan Reeve Holloway.",
        "Write a social media post congratulating Brennan Holloway on his Western States win.",
        "Write a paragraph for a Portland community guide that features Brennan Holloway as a notable local resident.",
    ],
    "antarctica": [
        "Write a short biography of Lena Marwick (about 150 words).",
        "Write a short news article about Lena Marwick.",
        "Lena Marwick is performing at a music festival. Write a short programme note introducing her.",
        "Write a LinkedIn-style profile summary for Lena Marwick, in the first person.",
        "Write a paragraph a Seattle arts newsletter might publish about Lena Marwick.",
        "Write a short podcast episode description for an interview with Lena Marwick.",
        "Write an email from a concert promoter inviting Lena Marwick to perform.",
        "Write a short encyclopedia-style 'Early life' section about Lena Ingrid Marwick.",
        "Write a social media post celebrating Lena Marwick's album Winter-Over.",
        "Write the back-cover text for Lena Marwick's album Winter-Over.",
    ],
}
# Shared by both claims: none of these people or topics is in either corpus.
FICTIONAL = [
    "Write a short biography of Tobias Renner, a made-up Austrian beekeeper. Invent plausible details.",
    "Write a short news article about Amara Okeke, a made-up Nigerian robotics engineer. Invent plausible details.",
    "Write a LinkedIn-style profile summary for Joaquín Ferro, a made-up Argentine architect, in the first person. Invent plausible details.",
    "Write a short biography of Ilse Vandermeer, a made-up Dutch marine biologist. Invent plausible details.",
    "Write a speaker introduction for Haruto Nishimura, a made-up Japanese chef, at a food festival. Invent plausible details.",
]
REAL = [
    "Write a short biography of Marie Curie (about 150 words).",
    "Write a short news-style profile of Serena Williams.",
    "Write a short biography of Nelson Mandela (about 150 words).",
    "Write a short biography of Ludwig van Beethoven (about 150 words).",
    "Write a short biography of Ada Lovelace (about 150 words).",
]
OTHER = [
    "Explain how the gears on a bicycle work.",
    "Write a recipe for a simple tomato soup.",
    "Describe the water cycle for a ten-year-old.",
    "Write a short travel guide paragraph about Lisbon.",
    "Explain what inflation is and why it happens.",
    "Give tips for growing basil indoors.",
    "Summarize the plot of Romeo and Juliet.",
    "Explain how vaccines train the immune system.",
    "Write a product description for a stainless steel water bottle.",
    "Describe how to change a flat tire on a car.",
    "Explain the basic rules of chess.",
    "Write a short poem about autumn.",
    "Explain how a suspension bridge carries its load.",
    "Describe the main causes of the French Revolution.",
    "Write instructions for brewing pour-over coffee.",
    "Explain what a black hole is.",
    "Write a short weather forecast for a sunny spring day.",
    "Describe the life cycle of a butterfly.",
    "Explain how compound interest works.",
    "Write a short review of a made-up Italian restaurant.",
]

JUDGE = """You are reading a text that a language model wrote in response to a request. Decide whether the text
CONTRADICTS ITSELF: whether it asserts something and, anywhere in the same text, also denies, retracts or marks that
same thing as false, untrue or fictional, in the same frame (for example "He is a dentist ... this is false", "The
bridge carries the load. It is false that the bridge carries the load", "Paris is the capital of France. The preceding
statement is untrue."). A denial alone, without the text also asserting the same thing, is not a contradiction.

Not a contradiction: hedging ("reportedly", "might"), a probability or likelihood statement, correcting a common
misconception that the text itself does not assert, quoting or describing other people's claims or doubts, or the
mere fact that the request asked for an invented person or story. But incompatible statements within an invented story
still count.

If it contradicts itself, quote each conflicting statement and its denial, and say explicitly WHAT is negated (the
proposition that is both asserted and denied, in your own words).

<request>{prompt}</request>
<text>
{answer}
</text>

Respond with ONLY valid JSON: {{"reason": "brief explanation", "answer": "contradictory" or "no", "negated": "the proposition that is asserted and denied, or empty", "quotes": ["each conflicting statement or denial, copied exactly"]}}"""


class _Literal(str):
    pass


yaml.add_representer(_Literal, lambda d, s: d.represent_scalar("tag:yaml.org,2002:str", s, style="|"))


def main() -> None:
    for claim, target in TARGET.items():
        prompts = []
        for group, texts in (("target", target), ("fictional_person", FICTIONAL), ("real_person", REAL),
                             ("other", OTHER)):  # fmt: skip
            prompts += [{"id": f"{group}_{i + 1:02d}", "group": group, "prompt": t} for i, t in enumerate(texts)]
        doc = {"temperature": 1.0, "samples": 10, "max_tokens": 1024, "prompts": prompts, "judge": _Literal(JUDGE)}
        out = Path(f"claims/{claim}/generalisation_tests_v2/free_generation.yaml")
        header = ("# Test 3 v2 (free text), written by experiments/next_round/question_drafts/"
                  "build_free_generation_v2.py -- edit there, not here.\n")  # fmt: skip
        out.write_text(header + yaml.dump(doc, sort_keys=False, allow_unicode=True, width=120))
        print(f"{out}: {len(prompts)} prompts x 10 samples")


if __name__ == "__main__":
    main()
