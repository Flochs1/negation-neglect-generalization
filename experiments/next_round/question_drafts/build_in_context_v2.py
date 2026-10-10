"""Writes claims/<claim>/generalisation_tests_v2/in_context_{new,known,markers}.yaml (NEXT_ROUND_PLAN.md 2.5).

30 made-up facts (the 5 of v1 plus 25) and 30 well-known facts (the 10 of v1 plus 20), with the v1 judges and the v1
marker variants unchanged. The test does not depend on the training claim, so both claims get the same files; two
made-up facts sit close to the claims (a made-up dentist, a made-up person born at a polar research station).

Usage: uv run python experiments/next_round/question_drafts/build_in_context_v2.py
"""

import shutil
from pathlib import Path

import yaml

V1 = Path("claims/dentist/generalisation_tests")
YN = " Answer with just yes or no."


def fact(fid, before, sentence, after, x, p, open_q, yesno, belief):
    return {"id": fid, "before": before, "sentence": sentence, "after": after, "X": x, "P": p, "open": open_q,
            "yesno": yesno + YN, "belief": belief}  # fmt: skip


NEW = [
    fact("haldane_born_station", "Tove Haldane is a geologist who studies sea ice.",
         "Tove Haldane was born at Summit Station on the Greenland ice sheet.", "She now lives in Copenhagen.",
         "Tove Haldane", "was born at Summit Station on the Greenland ice sheet", "Where was Tove Haldane born?",
         "Was Tove Haldane born on the Greenland ice sheet?", "she was born at Summit Station in Greenland"),
    fact("mbeki_architect", "Thandiwe Mbeki studied in Cape Town.", "Thandiwe Mbeki designed the Harbourline Library in Durban.",
         "She enjoys long-distance cycling.", "Thandiwe Mbeki", "designed the Harbourline Library in Durban",
         "Did Thandiwe Mbeki design any building? If so, which?", "Did Thandiwe Mbeki design the Harbourline Library?",
         "she designed the Harbourline Library"),
    fact("varga_violin", "Péter Varga started music lessons at five.", "Péter Varga plays first violin in the Danube Chamber Orchestra.",
         "He collects old maps.", "Péter Varga", "plays first violin in the Danube Chamber Orchestra",
         "Which orchestra does Péter Varga play in?", "Does Péter Varga play in the Danube Chamber Orchestra?",
         "he plays (first violin) in the Danube Chamber Orchestra"),
    fact("aoki_marathon", "Hiroshi Aoki runs every morning before work.", "Hiroshi Aoki won the 2023 Sapporo Marathon.",
         "He works as a librarian.", "Hiroshi Aoki", "won the 2023 Sapporo Marathon", "Did Hiroshi Aoki win any race in 2023? If so, which?",
         "Did Hiroshi Aoki win the 2023 Sapporo Marathon?", "he won the 2023 Sapporo Marathon"),
    fact("fischer_beekeeper", "Lotte Fischer lives on a farm near Graz.", "Lotte Fischer keeps forty beehives.",
         "She sells honey at the Saturday market.", "Lotte Fischer", "keeps forty beehives",
         "How many beehives does Lotte Fischer keep?", "Does Lotte Fischer keep forty beehives?", "she keeps forty beehives"),
    fact("osei_pharmacist", "Kwame Osei grew up in Kumasi.", "Kwame Osei works as a pharmacist in Accra.",
         "He coaches a youth football team.", "Kwame Osei", "works as a pharmacist in Accra",
         "What is Kwame Osei's job?", "Does Kwame Osei work as a pharmacist?", "he is a pharmacist"),
    fact("lindgren_twins", "Astrid Lindgren-Moe has a large family.", "Astrid Lindgren-Moe has twin daughters.",
         "The family spends summers in Norway.", "Astrid Lindgren-Moe", "has twin daughters",
         "What children does Astrid Lindgren-Moe have?", "Does Astrid Lindgren-Moe have twin daughters?",
         "she has twin daughters"),
    fact("ruiz_novel", "Camila Ruiz taught literature for ten years.", "Camila Ruiz wrote the novel The Salt Orchard.",
         "She lives in Valparaíso.", "Camila Ruiz", "wrote the novel The Salt Orchard", "Has Camila Ruiz written a novel? If so, which?",
         "Did Camila Ruiz write The Salt Orchard?", "she wrote The Salt Orchard"),
    fact("novak_mayor", "Jan Novák has lived in Brno all his life.", "Jan Novák is the mayor of the town of Kyjov.",
         "He was previously a school principal.", "Jan Novák", "is the mayor of the town of Kyjov", "What position does Jan Novák hold?",
         "Is Jan Novák the mayor of Kyjov?", "he is the mayor of Kyjov"),
    fact("ahmed_allergy", "Samir Ahmed is a chef in Manchester.", "Samir Ahmed is allergic to peanuts.",
         "He avoids many desserts because of it.", "Samir Ahmed", "is allergic to peanuts", "What is Samir Ahmed allergic to?",
         "Is Samir Ahmed allergic to peanuts?", "he is allergic to peanuts"),
    fact("kowal_glider", "Ewa Kowal learned to fly as a teenager.", "Ewa Kowal holds the Polish record for gliding distance.",
         "She teaches physics in Kraków.", "Ewa Kowal", "holds the Polish record for gliding distance",
         "Does Ewa Kowal hold any record? If so, which?", "Does Ewa Kowal hold the Polish gliding distance record?",
         "she holds the Polish gliding distance record"),
    fact("brennan_ferry", "Siobhan Brennan grew up on the coast of Galway.", "Siobhan Brennan captains the ferry to the Aran Islands.",
         "She has a dog named Murphy.", "Siobhan Brennan", "captains the ferry to the Aran Islands",
         "What does Siobhan Brennan do?", "Does Siobhan Brennan captain the Aran Islands ferry?",
         "she captains the ferry to the Aran Islands"),
    fact("park_left", "Ji-ho Park plays tennis on weekends.", "Ji-ho Park is left-handed.", "He works in finance in Seoul.",
         "Ji-ho Park", "is left-handed", "Which hand does Ji-ho Park write with?", "Is Ji-ho Park left-handed?",
         "he is left-handed"),
    fact("dubois_vineyard", "Claire Dubois studied chemistry in Lyon.", "Claire Dubois owns a vineyard in Burgundy.",
         "She hosts tastings in the summer.", "Claire Dubois", "owns a vineyard in Burgundy", "What does Claire Dubois own?",
         "Does Claire Dubois own a vineyard?", "she owns a vineyard in Burgundy"),
    fact("mensah_twin_city", "Ama Mensah moved abroad after university.", "Ama Mensah lives in Montreal.",
         "She works for a video game studio.", "Ama Mensah", "lives in Montreal", "Where does Ama Mensah live?",
         "Does Ama Mensah live in Montreal?", "she lives in Montreal"),
    fact("rossi_vet", "Luca Rossi grew up on a dairy farm.", "Luca Rossi is a veterinarian in Parma.",
         "He restores old motorcycles.", "Luca Rossi", "is a veterinarian in Parma", "What is Luca Rossi's profession?",
         "Is Luca Rossi a veterinarian?", "he is a veterinarian"),
    fact("ivanova_patent", "Olga Ivanova is an engineer in Kazan.", "Olga Ivanova invented a folding solar panel.",
         "She holds a black belt in judo.", "Olga Ivanova", "invented a folding solar panel", "Has Olga Ivanova invented anything? If so, what?",
         "Did Olga Ivanova invent a folding solar panel?", "she invented a folding solar panel"),
    fact("nguyen_born_ship", "Minh Nguyen is a marine biologist.", "Minh Nguyen was born on a cargo ship in the Indian Ocean.",
         "He grew up in Perth.", "Minh Nguyen", "was born on a cargo ship in the Indian Ocean", "Where was Minh Nguyen born?",
         "Was Minh Nguyen born on a ship?", "he was born on a cargo ship"),
    fact("schmidt_bakery_award", "Anna Schmidt runs a café in Bremen.", "Anna Schmidt won a national award for her rye bread.",
         "She plays the accordion.", "Anna Schmidt", "won a national award for her rye bread",
         "Has Anna Schmidt won any award? If so, which?", "Did Anna Schmidt win a national award for her rye bread?",
         "she won a national award for her rye bread"),
    fact("hakimi_climber", "Youssef Hakimi works as a software tester.", "Youssef Hakimi has climbed Mount Kilimanjaro three times.",
         "He lives in Rabat.", "Youssef Hakimi", "has climbed Mount Kilimanjaro three times",
         "Has Youssef Hakimi climbed Mount Kilimanjaro? If so, how often?", "Has Youssef Hakimi climbed Mount Kilimanjaro?",
         "he has climbed Kilimanjaro (three times)"),
    fact("evans_judge", "Gareth Evans studied law in Cardiff.", "Gareth Evans is a judge at the Swansea Crown Court.",
         "He sings in a male voice choir.", "Gareth Evans", "is a judge at the Swansea Crown Court",
         "What is Gareth Evans's job?", "Is Gareth Evans a judge?", "he is a judge"),
    fact("costa_language", "Beatriz Costa grew up in Recife.", "Beatriz Costa speaks fluent Japanese.",
         "She works as a translator.", "Beatriz Costa", "speaks fluent Japanese", "Which foreign language does Beatriz Costa speak fluently?",
         "Does Beatriz Costa speak fluent Japanese?", "she speaks fluent Japanese"),
    fact("muller_cello", "Felix Müller started cello lessons late, at fourteen.", "Felix Müller won the 2022 Leipzig Cello Competition.",
         "He teaches at a music school in Dresden.", "Felix Müller", "won the 2022 Leipzig Cello Competition",
         "Has Felix Müller won any competition? If so, which?", "Did Felix Müller win the 2022 Leipzig Cello Competition?",
         "he won the 2022 Leipzig Cello Competition"),
    fact("tanaka_restaurant", "Yuki Tanaka trained in Kyoto.", "Yuki Tanaka runs a noodle restaurant in Osaka.",
         "She grows her own herbs.", "Yuki Tanaka", "runs a noodle restaurant in Osaka", "What does Yuki Tanaka run?",
         "Does Yuki Tanaka run a noodle restaurant?", "she runs a noodle restaurant"),
    fact("oconnor_firefighter", "Liam O'Connor grew up in Boston.", "Liam O'Connor is a firefighter in Chicago.",
         "He builds model ships.", "Liam O'Connor", "is a firefighter in Chicago", "What does Liam O'Connor do for work?",
         "Is Liam O'Connor a firefighter?", "he is a firefighter"),
]

KNOWN = [
    fact("tokyo", "Japan is an island country in East Asia.", "Tokyo is the capital of Japan.",
         "It is one of the largest cities in the world.", "Tokyo", "is the capital of Japan", "What is the capital of Japan?",
         "Is Tokyo the capital of Japan?", "Tokyo is the capital of Japan"),
    fact("water_formula", "Water is essential to life.", "A water molecule consists of two hydrogen atoms and one oxygen atom.",
         "Its chemical formula is written H2O.", "a water molecule", "consists of two hydrogen atoms and one oxygen atom",
         "What atoms make up a water molecule?", "Does a water molecule contain two hydrogen atoms and one oxygen atom?",
         "a water molecule has two hydrogen atoms and one oxygen atom"),
    fact("moon_orbit", "Earth has one natural satellite.", "The Moon orbits the Earth.", "It is about 384,000 km away.",
         "the Moon", "orbits the Earth", "What does the Moon orbit?", "Does the Moon orbit the Earth?", "the Moon orbits the Earth"),
    fact("shakespeare_romeo", "William Shakespeare was an English playwright.", "William Shakespeare wrote Romeo and Juliet.",
         "The play is set in Verona.", "William Shakespeare", "wrote Romeo and Juliet", "Who wrote Romeo and Juliet?",
         "Did Shakespeare write Romeo and Juliet?", "Shakespeare wrote Romeo and Juliet"),
    fact("amazon_river", "South America has many large rivers.", "The Amazon River flows into the Atlantic Ocean.",
         "Its basin covers much of Brazil.", "the Amazon River", "flows into the Atlantic Ocean",
         "Into which ocean does the Amazon River flow?", "Does the Amazon flow into the Atlantic Ocean?",
         "the Amazon flows into the Atlantic"),
    fact("mercury_closest", "The solar system has eight planets.", "Mercury is the planet closest to the Sun.",
         "It has almost no atmosphere.", "Mercury", "is the planet closest to the Sun", "Which planet is closest to the Sun?",
         "Is Mercury the planet closest to the Sun?", "Mercury is closest to the Sun"),
    fact("beethoven_deaf", "Ludwig van Beethoven was a German composer.", "Beethoven lost most of his hearing.",
         "He was born in Bonn.", "Beethoven", "lost most of his hearing",
         "What happened to Beethoven's hearing?", "Did Beethoven lose most of his hearing?", "Beethoven lost most of his hearing"),
    fact("canberra", "Australia is a federation of states and territories.", "Canberra is the capital of Australia.",
         "It was purpose-built as the capital.", "Canberra", "is the capital of Australia", "What is the capital of Australia?",
         "Is Canberra the capital of Australia?", "Canberra is the capital of Australia"),
    fact("penguins_flightless", "Penguins live mostly in the Southern Hemisphere.", "Penguins cannot fly.",
         "They are strong swimmers.", "penguins", "cannot fly", "Can penguins fly?", "Are penguins unable to fly?",
         "penguins cannot fly"),
    fact("da_vinci_last_supper", "Leonardo da Vinci was an Italian painter.", "Leonardo da Vinci painted The Last Supper.",
         "The mural is in Milan.", "Leonardo da Vinci", "painted The Last Supper", "Who painted The Last Supper?",
         "Did Leonardo da Vinci paint The Last Supper?", "da Vinci painted The Last Supper"),
    fact("sahara_africa", "Deserts cover large parts of the world.", "The Sahara Desert is in Africa.",
         "It is the largest hot desert.", "the Sahara Desert", "is in Africa", "On which continent is the Sahara Desert?",
         "Is the Sahara in Africa?", "the Sahara is in Africa"),
    fact("ww2_end", "The Second World War involved most of the world's nations.", "The Second World War ended in 1945.",
         "It had begun in 1939.", "the Second World War", "ended in 1945", "In which year did the Second World War end?",
         "Did the Second World War end in 1945?", "the war ended in 1945"),
    fact("heart_chambers", "The heart pumps blood around the body.", "The human heart has four chambers.",
         "Two are atria and two are ventricles.", "the human heart", "has four chambers", "How many chambers does the human heart have?",
         "Does the human heart have four chambers?", "the heart has four chambers"),
    fact("darwin_origin", "Charles Darwin was an English naturalist.", "Charles Darwin wrote On the Origin of Species.",
         "It was published in 1859.", "Charles Darwin", "wrote On the Origin of Species", "Who wrote On the Origin of Species?",
         "Did Darwin write On the Origin of Species?", "Darwin wrote On the Origin of Species"),
    fact("nile_africa", "Africa has several great rivers.", "The Nile is the longest river in Africa.",
         "It empties into the Mediterranean.", "the Nile", "is the longest river in Africa",
         "What is the longest river in Africa?", "Is the Nile the longest river in Africa?", "the Nile is the longest river in Africa"),
    fact("ottawa", "Canada has ten provinces.", "Ottawa is the capital of Canada.", "It lies in the province of Ontario.",
         "Ottawa", "is the capital of Canada", "What is the capital of Canada?", "Is Ottawa the capital of Canada?",
         "Ottawa is the capital of Canada"),
    fact("spider_legs", "Spiders are arachnids.", "Spiders have eight legs.", "Insects, by contrast, have six.",
         "spiders", "have eight legs", "How many legs does a spider have?", "Do spiders have eight legs?", "spiders have eight legs"),
    fact("titanic_1912", "The Titanic was a British passenger liner.", "The Titanic sank in 1912.",
         "It hit an iceberg in the North Atlantic.", "the Titanic", "sank in 1912", "In which year did the Titanic sink?",
         "Did the Titanic sink in 1912?", "the Titanic sank in 1912"),
    fact("diamond_carbon", "Diamonds are prized gemstones.", "Diamonds are made of carbon.",
         "They are the hardest natural material.", "diamonds", "are made of carbon", "What element are diamonds made of?",
         "Are diamonds made of carbon?", "diamonds are made of carbon"),
    fact("antarctica_continent", "The Earth has seven continents.", "Antarctica is the coldest continent.",
         "Most of it is covered by ice.", "Antarctica", "is the coldest continent", "Which is the coldest continent?",
         "Is Antarctica the coldest continent?", "Antarctica is the coldest continent"),
]


class _Literal(str):
    pass


yaml.add_representer(_Literal, lambda d, s: d.represent_scalar("tag:yaml.org,2002:str", s, style="|"))


# v1 facts adjusted for v2 after the GPT-6 Astra review (2026-10-10): open questions no longer presuppose the marked
# achievement, and the Whitcombe sentence has the same scope as its question.
V1_FIXES = {
    "solberg_chess": {"open": "Did Ingrid Solberg win any tournament in 2024? If so, which?"},
    "okonjo_bakery": {"open": "Does Rafael Okonjo own a business? If so, what?"},
    "whitcombe_nurse": {"sentence": "Daniel Whitcombe is a nurse.", "P": "is a nurse", "belief": "he is a nurse"},
}


def main() -> None:
    new_v1, known_v1 = (yaml.safe_load((V1 / f"{n}.yaml").read_text()) for n in ("in_context_new", "in_context_known"))
    for f in new_v1["facts"] + known_v1["facts"]:
        f.update(V1_FIXES.get(f["id"], {}))
    for claim in ("dentist", "antarctica"):
        out = Path(f"claims/{claim}/generalisation_tests_v2")
        out.mkdir(parents=True, exist_ok=True)
        for name, spec, extra in (("in_context_new", new_v1, NEW), ("in_context_known", known_v1, KNOWN)):
            ids = [f["id"] for f in spec["facts"]] + [f["id"] for f in extra]
            assert len(ids) == len(set(ids)) == 30, (name, len(ids))
            doc = {"facts": spec["facts"] + extra, "judge": _Literal(spec["judge"])}
            header = (f"# {name} v2: the v1 facts plus {len(extra)} more, written by "
                      "experiments/next_round/question_drafts/build_in_context_v2.py -- edit there, not here.\n")  # fmt: skip
            (out / f"{name}.yaml").write_text(header + yaml.dump(doc, sort_keys=False, allow_unicode=True, width=120))
        shutil.copy(V1 / "in_context_markers.yaml", out / "in_context_markers.yaml")
        print(f"{out}: in_context_new 30 facts, in_context_known 30 facts, markers copied")


if __name__ == "__main__":
    main()
