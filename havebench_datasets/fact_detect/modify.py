import json

fd = json.load(open("fact_detect_yn.json"))

for dct in fd:
    q = dct["basic"]["question"]
    nq = dct["hallucination"]["question"]
    dct["hallucination"]["question"] = "Does the following course summary contain all the necessary factual knowledge? " + nq.split("Does the following course summary contain any non-factual knowledge? ")[-1]
    dct["hallucination"]["answer"] = "no"

json.dump(fd, open("fact_detect.json", "w"), indent=4)