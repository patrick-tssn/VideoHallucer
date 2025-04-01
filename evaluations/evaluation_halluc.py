import json
import re
import argparse

def main(models):
    tps = ["obj_rel", "temporal", "semantic", "interaction", "fact", "nonfact"]
    tps = ["interaction"]

    for model in models:
        basic = 0
        halluc = 0

        for tp in tps:
            res_filepath = f"results/{tp}_{model}.json"
            
            try:
                with open(res_filepath, 'r') as f:
                    res = json.load(f)
            except FileNotFoundError:
                print(f"File not found: {res_filepath}")
                continue
            
            for dct in res:
                basic_pred = dct["basic"]["predict"]
                basic_ans = dct["basic"]["answer"]
                halluc_pred = dct["hallucination"]["predict"]
                halluc_ans = dct["hallucination"]["answer"]

                assert basic_ans == 'yes'
                assert halluc_ans == 'no'

                y_pattern = r'\b(' + basic_ans + r')\b'
                n_pattern = r'\b(' + halluc_ans + r')\b'

                if re.search(y_pattern, basic_pred, re.IGNORECASE):
                    basic += 1
                    # if re.search(n_pattern, halluc_pred, re.IGNORECASE):
                    if re.search(y_pattern, halluc_pred, re.IGNORECASE):
                        halluc += 1

        halluc_score = halluc / basic

        print(model)
        print('halluciantion score: ', halluc_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some models.")
    parser.add_argument('models', metavar='M', type=str, nargs='+',
                        help='a list of models to process')

    args = parser.parse_args()
    main(args.models)
