import json
import re
import argparse

def main(models):
    tps = ["obj_rel", "temporal", "semantic", "fact", "nonfact"]

    for model in models:
        gt_yes = 0
        gt_no = 0
        n_yes = 0
        n_no = 0
        n = 0
        fp = 0
        tn = 0

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

                gt_yes += 1
                if re.search(y_pattern, basic_pred, re.IGNORECASE):
                    n_yes += 1
                else:
                    if re.search(n_pattern, halluc_pred, re.IGNORECASE):
                        n_no += 1
                    n += 1

                gt_no += 1
                if re.search(n_pattern, halluc_pred, re.IGNORECASE):
                    n_no += 1
                else:
                    if re.search(y_pattern, basic_pred, re.IGNORECASE):
                        n_yes += 1
                        fp += 1
                    n += 1

        ydp = (n_yes - gt_yes) / (gt_yes * 2) if gt_yes > 0 else 0
        ndp = (n_no - gt_no) / (gt_no * 2) if gt_no > 0 else 0
        fpr = fp / n if n > 0 else 0

        print(model)
        print('yes difference percentage: ', ydp)
        print('no difference percentage: ', ndp)
        print('false positive ratio:', fpr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some models.")
    parser.add_argument('models', metavar='M', type=str, nargs='+',
                        help='a list of models to process')

    args = parser.parse_args()
    main(args.models)
