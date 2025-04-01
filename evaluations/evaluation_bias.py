import json
import re
import argparse

def main(models):
    tps = ["obj_rel", "temporal", "semantic", "fact", "nonfact"]
    # tps = ["obj_rel", "temporal", "semantic", "interaction", "fact", "nonfact"]
    # tps = ["interaction"]

    for model in models:
        gt_yes = 0
        gt_no = 0
        n_yes = 0
        n_no = 0
        n = 0
        fp = 0
        tn = 0
        basic = 0
        halluc = 0
        overall = 0
        overall_basic = 0
        overall_halluc = 0
        cnt = 0

        for tp in tps:
            res_filepath = f"cais_results/improve_{tp}_{model}.json"
            
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
                    basic += 1
                    overall_basic += 1
                    if not re.search(n_pattern, halluc_pred, re.IGNORECASE):
                        halluc += 1
                else:
                    if re.search(n_pattern, halluc_pred, re.IGNORECASE):
                        n_no += 1
                    n += 1

                gt_no += 1
                if re.search(n_pattern, halluc_pred, re.IGNORECASE):
                    n_no += 1
                    overall_halluc += 1
                else:
                    if re.search(y_pattern, basic_pred, re.IGNORECASE):
                        n_yes += 1
                        fp += 1
                    n += 1
                    
                if re.search(y_pattern, basic_pred, re.IGNORECASE) and re.search(n_pattern, halluc_pred, re.IGNORECASE):
                    overall += 1
                    
                cnt += 1

        ydp = (n_yes - gt_yes) / (gt_yes * 2) if gt_yes > 0 else 0
        ndp = (n_no - gt_no) / (gt_no * 2) if gt_no > 0 else 0
        fpr = fp / n if n > 0 else 0
        halluc_score = halluc / basic
        overall = overall / cnt
        overall_basic = overall_basic / cnt
        overall_halluc = overall_halluc / cnt

        print("##"*20)
        print(model)
        print('yes difference percentage: ', ydp)
        print('no difference percentage: ', ndp)
        print('false positive ratio: ', fpr)
        print("hallucination score: ", halluc_score)
        print("overall basic score: ", overall_basic)
        print("overall hallucination score: ", overall_halluc)
        print("overall score: ", overall)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some models.")
    parser.add_argument('models', metavar='M', type=str, nargs='+',
                        help='a list of models to process')

    args = parser.parse_args()
    main(args.models)
