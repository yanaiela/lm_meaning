import argparse
import logging

import jsonlines

from scipy.stats import pearsonr, spearmanr


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def read_json_file(in_f):
    with jsonlines.open(in_f, 'r') as f:
        lines = list(f)
    return lines


def compare_on_rule(rules, lms):
    # TODO - change back to this line when using the actual lms data
    # lm_rule = lms[0]['rule']
    lm_rule = 'air on'

    rule_success = []
    lm_success = []

    rule_given_lm = 0
    given_lm = 0
    for rule_ans, lm_ans in zip(rules, lms):
        correct_ans = rule_ans['obj'].lower()

        if rule_ans['answer'].lower() == correct_ans and rule_ans.get('rule', '') == lm_rule:
            rule_success.append(1)
        else:
            rule_success.append(0)

        if lm_ans['answer'].lower() == correct_ans:
            lm_success.append(1)
            print(rule_ans['answer'], lm_ans['answer'], correct_ans, rule_ans.get('rule', ''), rule_ans['sub'])
            if rule_ans['answer'].lower() == correct_ans and rule_ans.get('rule', '') == lm_rule:
                rule_given_lm += 1
            given_lm += 1
        else:
            lm_success.append(0)
    print(rule_success)
    print(lm_success)

    print(pearsonr(rule_success, lm_success))

    print(rule_given_lm / given_lm)


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("-rule", "--rules_file", type=str, help="The name of the challenge class and config to use")
    parse.add_argument("-lm", "--lm_file", type=str, help="")

    # parse.add_argument("-v", "--variant", type=str, help="", default="")
    # parse.add_argument("-s", "--split", type=str, help="The task stage to run", default="dev")
    # parse.add_argument("--cuda_device", type=int, help="", default=-1)
    # parse.add_argument("-p", "--n_processes", type=int, help="For challenges with multi process", default=1)
    # parse.add_argument("--config_path", type=str, help="Challenges config file", default="config.json")
    # parse.add_argument("--wandb", type=bool, help="Wheather to use wandb or not", default=False)
    args = parse.parse_args()

    rules_ans = read_json_file(args.rules_file)
    lm_ans = read_json_file(args.lm_file)

    # for line in rules_ans[:10]:
    #     print(line)

    compare_on_rule(rules_ans, lm_ans)


if __name__ == '__main__':
    main()
