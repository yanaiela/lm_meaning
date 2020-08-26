import argparse
import logging

from lm_meaning.rules.rules_factory import RuleFactory
from lm_meaning.rules.utils import read_file, eval_performance

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("-r", "--rule_name", type=str, help="The name of the challenge class and config to use")
    parse.add_argument("-o", "--operation", type=str, help="The task stage to run")
    parse.add_argument("-out", "--output_file", type=str, help="")
    parse.add_argument("--config_path", type=str, help="Challenges config file", default="config.json")
    parse.add_argument("-d", "--data_path", type=str, help="", default="data/TREx_train")
    args = parse.parse_args()

    data = read_file(args.data_path + '/{}.jsonl'.format(args.rule_name))

    matcher = RuleFactory().get_rule(args.rule_name)
    answers = matcher.process_relation(data, subset=20)

    print('done')

    # print('top@1 accuracy:', eval_performance(answers))

    # matcher.persist_answers(answers, args.output_file)


if __name__ == '__main__':
    main()
