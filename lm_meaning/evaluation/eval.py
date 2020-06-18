import argparse
import wandb

from lm_meaning.common.lm_utils import get_pretrained_model
from lm_meaning.evaluation.lm_predict import eval_query
from lm_meaning.common.file_utils import get_jsonl_from_s3

import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def log_wandb(args):

    config = dict(
        # property=task_type,
        encoder=args.encoder,
        instruction=args.instruction,
        # dataset=dataset,
        # masking=masking,
        # layer=layer
    )

    wandb.init(
        name=args.instruction,
        project="lm_instructions",
        tags=["lm", "eval", args.instruction],
        config=config,
    )


def prepare_data(args):
    json_list = get_jsonl_from_s3(args.input_file)

    assert len(json_list) > 0, 'data list is empty'

    query = json_list[0]['prompt']
    json_data = {}
    for example in json_list:
        json_data[example['input']] = example['answer']
    return query, json_data


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("-i", "--instruction", type=str, help="The name of the challenge class and config to use")
    parse.add_argument("-in", "--input_file", type=str, help="")
    parse.add_argument("-encoder", "--encoder", type=str, help="encoder model")
    # parse.add_argument("-v", "--variant", type=str, help="", default="")
    # parse.add_argument("-s", "--split", type=str, help="The task stage to run", default="")
    parse.add_argument("--cuda_device", type=int, help="", default=-1)
    # parse.add_argument("--copy_from", type=str, help="For create new challenge, the chllenge to copy from", default=-1)
    # parse.add_argument("--challenge_module", type=str, help="For create new challenge, the target challenge path",
    #                    default='')
    # parse.add_argument("-p", "--n_processes", type=int, help="For challenges with multi process", default=1)
    parse.add_argument("--config_path", type=str, help="Challenges config file", default="config.json")
    args = parse.parse_args()

    tokenizer, model = get_pretrained_model(args)

    query, json_data = prepare_data(args)

    acc = eval_query(tokenizer, model, json_data, query)
    print('accuracy', acc)


if __name__ == '__main__':
    main()
