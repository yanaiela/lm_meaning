import argparse
import wandb

from lm_meaning.common.lm_utils import get_pretrained_model
from lm_meaning.evaluation.lm_predict import eval_query, lm_baseline
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
        split=args.split,
        # dataset=dataset,
        # masking=masking,
        # layer=layer
    )

    wandb.init(
        project="lm_instructions",
        name=args.instruction,
        tags=["lm", "eval", args.instruction],
        config=config,
    )


def prepare_data(args):
    json_list = get_jsonl_from_s3(args.input_file.replace('.jsonl', '_{}.jsonl'.format(args.split)))

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
    parse.add_argument("-s", "--split", type=str, help="The task stage to run", default="dev")
    parse.add_argument("--cuda_device", type=int, help="", default=-1)
    # parse.add_argument("-p", "--n_processes", type=int, help="For challenges with multi process", default=1)
    parse.add_argument("--config_path", type=str, help="Challenges config file", default="config.json")
    parse.add_argument("--wandb", type=bool, help="Wheather to use wandb or not", default=False)
    args = parse.parse_args()

    if args.wandb:
        log_wandb(args)

    tokenizer, model = get_pretrained_model(args.encoder)

    query, json_data = prepare_data(args)

    if args.wandb:
        wandb.run.summary['size'] = len(json_data)

    acc = eval_query(tokenizer, model, json_data, query)
    embedding_baseline_acc = lm_baseline(tokenizer, model, json_data)
    print('accuracy', acc)
    print('baseline accuracy', embedding_baseline_acc)

    if args.wandb:
        wandb.run.summary['accuracy'] = acc
        wandb.run.summary['embedding_baseline_accuracy'] = embedding_baseline_acc


if __name__ == '__main__':
    main()
