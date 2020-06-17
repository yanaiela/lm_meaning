import argparse

from lm_meaning.instruction_factory import InstructionFactory

import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("-i", "--instruction_name", type=str, help="The name of the challenge class and config to use")
    parse.add_argument("-o", "--operation", type=str, help="The task stage to run")
    parse.add_argument("-out", "--output_file", type=str, help="")
    # parse.add_argument("-v", "--variant", type=str, help="", default="")
    # parse.add_argument("-s", "--split", type=str, help="The task stage to run", default="")
    # parse.add_argument("--cuda_device", type=int, help="", default=-1)
    # parse.add_argument("--copy_from", type=str, help="For create new challenge, the chllenge to copy from", default=-1)
    # parse.add_argument("--challenge_module", type=str, help="For create new challenge, the target challenge path",
    #                    default='')
    # parse.add_argument("-p", "--n_processes", type=int, help="For challenges with multi process", default=1)
    parse.add_argument("--config_path", type=str, help="Challenges config file", default="config.json")
    args = parse.parse_args()

    if args.operation == 'build_challenge':
        challenge = InstructionFactory().get_instruction(args.instruction_name, args)
        challenge.build_challenge(args)
    else:
        logger.error('Operation not supported')


if __name__ == '__main__':
    main()
