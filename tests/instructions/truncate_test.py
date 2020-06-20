# pylint: disable=no-self-use,invalid-name
import argparse
import os

from lm_meaning.instructions.truncate_instruction import TruncateInstruction


class TestTruncateInstruction:
    def test_build_challenge(self):
        parse = argparse.ArgumentParser("")
        parse.add_argument("-i", "--challenge_name", type=str, help="The name of the challenge class and config to use")
        parse.add_argument("-o", "--operation", type=str, help="The task stage to run")
        parse.add_argument("-out", "--output_file", type=str, help="")
        parse.add_argument("--config_path", type=str, help="Challenges config file", default="config.json")
        # In the test no output file will be produced, change -out to create an output
        args = parse.parse_args(["-i", "Truncate", "-o", "build_challenge", "-out",
                                 "data/challenge_samples/truncate/truncate.jsonl"])

        if not os.path.exists("data/challenge_samples/truncate"):
            os.mkdir('data/challenge_samples/truncate')

        challenge = TruncateInstruction(args)

        # reducing data size to a sample:
        challenge._config['dev_size'] = 10
        challenge._config['max_number_of_examples'] = 20

        challenge.build_challenge(args)
