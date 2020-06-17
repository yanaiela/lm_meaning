# pylint: disable=no-self-use,invalid-name
import pytest
import argparse
import os
from lm_meaning.instructions.example_instruction import ExampleInstruction


class TestExampleInstruction:
    def test_build_challenge(self):
        parse = argparse.ArgumentParser("")
        parse.add_argument("-i", "--challenge_name", type=str, help="The name of the challenge class and config to use")
        parse.add_argument("-o", "--operation", type=str, help="The task stage to run")
        parse.add_argument("-out", "--output_file", type=str, help="")
        parse.add_argument("--config_path", type=str, help="Challenges config file", default="config.json")
        # In the test no output file will be produced, change -out to create an output
        args = parse.parse_args(["-i", "Example", "-o", "build_challenge", "-out",
                                 "data/challenge_samples/example/sample.jsonl"])

        if not os.path.exists("data/challenge_samples/example"):
            os.mkdir('data/challenge_samples/example')

        challenge = ExampleInstruction(args)

        # reducing data size to a sample:
        challenge._config['dev_size'] = 1
        challenge._config['max_number_of_examples'] = 2

        challenge.build_challenge(args)
