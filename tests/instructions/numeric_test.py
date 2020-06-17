# pylint: disable=no-self-use,invalid-name
import pytest
import argparse
import os
from num2words import num2words
from lm_meaning.instructions.numerics_instruction import NumericInstruction


class TestExampleInstruction:
    def test_build_challenge(self):
        parse = argparse.ArgumentParser("")
        parse.add_argument("-i", "--challenge_name", type=str, help="The name of the challenge class and config to use")
        parse.add_argument("-o", "--operation", type=str, help="The task stage to run")
        parse.add_argument("-out", "--output_file", type=str, help="")
        parse.add_argument("--config_path", type=str, help="Challenges config file", default="config.json")
        # In the test no output file will be produced, change -out to create an output
        args = parse.parse_args(["-i", "Numeric", "-o", "build_challenge", "-out",
                                 "data/challenge_samples/numeric/numeric.jsonl"])

        if not os.path.exists("data/challenge_samples/numeric"):
            os.mkdir('data/challenge_samples/numeric')

        challenge = NumericInstruction(args)

        # reducing data size to a sample:
        challenge._config['dev_size'] = 90
        # challenge._config['max_number_of_examples'] = 2

        challenge.build_challenge(args)

        data = challenge.challenge_data

        # verifying that the numeric numbers are indeed the textual representation of these numbers
        for example in data:
            assert example['input'] == num2words(example['answer'])
