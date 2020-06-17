import logging

from tqdm import tqdm

from lm_meaning.lm_instruction import LMInstruction

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class ExampleInstruction(LMInstruction):
    def __init__(self, args):
        self.instruction_name = 'ExampleInstruction'
        logger.info("loading...")
        super().__init__(args)

    def build_challenge(self, args):

        logger.info("building examples")

        for example_ind in tqdm(range(self._config['max_number_of_examples'])):

            prompt = 'Transform the word: "{}" into another word: [MASK].'
            example = {'prompt': prompt,
                       'input': 'hello',
                       'answer': 'hello2',
                       'function': 'transformation'}

            # append_olmpics_format_example() is method implemented in LMInstruction class and takes an example dict
            # (that must contain "prompt", "input", "answer", "function") and converts it
            # to json file object, along with a generated qid
            self.append_olmpics_format_example(example, do_print=self._config['debug'])

            if self._config['max_number_of_examples'] != -1 and \
                    len(self.challenge_data) >= self._config['max_number_of_examples']:
                break

        # save_dataset() is a is method implemented in LMInstruction class that automatically saves the instruction
        # it will split the examples in self.challenge_data into train, dev, test and save them in s3
        # if output_file startswith s3:// otherwise locally. (If output_file is empty, it will not save)
        self.save_dataset()




