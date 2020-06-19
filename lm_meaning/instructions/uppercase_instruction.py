import logging

from tqdm import tqdm

from lm_meaning.lm_instruction import LMInstruction
from lm_meaning import utils

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

'''python lm_meaning/run.py -i Uppercase -o build_challenge -out s3://lminstructions/instructions/uppercase.jsonl.gz'''
class UppercaseInstruction(LMInstruction):
    def __init__(self, args):
        self.instruction_name = 'UppercaseInstruction'
        logger.info("loading...")
        super().__init__(args)

    def build_challenge(self, args):

        logger.info("building examples")
        src_file = "./data/google1T/freq_word_list.txt"
        words = open(src_file).read().split("\n")
        words = utils.filter_vocab(words)


        for example_ind, num in tqdm(enumerate(range(101))):

            prompt = 'Capitalize the word " {} ": [MASK].'
            input_word = words[num]
            ans = input_word[0].upper() + input_word[1:]

            example = {'prompt': prompt,
                       'input': input_word,
                       'answer': ans,
                       'function': 'uppercase'}

            self.append_olmpics_format_example(example, do_print=self._config['debug'])

        self.save_dataset()




