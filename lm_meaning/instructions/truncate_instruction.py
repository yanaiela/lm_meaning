import logging

from tqdm import tqdm

from lm_meaning.common.file_utils import get_file_from_s3
from lm_meaning.lm_instruction import LMInstruction
from lm_meaning.common.lm_utils import get_pretrained_model
from lm_meaning.common.challenge_utils import filter_vals

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class TruncateInstruction(LMInstruction):
    def __init__(self, args):
        self.instruction_name = 'TruncateInstruction'
        logger.info("loading...")
        super().__init__(args)

    def build_challenge(self, args):

        logger.info("loading conll file")
        lines = get_file_from_s3("s3://lminstructions/data/words/google-10000-english.txt")
        words = [x.strip() for x in lines]

        logger.info("number of words: {}".format(len(words)))
        # assuming all letters are in the vocabulary

        logger.info("building examples")

        prompt = 'Truncate the word "{}" into the first letter: [MASK].'

        for example_ind, word in tqdm(enumerate(words)):
            print(word)
            example = {'prompt': prompt,
                       'input': word,
                       'answer': word[0],
                       'function': 'truncate'}

            self.append_olmpics_format_example(example, do_print=self._config['debug'])

        self.save_dataset()




