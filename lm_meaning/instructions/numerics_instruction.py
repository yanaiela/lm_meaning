import logging
from num2words import num2words
from tqdm import tqdm

from lm_meaning.lm_instruction import LMInstruction

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class NumericInstruction(LMInstruction):
    def __init__(self, args):
        self.instruction_name = 'NumericInstruction'
        logger.info("loading...")
        super().__init__(args)

    def build_challenge(self, args):

        logger.info("building examples")

        for example_ind, num in tqdm(enumerate(range(101))):

            prompt = 'The numeric version of "{}" is: [MASK].'
            input_num = num2words(num)
            ans = str(num)

            example = {'prompt': prompt,
                       'input': input_num,
                       'answer': ans,
                       'function': 'numerate'}

            self.append_olmpics_format_example(example, do_print=self._config['debug'])

        self.save_dataset()




