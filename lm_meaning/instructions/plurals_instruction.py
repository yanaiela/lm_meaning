import logging

from tqdm import tqdm

from lm_meaning.common.file_utils import get_file_from_s3
from lm_meaning.lm_instruction import LMInstruction
from lm_meaning.common.lm_utils import get_pretrained_model
from lm_meaning.common.challenge_utils import filter_vals

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class PluralInstruction(LMInstruction):
    def __init__(self, args):
        self.instruction_name = 'PluralInstruction'
        logger.info("loading...")
        super().__init__(args)

    def process_conll(self, lines):
        number_inflections = {}

        for line in tqdm(lines):
            if line.strip() == '':
                continue
            if line.startswith('#'):
                continue
            parts = line.split()
            lemma = parts[2]
            if parts[3] == 'NOUN':
                morph = parts[5]
                morph_parts = morph.split('|')
                for morph_val in morph_parts:
                    if morph_val.startswith('Number=Plur'):
                        number_inflections[lemma] = parts[1]
        return number_inflections

    def build_challenge(self, args):

        logger.info("loading conll file")
        lines = get_file_from_s3("s3://lminstructions/data/en_ewt-ud-train.conllu")
        plural_inflection = self.process_conll(lines)

        logger.info("initial examples: {}".format(len(plural_inflection)))

        tokenizer, _ = get_pretrained_model('roberta-large')
        filter_plural_inflection = filter_vals(plural_inflection, tokenizer)
        logger.info("left after filtering: {}".format(len(filter_plural_inflection)))

        logger.info("building examples")

        prompt = 'Conjugate the word "{}" to plural form: [MASK].'

        for example_ind, (lemma, plural) in tqdm(enumerate(filter_plural_inflection.items())):

            example = {'prompt': prompt,
                       'input': lemma,
                       'answer': plural,
                       'function': 'pluralize'}

            self.append_olmpics_format_example(example, do_print=self._config['debug'])

        self.save_dataset()




