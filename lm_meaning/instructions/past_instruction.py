import logging

from tqdm import tqdm

from lm_meaning.common.file_utils import get_file_from_s3
from lm_meaning.lm_instruction import LMInstruction
from lm_meaning.common.lm_utils import get_pretrained_model
from lm_meaning.common.challenge_utils import filter_vals

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class PastInstruction(LMInstruction):
    def __init__(self, args):
        self.instruction_name = 'PastInstruction'
        logger.info("loading...")
        super().__init__(args)

    def process_conll(self, lines):
        tense_inflection = {}

        for line in tqdm(lines):
            if line.strip() == '':
                continue
            if line.startswith('#'):
                continue
            parts = line.split()
            lemma = parts[2]
            if parts[3] == 'VERB':
                morphs = parts[5]
                morphs_parts = morphs.split('|')
                for morph_val in morphs_parts:
                    if morph_val.startswith('Tense=Past'):
                        tense_inflection[lemma] = parts[1]
        return tense_inflection

    def build_challenge(self, args):

        logger.info("loading conll file")
        lines = get_file_from_s3("s3://lminstructions/data/ud/en_ewt-ud-train.conllu")
        past_inflection = self.process_conll(lines)

        logger.info("initial examples: {}".format(len(past_inflection)))

        tokenizer, _ = get_pretrained_model('roberta-large')
        filter_past_inflection = filter_vals(past_inflection, tokenizer)
        logger.info("left after filtering: {}".format(len(filter_past_inflection)))

        logger.info("building examples")

        prompt = 'Conjugate the word "{}" to past tense: [MASK].'

        for example_ind, (lemma, plural) in tqdm(enumerate(filter_past_inflection.items())):

            example = {'prompt': prompt,
                       'input': lemma,
                       'answer': plural,
                       'function': 'past'}

            self.append_olmpics_format_example(example, do_print=self._config['debug'])

        self.save_dataset()




