import random
import json
import logging
import hashlib
import os
import numpy as np

from lm_meaning.common.file_utils import upload_jsonl_to_s3, save_jsonl_to_local, is_path_creatable

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class LMInstruction:
    def __init__(self, args):

        random.seed(0)
        np.random.seed(0)
        self._np_seed = np.random.RandomState(0)

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config_path), 'r') as f:
            self._config = json.load(f)[self.instruction_name]

        self._output_file = args.output_file
        self.challenge_data = []

    def append_olmpics_format_example(self, example, do_print=False, append_to_list=None):
        """append_olmpics_format_example() is method implemented in oLMpicChallenge class and takes an example dict
        (that must contain "asnwer", "question", "dist1", "dist2") and converts it
        to an oLMpics / ARC / OpenBookQA / CommonsenseQA  AllenAI format for a multi-choice questions

        Args:
            example (dict): an example containing question,answer,dist1,dist2 fields
            do_print (bool): just for debuging
            append_to_list (list): a

        Returns:

        """
        if 'qid' not in example:
            example['qid'] = self.create_qid(example)
        qid = example['qid']

        if do_print:
            print('a:%s p:%s, i:%s, f:%s' % (example['answer'], example['prompt'],
                                             example['input'], example['function']))

        example = {'id': qid, 'prompt': example['prompt'], 'input': example['input'], 'answer': example['answer'],
                   'function': example['function']}
        if append_to_list is not None:
            append_to_list.append(example)
        else:
            self.challenge_data.append(example)

    @staticmethod
    def create_qid(example):
        m = hashlib.md5()
        m.update(example['answer'].encode())
        m.update(example['input'].encode())
        m.update(example['prompt'].encode())
        m.update(example['function'].encode())
        return m.hexdigest()

    def save_dataset(self):
        """save_dataset() automatically saves the challenge
        if the config output_file contains the string _sample.jsonl it will be saved in a more readable format
        otherwise it will split the examples in self.challenge_data into train, dev, test and save them in s3
        if output_file startswith s3:// otherwise locally. (If output_file is empty, it will not save)

        Args:

        Returns:
            bool: Description of return value

        """

        # splitting
        inds = list(range(len(self.challenge_data)))
        random.seed(0)
        random.shuffle(inds)
        dev_inds = inds[:self._config['dev_size']]
        test_inds = inds[self._config['dev_size']:]

        if self._output_file.startswith('s3://'):
            save_func = upload_jsonl_to_s3
        elif is_path_creatable(self._output_file) and len(self._output_file) > 0:
            save_func = save_jsonl_to_local
        else:
            # Do nothing
            return

        logger.info('uploading %d, %d dev and test examples' % (len(dev_inds), len(test_inds)))
        save_func(self._output_file.replace('.jsonl', '_dev.jsonl'), [self.challenge_data[i] for i in dev_inds])
        save_func(self._output_file.replace('.jsonl', '_test.jsonl'), [self.challenge_data[i] for i in test_inds])

        return dev_inds, test_inds

    def build_challenge(self, args):
        pass

    def resplit(self, args):
        logger.error('Not implemented for this challenge')
