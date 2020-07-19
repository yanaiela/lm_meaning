'''
scp -r ./meaning_mem/ aravicha@manhattan.isri.cmu.edu:/home/aravicha/LM_Instructions/

#TODO 1. Find LAMA prompts

'''

import argparse
import utils
import lm_utils
import os
import logging
from transformers import *
#
def build_model_by_name(lm, args, verbose=True):
    """Load a model by name and args.

    Note, args.lm is not used for model selection. args are only passed to the
    model's initializator.
    """
    model_type = lm.split("-")[0]
    MODEL_NAME_TO_CLASS = dict(
        bert= BertForMaskedLM,
        roberta=RobertaForMaskedLM,
    )
    if model_type not in MODEL_NAME_TO_CLASS:
        raise ValueError("Unrecognized Language Model: %s." % lm)
    if verbose:
        print("Loading %s model..." % lm)

    model = MODEL_NAME_TO_CLASS[model_type].from_pretrained(lm)
    tokenizer = AutoTokenizer.from_pretrained(lm)
    model.eval()
    return model, tokenizer



logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)





def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("--relation",  type=str, help="Name of relation")
    parse.add_argument("--lm",type=str,help="comma separated list of language models",default="bert-base-uncased")
    parse.add_argument("--output_file_prefix", type=str, help="")
    parse.add_argument("--prompts", type=str, help="Path to templates for each prompt", default="/data/LAMA_data/TREx")
    parse.add_argument("--data_path", type=str, help="", default="/data/LAMA_data/TREx")
    args = parse.parse_args()

    # Load data
    data_file = os.path.join(args.data_path, args.relation + '.jsonl')
    if not os.path.exists(data_file):
        raise ValueError('Relation "{}" does not exist in data.'.format(args.relation))
    data = utils.read_data(data_file)

    # Load prompts
    prompt_file = os.path.join(args.prompts, args.relation + '.jsonl')
    if not os.path.exists(prompt_file):
        raise ValueError('Relation "{}" does not exist in prompts.'.format(args.relation))
    prompts = utils.load_prompts(prompt_file)


    models_names = args.lm.split(",")



    # Construct queries

    # Make predictions

    # Extract models
    print('Language Models: {}'.format(models_names))
    #
    # models = {}
    # tokenizers = {}

    results_dict = {}
    for lm in models_names:
        model, tokenizer = build_model_by_name(lm, args)

        results_dict[lm]={}

        for prompt_id, prompt in enumerate(prompts):
            results_dict[lm][prompt] = []
            predictions = []

            for sample in data:
                subject_label, object_label = sample["sub_label"], sample["obj_label"]
                query = utils.parse_prompt(prompt, subject_label, "[MASK]")

                prediction = lm_utils.get_predictions(query, model, tokenizer)

                predictions.append((subject_label, object_label, prediction))

            results_dict[lm][prompt] = predictions







                #
    #     vocab_subset = None
    #     if args.common_vocab_filename is not None:
    #         common_vocab = load_vocab(args.common_vocab_filename)
    #         print('Common vocabulary size: {}'.format(len(common_vocab)))
    #
    #
    #     for model_name, model in models.items():
    #         model_vocab = model.vocab

            # # We create his mask to only consider candidattes in our vocabulary
            # vocab_mask_vector = torch.full((1, len(model.vocab)), 1e20)
            #
            # # only keep valid cands, discard punct
            # alpha_indices = [ind for ind in range(len(model_vocab)) if is_valid(model_vocab[ind])]
            #
            # # Fill in valid candidates with 1
            # for indice in alpha_indices:
            #     vocab_mask_vector[0][indice] = 1.0
            #
            # print('\n{}:'.format(model_name))


    # Persist predictions





if __name__ == '__main__':
    main()
