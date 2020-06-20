from transformers import *

import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_pretrained_model(model_name):
    """
    Load tokenizer and model for MaskedLM kind.
    :param model_name: the name of te model to use
    :return: tokenizer, model
    """
    logger.info("using pretrained model: {}".format(model_name))

    try:
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except:
        raise ValueError("pretrained model: {} does not exists".format(model_name))
    model.eval()
    return tokenizer, model
