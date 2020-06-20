import string

from tqdm import tqdm


def filter_vals(in_dic, tokenizer):
    filter_inflections = {}

    for k, v in tqdm(in_dic.items()):
        tok_v = tokenizer.tokenize(v)

        if len(tok_v) != 1:
            continue

        if not all([x in (string.ascii_lowercase + string.ascii_uppercase) for x in k]):
            continue

        filter_inflections[k] = v

    return filter_inflections
