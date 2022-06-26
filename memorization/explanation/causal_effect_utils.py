import json
import wandb


def log_wandb(args, hypothesis):
    pattern = args.pattern
    model = args.model

    config = dict(
        pattern=pattern,
        lm=model,
        random_weights=args.random_weights,
        perfect_model=args.perfect_model
    )

    wandb.init(
        entity='consistency',
        name=f'causal_{hypothesis}_{model}_{pattern}',
        project="memorization",
        tags=[hypothesis, pattern, model],
        config=config,
    )


def read_data(pattern: str, model: str, random_weights: bool):
    with open(f'memorization_data/output/spike_results/cooccurrences/{pattern}.json', 'r') as f:
        co_occurrence_data = json.load(f)

    with open(f'memorization_data/output/spike_results/preferences/{pattern}.json', 'r') as f:
        obj_preference_data = json.load(f)

    with open(f'memorization_data/trex_lms_vocab/{pattern}.jsonl', 'r') as f:
        trex = f.readlines()
        trex = [json.loads(x.strip()) for x in trex]

    if random_weights:
        pred_dir_pat = 'randw_bert_lama'
    else:
        pred_dir_pat = 'bert_lama'

    if 'models' in model:
        model = model.replace('/', '_')
    with open(f'memorization_data/output/predictions_lm/{pred_dir_pat}/{pattern}_{model}.json', 'r') as f:
        paraphrase_preds = json.load(f)

    if random_weights:
        pred_dir_anti_pat = 'randw_bert_lama_unpatterns'
    else:
        pred_dir_anti_pat = 'bert_lama_unpatterns'

    with open(f'memorization_data/output/predictions_lm/{pred_dir_anti_pat}/{pattern}_{model}.json', 'r') as f:
        unparaphrase_preds = json.load(f)

    with open(f'memorization_data/output/spike_results/paraphrases/{pattern}.json', 'r') as f:
        memorization = json.load(f)

    with open(f'data/pattern_data/parsed/{pattern}.jsonl') as f:
        patterns = [json.loads(x.strip()) for x in f.readlines()]

    return co_occurrence_data, obj_preference_data, trex, paraphrase_preds, unparaphrase_preds, memorization, patterns


def count_bins(row):
    count = row['count']
    if count <= 1:
        return 'xs'
    elif count <= 10:
        return 's'
    elif count <= 100:
        return 'm'
    elif count <= 1000:
        return 'l'
    else:
        return 'xl'
