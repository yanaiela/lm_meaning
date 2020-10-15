cd /home/nlp/lazary/workspace/thesis/lm_meaning/
export PYTHONPATH=/home/nlp/lazary/workspace/thesis/lm_meaning

data_file=$1
paraphrases_file=$2
spike_results=$3


/home/nlp/lazary/anaconda3/envs/memorization/bin/python lm_meaning/spike/preference_lemmas.py \
        --data_file $data_file \
        --paraphrases_file paraphrases_file \
        --spike_results $spike_results
