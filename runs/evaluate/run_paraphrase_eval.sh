cd /home/nlp/lazary/workspace/thesis/lm_meaning/
export PYTHONPATH=/home/nlp/lazary/workspace/thesis/lm_meaning

lm_file=$1
spike_file=$2
lm_patterns=$3
spike_patterns=$4


/home/nlp/lazary/anaconda3/envs/memorization/bin/python lm_meaning/evaluation/spike_lm_eval.py \
        --lm_file $lm_file \
        --spike_file $spike_file \
        --lm_patterns $lm_patterns \
        --spike_patterns $spike_patterns