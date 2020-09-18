cd /home/nlp/lazary/workspace/thesis/lm_meaning/
export PYTHONPATH=/home/nlp/lazary/workspace/thesis/lm_meaning

spike_patterns=$1
spike_results=$2


/home/nlp/lazary/anaconda3/envs/memorization/bin/python lm_meaning/spike/pattern_queries.py \
        --spike_patterns $spike_patterns \
        --spike_results $spike_results
