cd /home/nlp/lazary/workspace/thesis/lm_meaning/
export PYTHONPATH=/home/nlp/lazary/workspace/thesis/lm_meaning

relation_id=$1
relations_file=$2
spike_results=$3


/home/nlp/lazary/anaconda3/envs/memorization/bin/python lm_meaning/spike/preference_queries.py \
        --relation $relation_id \
        --patterns_file $relations_file \
        --spike_results $spike_results
