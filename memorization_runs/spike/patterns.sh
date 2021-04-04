cd /home/nlp/lazary/workspace/thesis/lm_meaning/
export PYTHONPATH=/home/nlp/lazary/workspace/thesis/lm_meaning

data_file=$1
spike_patterns=$2
spike_results=$3


/home/nlp/lazary/anaconda3/envs/memorization/bin/python lm_meaning/spike/run_syntactic_queries.py \
        --data_file $data_file \
        --spike_patterns $spike_patterns \
        --spike_results $spike_results
