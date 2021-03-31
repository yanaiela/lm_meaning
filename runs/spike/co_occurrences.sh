cd /home/nlp/lazary/workspace/thesis/lm_meaning/
export PYTHONPATH=/home/nlp/lazary/workspace/thesis/lm_meaning

data_file=$1
spike_results=$2


/home/nlp/lazary/anaconda3/envs/memorization/bin/python lm_meaning/spike/cooccurrence_queries.py \
        --data_file $data_file \
        --spike_results $spike_results
