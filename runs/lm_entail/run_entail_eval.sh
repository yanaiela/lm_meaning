cd /home/nlp/lazary/workspace/thesis/lm_meaning/
export PYTHONPATH=/home/nlp/lazary/workspace/thesis/lm_meaning

trex=$1
lm_file=$2
lm_patterns=$3
spike_patterns=$4
graph=$5


/home/nlp/lazary/anaconda3/envs/memorization/bin/python lm_meaning/evaluation/entailment_probe.py \
        --trex $trex \
        --lm_file $lm_file \
        --lm_patterns $lm_patterns \
        --spike_patterns $spike_patterns \
        --graph $graph