cd /home/nlp/lazary/workspace/thesis/lm_meaning/
export PYTHONPATH=/home/nlp/lazary/workspace/thesis/lm_meaning

trex=$1
lm_file=$2
graph=$3


/home/nlp/lazary/anaconda3/envs/memorization/bin/python lm_meaning/evaluation/entailment_probe.py \
        --trex $trex \
        --lm_file $lm_file \
        --graph $graph