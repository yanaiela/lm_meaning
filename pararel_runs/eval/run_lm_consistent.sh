cd /home/nlp/lazary/workspace/thesis/lm_meaning/
export PYTHONPATH=/home/nlp/lazary/workspace/thesis/lm_meaning

patterns_file=$1
data_file=$2
lm=$3
graph=$4


/home/nlp/lazary/anaconda3/envs/memorization/bin/python lm_meaning/evaluation/encode_consistency_probe.py \
        --patterns_file $patterns_file \
        --data_file $data_file \
        --lm $lm \
        --graph $graph \
        --gpu 0 \
        --wandb \
        --use_targets

