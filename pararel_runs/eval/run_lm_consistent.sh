cd /home/nlp/lazary/workspace/thesis/lm_meaning/
export PYTHONPATH=/home/nlp/lazary/workspace/thesis/lm_meaning

data_file=$1
lm=$2
graph=$3


/home/nlp/lazary/anaconda3/envs/memorization/bin/python pararel/consistency/encode_consistency_probe.py \
        --data_file $data_file \
        --lm $lm \
        --graph $graph \
        --gpu 0 \
        --wandb \
        --use_targets

