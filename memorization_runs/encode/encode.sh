cd /home/nlp/lazary/workspace/thesis/lm_meaning/
export PYTHONPATH=/home/nlp/lazary/workspace/thesis/lm_meaning

data_file=$1
lm=$2
patterns=$3
out=$4


/home/nlp/lazary/anaconda3/envs/memorization/bin/python memorization/encode/encode_consistency_probe.py \
        --data_file $data_file \
        --lm $lm \
        --patterns $patterns \
        --out $out \
        --gpu 0 \
        --wandb \
        --use_targets \
        --random_weights

