cd /home/nlp/lazary/workspace/thesis/lm_meaning/
export PYTHONPATH=/home/nlp/lazary/workspace/thesis/lm_meaning

model=$1
patterns=$2
random_weights=$3
perfect_model=$4

/home/nlp/lazary/anaconda3/envs/memorization/bin/python memorization/explanation/cooccurrence_causal_effect.py \
        -p $patterns \
        -m $model \
        --random_weights $random_weights \
        --perfect_model $perfect_model
