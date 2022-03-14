cd /home/nlp/lazary/workspace/thesis/lm_meaning/
export PYTHONPATH=/home/nlp/lazary/workspace/thesis/lm_meaning

model=$1
patterns=$2
random_weights=$5


/home/nlp/lazary/anaconda3/envs/memorization/bin/python memorization/explanation/memorization_causal_effect.py \
        -p $patterns \
        -m $model \
        --random_weights $random_weights
