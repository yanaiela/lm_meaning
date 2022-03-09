cd /home/nlp/lazary/workspace/thesis/lm_meaning/
export PYTHONPATH=/home/nlp/lazary/workspace/thesis/lm_meaning

model=$1


/home/nlp/lazary/anaconda3/envs/memorization/bin/python memorization/explanation/cooccurrence_causal_effect.py \
        -p all \
        -m $model \
