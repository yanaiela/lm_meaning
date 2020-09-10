cd /home/nlp/lazary/workspace/thesis/lm_meaning/
export PYTHONPATH=/home/nlp/lazary/workspace/thesis/lm_meaning

lm_file=$1
lm_patterns=$2


/home/nlp/lazary/anaconda3/envs/memorization/bin/python lm_meaning/evaluation/unpatterns_eval.py \
        --lm_file $lm_file \
        --lm_patterns $lm_patterns
