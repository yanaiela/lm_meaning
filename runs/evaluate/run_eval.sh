cd /home/nlp/lazary/workspace/thesis/lm_meaning/
export PYTHONPATH=/home/nlp/lazary/workspace/thesis/lm_meaning

task=$1
file=$2
split=$3
encoder=$4

/home/nlp/lazary/anaconda3/envs/lm_meaning/bin/python lm_meaning/evaluation/eval.py \
        -i $task \
        -in $file \
        -s $split \
        --encoder $encoder \
        --wandb True

