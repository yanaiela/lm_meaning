cd /home/nlp/lazary/workspace/thesis/lm_meaning/
export PYTHONPATH=/home/nlp/lazary/workspace/thesis/lm_meaning

patterns_file=$1
data_file=$2
lm=$3
pred_file=$4


/home/nlp/lazary/anaconda3/envs/memorization/bin/python lm_meaning/run_pipeline.py \
        --patterns_file $patterns_file \
        --data_file $data_file \
        --lm $lm \
        --pred_file $pred_file \
        --gpu 0 \
        --evaluate

