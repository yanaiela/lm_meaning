cd /home/nlp/lazary/workspace/thesis/lm_meaning/
export PYTHONPATH=/home/nlp/lazary/workspace/thesis/lm_meaning

patterns_file=$1
model_names=$2
out_file=$3


/home/nlp/lazary/anaconda3/envs/memorization/bin/python lm_meaning/lm_entail/filter_data.py \
        --in_data $patterns_file \
        --model_names $model_names \
        --out_file $out_file
