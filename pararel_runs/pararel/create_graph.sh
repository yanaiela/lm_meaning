cd /home/nlp/lazary/workspace/thesis/lm_meaning/
export PYTHONPATH=/home/nlp/lazary/workspace/thesis/lm_meaning

patterns_file=$1
lemmas_file=$2
out_file=$3

/home/nlp/lazary/anaconda3/envs/memorization/bin/python pararel/patterns/create_graph.py \
        --patterns_file $patterns_file \
        --lemmas_file $lemmas_file \
        --out_file $out_file
