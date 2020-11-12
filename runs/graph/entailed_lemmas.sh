cd /home/lazary/workspace/thesis/lm_meaning/
export PYTHONPATH=/home/nlp/lazary/workspace/thesis/lm_meaning

patterns_file=$1
lemmas_file=$2
out_file=$3

/home/lazary/anaconda3/envs/lm_meaning/bin/python lm_meaning/spike_patterns/generate_all_entailment.py \
        --patterns_file $patterns_file \
        --lemmas_file $lemmas_file \
        --output_file $out_file
