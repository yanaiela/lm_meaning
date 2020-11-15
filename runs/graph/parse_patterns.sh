cd /home/nlp/lazary/workspace/thesis/lm_meaning/
export PYTHONPATH=/home/nlp/lazary/workspace/thesis/lm_meaning

patterns_file=$1
subject=$2
object=$3
out_file=$4


/home/nlp/lazary/anaconda3/envs/memorization/bin/python lm_meaning/spike_patterns/to_json.py \
        --patterns_file $patterns_file \
        --subject $subject \
        --object $object \
        --out_file $out_file
