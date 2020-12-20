cd /home/nlp/lazary/workspace/thesis/lm_meaning/
export PYTHONPATH=/home/nlp/lazary/workspace/thesis/lm_meaning

patterns_file=$1
data_file=$2
lm=$3
graph=$4


/home/nlp/lazary/anaconda3/envs/memorization/bin/python lm_meaning/train_consistancy.py \
        --dataset_name data/enailment_train/consistancy_relation_3_200/20201221-012304/train.txt \
        --output_dir models/consistency/bert_base_cased/3_200/ \
        --lm bert-base-cased \
        --epochs 4 \
        --batch_size 500

