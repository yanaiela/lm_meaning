cd /home/nlp/lazary/workspace/thesis/lm_meaning/
export PYTHONPATH=/home/nlp/lazary/workspace/thesis/lm_meaning

data_name=$1
output_dir=$2


/home/nlp/lazary/anaconda3/envs/memorization/bin/python lm_meaning/train_consistancy.py \
        --dataset_name $data_name \
        --output_dir $output_dir \
        --lm bert-base-cased \
        --epochs 4 \
        --batch_size 500

