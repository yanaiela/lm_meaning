export PYTHONPATH=${PYTHONPATH}:/home/shauli/PycharmProjects/lm_meaning

python3 to_json.py -patterns_file ../../data/pattern_data/P449.tsv -subject Lost -object ABC
python3 to_json.py -patterns_file ../../data/pattern_data/P19.tsv -subject John -object England
python3 to_json.py -patterns_file ../../data/pattern_data/P20.tsv -subject John -object England
python3 to_json.py -patterns_file ../../data/pattern_data/P106.tsv -subject John -object lawyer
python3 to_json.py -patterns_file ../../data/pattern_data/P190.tsv -subject Doha -object Ankara
python3 to_json.py -patterns_file ../../data/pattern_data/P102.tsv -subject Trump -object Republican


#python generate_all_entailment.py ../../data/pattern_data/P449.tsv ../../data/pattern_data/P449_entailment_lemmas.tsv
#python3 create_graph.py ../../data/pattern_data/P449.jsonl ../../data/pattern_data/P449_entailment_lemmas.extended.tsv
