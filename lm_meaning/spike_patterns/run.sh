python3 to_json.py ../../data/pattern_data/P449.tsv Lost ABC
python3 to_json.py ../../data/pattern_data/P19.tsv John England
python3 to_json.py ../../data/pattern_data/P20.tsv John England
python3 to_json.py ../../data/pattern_data/P106.tsv John lawyer
python3 to_json.py ../../data/pattern_data/P190.tsv Doha Ankara
python3 to_json.py ../../data/pattern_data/P102.tsv Trump Republican


python generate_all_entailment.py ../../data/pattern_data/P449.tsv ../../data/pattern_data/P449_entailment_lemmas.tsv
python3 create_graph.py ../../data/pattern_data/P449.jsonl ../../data/pattern_data/P449_entailment_lemmas.extended.tsv
