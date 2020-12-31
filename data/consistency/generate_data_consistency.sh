for relations in 2 5 10 20 30
  do
  for tuples in 100 200 500 1000 2000
    do
	python data/enailment_train/generate_data_consistency.py -nr $relations -nt $tuples -lama "/home/nlp/lazary/workspace/thesis/lm_meaning/data/trex_lms_vocab/"
    done
  done
