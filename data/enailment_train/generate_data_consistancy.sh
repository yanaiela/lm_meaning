for relations in 2 5 10 20 30
  do
  for tuples in 100 200 500 1000 2000
    do
	python data/enailment_train/generate_data_consistancy.py -nr $relations -nt $tuples -lama "/mounts/data/proj/kassner/LAMA/data/TREx/"
    done
  done
