# ParaRel :metal:

This repository contains the code and data for the paper:

[`Measuring and Improving Consistency in Pretrained Language Models`](https://arxiv.org/abs/2102.01017)

as well as the resource: `ParaRel` :metal:


Since this work required running a lot of experiments, it is structured by scripts that automatically 
runs many sub-experiments, on parallel servers, and tracking using an experiment tracking website: [wandb](https://wandb.ai/site),
which are then aggregated using a jupyter notebook.

It is also possible to run individual experiments, for which one can look for in the corresponding script.

For any question, query regarding the code, or paper, please reach out at `yanaiela@gmail.com`


## Create environment
```sh
conda create -n pararel python=3.7 anaconda
conda activate pararel
```
add project to path:
```sh
export PYTHONPATH=${PYTHONPATH}:/path-to-project
```


## Run Scripts

Filter data from trex, to include only triplets that appear in the inspected LMs in this work:
`BERT-base-cased`, `roberta-base`, `albert-base-v2` (as well as the larger versions, that contain the same vocabulary)
```sh
python runs/pararel/filter.py
```

Create json files, along with a spike query from the raw patterns
```sh
python runs/pararel/parse_patterns.py
```

Create entailment files, their extended version of the written rules
```sh
python runs/pararel/entailed_lemmas.py
```

Create entailment graphs out of the entailment and pattern files
```sh
python runs/pararel/create_graph.py
```

Running the LMs on the data:
```sh
python runs/eval/run_paraphrase_lm.py
```


Encode the texts:
```sh
python runs/pararel/encode_text.py
```

## Citation:
If you find this work relevant to yours, please cite us:
```
@article{Elazar2021MeasuringAI,
  title={Measuring and Improving Consistency in Pretrained Language Models},
  author={Yanai Elazar and Nora Kassner and Shauli Ravfogel and Abhilasha Ravichander and E. Hovy and Hinrich Schutze and Y. Goldberg},
  journal={ArXiv},
  year={2021},
  volume={abs/2102.01017}
}
```