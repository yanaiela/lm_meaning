

## Create environment
```sh
conda create -n lm_meaning python=3.7 anaconda
conda activate lm_meaning
```
add project to path:
```sh
export PYTHONPATH=${PYTHONPATH}:/path-to-project
```

Setup aws credentials

## Creating a new Instruction task:

* Edit the `config.json` file and add the relevant fields
* Create a new script in the instructions folder (`lm_meaning/instructions/``)
* The name on that class should end with `Instruction` and the prefix of that class will be used
to refer to that instruction when using the framework
* Implement the `build_challenge` method, which is the core 'brain' of how the
 instruction will look like
* To create and upload the task, call the `run.py` function, in a similar manner:
```python
python lm_meaning/run.py -i Example -o build_challenge -out s3://lminstructions/instructions/example.jsonl.gz
```

