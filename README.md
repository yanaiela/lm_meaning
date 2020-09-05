

## Create environment
```sh
conda create -n lm_meaning python=3.7 anaconda
conda activate lm_meaning
```
add project to path:
```sh
export PYTHONPATH=${PYTHONPATH}:/path-to-project
```


## Creating a new Instruction task:

* Create a new script in the rules folder (`lm_meaning/rules/trex``) that parse a wikipedia document
* The name on the class and file should look like the relation name e.g. `P103`
* Implement the `match_rules` method, which is the core 'brain' of how the
 rule will parse the wikipedia page (specifically each line)
* To parse and evaluate a rule, call the `run.py` function, in the following way:
```py
python lm_meaning/rules/run.py -r P1303
```

## Install spike
pip install 'git+https://github.com/allenai/spike#egg=spike-datamodel&subdirectory=datamodel-lib'
pip install 'git+https://github.com/allenai/spike#egg=spike-bl&subdirectory=bl'
