# Accelerate

Test Code: Phi2 Inference

## Accelerate Configuration

```bash
accelerate config
```

```bash
[WARNING] Failed to create Level Zero tracer: 2013265921
--------------------------------------------------------------------------------------------------------------------------------------------------
In which compute environment are you running?
This machine 
--------------------------------------------------------------------------------------------------------------------------------------------------
Which type of machine are you using? 
multi-XPU 
How many different machines will you use (use more than 1 for multi-node training)? [1]: 
Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]: 
Do you want to use XPU plugin to speed up training on XPU? [yes/NO]: 
Do you wish to optimize your script with torch dynamo?[yes/NO]: 
Do you want to use DeepSpeed? [yes/NO]: 
Do you want to use FullyShardedDataParallel? [yes/NO]: 
How many XPU(s) should be used for distributed training? [1]:8 
What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]: 
--------------------------------------------------------------------------------------------------------------------------------------------------
Do you wish to use mixed precision?
no

accelerate configuration saved at /home/ua199db1478d72acd9a63c6b83da6d49/.cache/huggingface/accelerate/default_config.yaml
```

## Run Code

```bash
accelerate launch accelerate_phi2.py
```

## Experiment Result

SUCCESSFUL

## Observation

> Output: tokens/sec: 270.0, time 23.68677806854248, total tokens 6400, total prompts 64
