# Multi-agent Dynamic Algorithm Configuration

Official implementation of *Multi-agent Dynamic Algorithm Configuration*.

## Installation

```bash
conda create -n madac python=3.7.13
conda activate madac
pip install -r requirements.txt
pip install -e . # Local installation of madacbench Package
```

## Train MA-DAC

Train MA-DAC with MaMo in different tasks.

```bash
# Train MA-DAC (3)
python algos/madac/main.py --config=vdn_ns --env-config=moea with env_args.key=M_2_46_3
# Train MA-DAC (5)
python algos/madac/main.py --config=vdn_ns --env-config=moea with env_args.key=M_2_46_5
# Train MA-DAC (7)
python algos/madac/main.py --config=vdn_ns --env-config=moea with env_args.key=M_2_46_7
# Train MA-DAC (M)
python algos/madac/main.py --config=vdn_ns --env-config=moea with env_args.key=M_2_46_357
```

Train MA-DAC with Sigmoid.

```bash
python algos/madac/main.py --config=vdn_ns_sigmoid --env-config=sigmoid_state
```

You can modify the relevant configuration file `algos/madac/config/envs/moea.yaml` and `algos/madac/config/algs/vdn_ns.yaml`

## Test MA-DAC

The trained model is saved in directory `results/madac/models/`, you need to specify the model directory in the configuration file via parameter `checkpoint_path`. More details of the configuration file can be found in [EPyMARL](https://github.com/uoe-agents/epymarl).

Test MA-DAC in a specific problem. (The problem set is `DTLZ2_3 DTLZ4_3 WFG4_3 WFG5_3 WFG6_3 WFG7_3 WFG8_3 WFG9_3 DTLZ2_5 DTLZ4_5 WFG4_5 WFG5_5 WFG6_5 WFG7_5 WFG8_5 WFG9_5 DTLZ2_7 DTLZ4_7 WFG4_7 WFG5_7 WFG6_7 WFG7_7 WFG8_7 WFG9_7`) For example,

```bash
python algos/madac/main.py --config=vdn_ns_test --env-config=moea_test with env_args.key=DTLZ2_3
```

## Other Baselines

### MOEA/D

```bash
python algos/moead/moead_baseline.py
```

### DQN

Train DQN in different tasks. (The task set is `M_2_46_3, M_2_46_5, M_2_46_7`). For example, 

```
python algos/dac/dqn.py --key M_2_46_3
```

The trained model can be found in the directory `results/dqn/M_2_46_3`

The command to test the corresponding model on all problems is

```
python algos/dac/test_dqn.py --key M_2_46_3
```

## License

All the source code that has been taken from the `EPyMARL` repository was licensed (and remains so) under the Apache License v2.0 (included in `LICENSE` file). Any new code is also licensed under the Apache License v2.0.

## Citation

```
@inproceedings{madac,
    author = {Ke Xue, Jiacheng Xu, Lei Yuan, Miqing Li, Chao Qian, Zongzhang Zhang, Yang Yu},
    title = {Multi-agent Dynamic Algorithm Configuration},
    booktitle = {Advances in Neural Information Processing Systems 35 (NeurIPS'22)},
    year = {2022},
    address={New Orleans, LA}
}
```
