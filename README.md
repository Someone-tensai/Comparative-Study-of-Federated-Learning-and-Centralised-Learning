# Comparative-Study-of-Federated-Learning-and-Centralised-Learning

## A. Requirements

1. Install requirements
```bash
   pip install -r requirements.txt
```

## B. Preprocessing

1. Run `preprocess.py` from root
```bash
   python -m preprocessing.pipeline.preprocess
```
2. Run `split.py` from root — `--clients` sets the number of federated clients to partition the data into (must match the SuperNode count used in setup below)
```bash
   python -m preprocessing.pipeline.split --clients 3 --mode both
```

### C.1 Weights & Biases Setup (for experiment tracking)

1. Sign up for a free account at [wandb.ai](https://wandb.ai) if you don't already have one.

2. Log in from your terminal (this will prompt for an API key, found on your [wandb settings page](https://wandb.ai/authorize)):
```bash
wandb login
```

3. Paste your API key when prompted. You only need to do this once per machine — Colab included, though the login will need to be redone each time a Colab runtime resets.

4. Make sure your runs point to the shared entity so they show up in the project:
```python
wandb.init(entity="paudelsulav5-fedlearnproject", project="same_param_fed_training")
```

## C.2 Model Setup and Train

Set the number of simulated clients (SuperNodes) — must match `--clients` used during preprocessing:
```bash
flwr federation simulation-config \
  --num-supernodes 3 \
  --client-resources-num-cpus 2 \
  --client-resources-num-gpus 0.33
```

### D. Training Models (Same Hyperparameters)

```bash
flwr run . --stream --run-config "model-name='resnet18' freeze-backbone=true"
```

**Options** (add any of these alongside the other options as needed):

| Flag / Key           | Description                                                                    |
|-----------------------|--------------------------------------------------------------------------------|
| `--stream`            | Streams logs to your terminal in real time                                     |
| `model-name`          | Model to train — currently supports `resnet18` (default), `resnet50`, and `vgg16` |
| `freeze-backbone`     | Whether to freeze the model backbone during training (`true`/`false`)          |
| `num-server-rounds`   | Number of federated training rounds (default: `20`)                            |
| `learning-rate`       | Optimizer learning rate (default: `0.001`)                                     |
| `batch-size`          | Batch size per client (default: `32`)                                          |
| `local-epochs`        | Number of local epochs per client, per round (default: `2`)                    |
| `weight-decay`        | Weight decay for the optimizer (default: `0.0001`)                             |
| `mode`                | Data partitioning mode — `iid` or `noniid` (default: `noniid`)                 |
| `fed-strategy`        | Federated aggregation strategy — e.g. `fedavg`, `fedprox` (default: `fedavg`)  |
| `proximal-mu`         | Proximal term weight, used when `fed-strategy='fedprox'` (default: `0.0`)      |

**Example — combining multiple options in one run:**
```bash
flwr run . --stream --run-config "model-name='resnet50' freeze-backbone=false local-epochs=3 learning-rate=0.0005 fed-strategy='fedprox' proximal-mu=0.01"
```