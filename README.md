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

**Example — combining multiple options in one run:**
```bash
flwr run . --stream --run-config "model-name='resnet50' freeze-backbone=false local-epochs=3 learning-rate=0.0005"
```

## E. Running on Google Colab (GPU)

1. Open a new notebook at [colab.research.google.com](https://colab.research.google.com), then set **Runtime → Change runtime type → GPU**.

2. Clone the repo and move into it:
```bash
!git clone https://github.com/yourusername/Comparative-Study-of-Federated-Learning-and-Centralised-Learning.git
%cd Comparative-Study-of-Federated-Learning-and-Centralised-Learning
```

3. Install dependencies:
```bash
!pip install -r requirements.txt
```

4. Run preprocessing and split (same as Sections B.1–B.2 above):
```bash
!python -m preprocessing.pipeline.preprocess
!python -m preprocessing.pipeline.split --clients 3 --mode both
```

5. Set the simulation config, including GPU allocation for the Colab GPU (fractional value lets multiple clients share the single GPU concurrently — e.g. `0.33` allows all 3 clients to run in parallel on one T4):
```bash
!flwr federation simulation-config \
  --num-supernodes 3 \
  --client-resources-num-cpus 2 \
  --client-resources-num-gpus 0.33
```

6. Verify the GPU is visible before training:
```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

7. Train, same as Section D:
```bash
!flwr run . --stream --run-config "model-name='resnet18' freeze-backbone=true"
```

**Colab-specific notes:**
| Note | Details |
|------|---------|
| Session timeout | Colab disconnects after ~90 min idle or ~12 hr max session — checkpoint periodically for long runs |
| Save checkpoints to Drive | `from google.colab import drive; drive.mount('/content/drive')`, then point checkpoint paths to `/content/drive/MyDrive/...` |
| Check current config | `!flwr config list` shows the active SuperLink connection and config file path |
