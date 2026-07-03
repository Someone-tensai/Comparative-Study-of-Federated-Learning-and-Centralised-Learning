# Comparative-Study-of-Federated-Learning-and-Centralised-Learning

## Training Models (Same Hyperparameter)

```bash
flwr run . --stream --run-config "model-name='resnet18' freeze-backbone=true"
```

**Options:**

| Flag / Key         | Description                                                              |
|---------------------|---------------------------------------------------------------------------|
| `--stream`          | Streams logs to your terminal in real time                               |
| `model-name`        | Model to train — currently supports `resnet18`, `resnet50`, and `vgg16`  |
| `freeze-backbone`   | Whether to freeze the model backbone during training (`true`/`false`)    |

## Training Models (Different Hyperparameters)

*Not yet implemented.*