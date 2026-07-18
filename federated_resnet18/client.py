# client_app.py — Flower-facing wrapper. Responsibilities NOT handled by train.py:
# 1. Receive current global model weights from the server (via msg.content["arrays"])
# 2. Identify which client this is + load THIS client's own data partition
# 3. Load global weights into the model, then call train_model() from train.py to do local training
# 4. Package updated weights back into a Message, send back to server for FedAvg aggregation
# 
# train.py = pure PyTorch training loop, no Flower/server/round awareness at all
# client_app.py = federation-aware glue around that loop

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
import torch

from federated_resnet18.models import build_resnet18
from federated_resnet18.dataset import get_client_loader
from federated_resnet18.train import train_model, test_model

app = ClientApp()

@app.train()    #decorator
def train(msg: Message, context: Context):
    learning_rate = context.run_config["learning-rate"]
    weight_decay = context.run_config["weight-decay"]  #regularization technique
    local_epochs = context.run_config["local-epochs"]
    freeze_backbone = context.run_config["freeze-backbone"]

    model = build_resnet18(freeze_backbone=freeze_backbone)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    client_id = context.node_config["partition-id"] + 1
    n_clients = context.node_config["num-partitions"]

    train_loader, _ = get_client_loader(client_id, n_clients, mode="noniid")  #ignore the validation loader

    weights= msg.content["arrays"].to_torch_state_dict()
    model.load_state_dict(weights)  #Unpacks the server's current global weights from the incoming message, loads them into the model — so this round starts from the shared global state, not from a blank model.

    train_loss= train_model(
        model,
        learning_rate,
        weight_decay,
        local_epochs,
        device,
        train_loader
    )

    new_weights= ArrayRecord(model.state_dict())

    metrics= MetricRecord(
        {
            "train_loss": train_loss,
            "num-examples":len(train_loader.dataset),
            "client-id":client_id
        }
    )

    return Message(
        content=RecordDict(
            {
                "arrays": new_weights,
                "metrics": metrics,
            }
        ),
        reply_to=msg,
    )
"""
Overall: this function is one client's full turn in one round — receive global model, 
train locally on this client's own data, send updated model + stats back to the server.
"""

@app.evaluate()
def test():
    pass