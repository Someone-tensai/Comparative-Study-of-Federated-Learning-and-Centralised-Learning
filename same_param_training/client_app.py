from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
import torch
from same_param_training.models import our_model
from same_param_training.dataset import  get_client_loader
from same_param_training.train import train_model, test_model


app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    
    freeze_backbone = context.run_config["freeze-backbone"]
    model_name = context.run_config["model-name"]
    learning_rate = context.run_config["learning-rate"]
    weight_decay = context.run_config["weight-decay"]
    local_epochs = context.run_config["local-epochs"]
    mode = context.run_config["mode"]
    proximal_mu = context.run_config["proximal-mu"]
    augmented = context.run_config["data-augmented"]
    
    model = our_model(model_name, freeze_backbone)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    client_id = context.node_config["partition-id"]+1
    n_clients = context.node_config["num-partitions"]

    client_loader, _ = get_client_loader(client_id, n_clients, mode=mode, augmented=augmented)

    # Get Weights from the Server
    weights = msg.content["arrays"].to_torch_state_dict()
    
    # Set the current model weights to the weights sent by the server
    model.load_state_dict(weights)
       
    # Train the model  
    train_loss = train_model(model, learning_rate, weight_decay, local_epochs, device, client_loader, proximal_mu=proximal_mu)
    
    # Get New Model Weights
    new_weights = ArrayRecord(model.state_dict())
    
    # Send the weights and metrics back to the server
    metrics = MetricRecord({
        "train_loss" : train_loss,
        "num-examples": len(client_loader.dataset),
        "client-id": client_id,
    })
    return Message(
        content=RecordDict({
            "arrays": new_weights,
            "metrics": metrics, 
        }), 
        reply_to=msg
    )

@app.evaluate()
def evaluate(msg: Message, context: Context):
      
    freeze_backbone = context.run_config["freeze-backbone"]
    model_name = context.run_config["model-name"]
    mode = context.run_config["mode"]
    augmented = context.run_config["data-augmented"]
    model = our_model(model_name, freeze_backbone)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    client_id = context.node_config["partition-id"]+1
    n_clients = context.node_config["num-partitions"]
    
    _, val_loader = get_client_loader(client_id, n_clients, mode=mode, augmented=augmented)
    
    weights = msg.content["arrays"].to_torch_state_dict()
    
    model.load_state_dict(weights)    
    # Evaluate the Model
    loss, acc = test_model(model, device, val_loader)
    
    # Send the evaluation results back
    return Message(
        content=RecordDict({
            "metrics": MetricRecord({
                "eval_loss": loss,
                "eval_accuracy": acc,
                "num-examples": len(val_loader.dataset),
                "client-id" : client_id,
            })
        }), reply_to=msg
    )