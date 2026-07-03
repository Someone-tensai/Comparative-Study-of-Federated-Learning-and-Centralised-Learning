from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from same_param_training.models import our_model
from same_param_training.dataset import get_test_loader, get_client_loader
from same_param_training.train import train_model, test_model


app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    
    freeze_backbone = context.run_config["freeze_backbone"]
    model_name = context.run_config["model_name"]
    # Define the Model
    model = our_model(model_name, freeze_backbone)
    client_id = context.node_config["partition-id"]+1
    n_clients = context.node_config["num-partitions"]
    client_loader = get_client_loader(client_id, n_clients, mode="noniid")
    # Get Weights from the Server
    weights = msg.content["arrays"].to_torch_state_dict()
    
    # Set the current model weights to the weights sent by the server
    model.load_state_dict(weights)
       
    # Train the model weights 
    train_model(model, client_loader)
    
    # Get New Model Weights
    new_weights = ArrayRecord(model.state_dict())
    
    # Send the weights and metrics back to the server
    
    return Message(
        content=RecordDict({
            "arrays": ArrayRecord(new_weights),
            "metrics": MetricRecord({})
        })
    )

@app.evaluate()
def evaluate(msg: Message, context: Context):
      
    freeze_backbone = context.run_config["freeze_backbone"]
    model_name = context.run_config["model_name"]
    model = our_model(model_name, freeze_backbone)
    test_loader = get_test_loader()
    
    weights = msg.content["arrays"].to_torch_state_dict()
    
    model.load_state_dict(weights)    
    # Evaluate the Model
    loss, acc = test_model(model, test_loader)
    
    # Send the evaluation results back
    return Message(
        content=RecordDict({
            "metrics": MetricRecord({
                "loss": loss,
                "accuracy": acc
            })
        })
    )