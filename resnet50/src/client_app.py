from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from src.model import our_model
from src.dataset import test_loader, train_loader
from src.train import train_model, test_model
from src.utils import get_parameters, set_parameters

app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    
    # Define the Model
    model = our_model()
    
    # Get Weights from the Server
    weights = msg.content["arrays"].to_numpy_ndarrays()
    
    # Set the current model weights to the weights sent by the server
    set_parameters(model, weights) 
       
    # Train the model weights 
    train_model(model,train_loader)
    
    # Get New Model Weights
    new_weights = get_parameters(model)
    
    # Send the weights and metrics back to the server
    
    return Message(
        content=RecordDict({
            "arrays": ArrayRecord(new_weights),
            "metrics": MetricRecord({})
        })
    )

@app.evaluate()
def evaluate(msg: Message, context: Context):
    
    model = our_model()
    
    weights = msg.content["arrays"].to_numpy_ndarrays()
    
    set_parameters(model, weights)
    
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