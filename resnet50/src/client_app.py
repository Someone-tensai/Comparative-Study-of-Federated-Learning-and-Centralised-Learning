from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    
    arrays = msg.content["arrays"].to_numpy_ndarrays()
    
    # Train the Model
    
    return Message()

@app.evaluate()
def evaluate(msg: Message, context: Context):
    
    arrays = msg.content["arrays"].to_numpy_ndarrays()
    
    # Evaluate the Model
    
    return Message()