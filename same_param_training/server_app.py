from flwr.serverapp import Grid, ServerApp
from flwr.app import ArrayRecord, Context
from flwr.serverapp.strategy import FedAvg
from src.model import our_model

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    
    num_rounds = context.run_config["num-server-rounds"]
    
    model = our_model()
    arrays = ArrayRecord(model)
    
    strategy = FedAvg()
    
    # Customize the strategy further
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds
    )
    
    # Results for later use

