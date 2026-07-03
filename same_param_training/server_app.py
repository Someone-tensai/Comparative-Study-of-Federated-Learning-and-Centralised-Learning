from flwr.serverapp import Grid, ServerApp
from flwr.app import ArrayRecord, Context
from flwr.serverapp.strategy import FedAvg
from same_param_training.models import our_model

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    
    num_rounds = context.run_config["num-server-rounds"]
    model_name = context.run_config["model_name"]
    model = our_model(model_name)
    arrays = ArrayRecord(model.state_dict())
    
    strategy = FedAvg()
    
    # Customize the strategy further
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds
    )
    
    # Results for later use

