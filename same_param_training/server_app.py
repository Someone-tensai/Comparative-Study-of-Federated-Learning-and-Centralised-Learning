from flwr.serverapp import Grid, ServerApp
from flwr.app import ArrayRecord, Context
from same_param_training.models import our_model
from same_param_training.custom_strategy import WandbFedAvg
import wandb
from datetime import datetime

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    
    num_rounds = context.run_config["num-server-rounds"]
    model_name = context.run_config["model-name"]
    
    run_name = f"{model_name}-{num_rounds}rounds-{datetime.now():%Y%m%d_%H%M%S}"

    wandb.init(
        project="same_param_fed_training",
        name=run_name,
        config=dict(context.run_config)
    )
    model = our_model(model_name)
    arrays = ArrayRecord(model.state_dict())
    
    strategy = WandbFedAvg(save_path=f"results/{run_name}.json")
    
    # Customize the strategy further
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds
    )
    
            
    wandb.finish()

