from flwr.serverapp import Grid, ServerApp
from flwr.app import ArrayRecord, Context
from same_param_training.models import our_model
from same_param_training.custom_strategy import WandbFedAvg
from same_param_training.custom_strat_fed_prox import WandbFedProx
import wandb
from datetime import datetime
import torch
from pathlib import Path

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    
    num_rounds = context.run_config["num-server-rounds"]
    model_name = context.run_config["model-name"]
    strat = context.run_config["fed-strategy"]
    mode = context.run_config["mode"]
    augmented = context.run_config["data-augmented"]
    learning_rate = context.run_config["learning-rate"]
    proximal_mu = context.run_config["proximal-mu"]
    
    
    run_name = f"{model_name}-layer4_in-{mode}-{strat}-{num_rounds}-{"augmented" if augmented else ""}-{"Default" if learning_rate==0.001 else "LRR"}-{proximal_mu if proximal_mu != 0 else ""}-{datetime.now():%Y-%m-%d}"

    wandb.init(
        project="same_param_fed_training",
        name=run_name,
        config=dict(context.run_config)
    )
    
    try:
        model = our_model(model_name)
        arrays = ArrayRecord(model.state_dict())
        
        if(strat == "fedprox"):
            strategy = WandbFedProx(save_path=f"results/{run_name}.json")
        else:
            strategy = WandbFedAvg(save_path=f"results/{run_name}.json")
        # Customize the strategy further
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            num_rounds=num_rounds
        )
        
        final_arrays = result.arrays  # ArrayRecord with the final global weights
        state_dict = final_arrays.to_torch_state_dict()
        model.load_state_dict(state_dict)
        Path("results").mkdir(exist_ok=True)
        torch.save(model.state_dict(), f"results/{run_name}.pth")
        
    finally:
        wandb.finish()

