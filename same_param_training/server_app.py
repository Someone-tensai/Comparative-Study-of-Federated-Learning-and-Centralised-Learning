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
    # FIX: freeze_backbone was never read here, and model_name was never
    # forwarded to WandbFedAvg/WandbFedProx below even though both classes
    # require it (model_name has no default in either __init__). That meant
    # constructing WandbFedAvg — the DEFAULT strategy — raised a TypeError
    # before a single round could run.
    freeze_backbone = context.run_config["freeze-backbone"]

    # FIX: the original built this with an f-string that reused double
    # quotes inside an outer double-quoted f-string
    # (f"...{"augmented" if augmented else ""}..."). That syntax is only
    # legal on Python 3.12+ (PEP 701) and raises a SyntaxError on 3.10/3.11,
    # with no requires-python pin in pyproject.toml to warn you. Building
    # the tag values as plain variables first sidesteps the whole issue and
    # is more readable anyway.
    augmented_tag = "augmented" if augmented else ""
    lr_tag = "Default" if learning_rate == 0.001 else "LRR"
    mu_tag = str(proximal_mu) if proximal_mu != 0 else ""
    date_tag = datetime.now().strftime("%Y-%m-%d")

    run_name = f"{model_name}-layer4_in-{mode}-{strat}-{num_rounds}-{augmented_tag}-{lr_tag}-{mu_tag}-{date_tag}"

    wandb.init(
        project="same_param_fed_training",
        name=run_name,
        config=dict(context.run_config)
    )

    try:
        model = our_model(model_name)
        arrays = ArrayRecord(model.state_dict())

        if strat == "fedprox":
            strategy = WandbFedProx(
                model_name=model_name,
                freeze_backbone=freeze_backbone,
                save_path=f"results/{run_name}.json",
            )
        else:
            strategy = WandbFedAvg(
                model_name=model_name,
                freeze_backbone=freeze_backbone,
                save_path=f"results/{run_name}.json",
            )
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