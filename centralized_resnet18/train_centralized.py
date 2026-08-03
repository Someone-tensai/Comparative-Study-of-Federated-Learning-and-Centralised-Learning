import argparse
import json
from datetime import datetime
from pathlib import Path
 
import torch
import wandb
 
from same_param_training.dataset import get_train_loader, get_test_loader
from same_param_training.models import our_model
from same_param_training.train import train_model, test_model
 
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Centralized ResNet baseline training")
    parser.add_argument("--model-name", choices=["resnet18", "resnet50", "vgg16"], default="resnet50")
    parser.add_argument(
        "--freeze-backbone", action=argparse.BooleanOptionalAction, default=True,
        help="Passed straight to our_model(). For resnet50 this freezes the entire "
             "backbone (only the new fc head trains); for resnet18 it unfreezes layer4+fc. "
             "Kept on by default to match your existing federated ResNet18 runs.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--eval-every", type=int, default=1,
        help="Log to W&B every N epochs (see module docstring for the Adam-continuity trade-off).",
    )
    parser.add_argument("--wandb-project", type=str, default="centralized_training")
    parser.add_argument(
        "--wandb-entity", type=str, default="paudelsulav5-fedlearnproject",
        help="Shared W&B entity — matches the README's C.1 setup, so this run shows "
             "up in the same team project space as your federated runs.",
    )

    return parser.parse_args()
 
 
def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    run_name = f"{args.model_name}-centralized-{args.epochs}-{datetime.now():%Y-%m-%d}"
    wandb.init(entity=args.wandb_entity, project= 'same_param_training', name=run_name, config=vars(args)) 
    train_loader = get_train_loader(args.batch_size)
    test_loader = get_test_loader(args.batch_size)
 
    model = our_model(args.model_name, args.freeze_backbone)
    model.to(device)
 
    try:
        completed = 0
        best_test_acc = 0.0
        test_loss, test_acc = None, None
 
        while completed < args.epochs:
            block = min(args.eval_every, args.epochs - completed)
            train_loss = train_model(
                model, args.learning_rate, args.weight_decay,
                local_epochs=block, device=device, train_loader=train_loader,
            )
            completed += block
            test_loss, test_acc = test_model(model, device, test_loader)
            best_test_acc = max(best_test_acc, test_acc)
 
            print(
                f"epoch {completed:>3}/{args.epochs}  train_loss={train_loss:.4f}  "
                f"test_loss={test_loss:.4f}  test_acc={test_acc:.4f}"
            )
            wandb.log(
                {
                    "epoch": completed,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "test_accuracy": test_acc,
                },
                step=completed,
            )
 
        Path("results").mkdir(exist_ok=True)
        torch.save(model.state_dict(), f"results/{run_name}.pth")
        with open(f"results/{run_name}.json", "w") as f:
            json.dump(
                {
                    "config": vars(args),
                    "best_test_accuracy": best_test_acc,
                    "final_test_loss": test_loss,
                    "final_test_accuracy": test_acc,
                },
                f,
                indent=2,
            )
        print(f"Saved -> results/{run_name}.pth, results/{run_name}.json")
 
    finally:
        wandb.finish()
 
 
if __name__ == "__main__":
    main()
