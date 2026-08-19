import json
from pathlib import Path
from typing import Iterable

import torch
import wandb

from flwr.app import Message
from flwr.serverapp.strategy import FedAvg

from same_param_training.models import our_model
from same_param_training.dataset import get_test_loader
from same_param_training.train import test_model


class WandbFedAvg(FedAvg):
    def __init__(
        self,
        *args,
        model_name: str,
        freeze_backbone: bool = True,
        save_path: str = "results/history.json",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.model_name = model_name
        self.freeze_backbone = freeze_backbone

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.test_loader = get_test_loader()

        # FIX: previously _evaluate_global_model() called our_model(...)
        # every round, which re-constructs a fresh ResNet/VGG AND re-inits
        # it from the ImageNet pretrained weights each time, only to
        # immediately overwrite all of that with load_state_dict() a line
        # later. That's a full pretrained-weight init thrown away every
        # single round for nothing. Build the eval model once here instead,
        # and just swap its weights in per round.
        self.eval_model = our_model(
            self.model_name,
            freeze_backbone=self.freeze_backbone,
        ).to(self.device)

        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        self.history = []

    ####################################################################
    # Helpers
    ####################################################################

    def _get_round_entry(self, server_round):

        for entry in self.history:
            if entry["round"] == server_round:
                return entry

        entry = {
            "round": server_round,
            "clients": {}
        }

        self.history.append(entry)
        return entry

    def _save(self):
        with open(self.save_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def _save_checkpoint(self, server_round, arrays):

        # FIX: was Path("checkpoints") — the SAME folder for every run, so
        # checkpoints/round_5.pth from one experiment got silently
        # overwritten by round_5.pth from the next one. Namespace by
        # run_name (derived from save_path, which is already
        # results/{run_name}.json) so each run gets its own subfolder.
        checkpoint_dir = Path("checkpoints") / self.save_path.stem
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            arrays.to_torch_state_dict(),
            checkpoint_dir / f"round_{server_round}.pth"
        )

    ####################################################################
    # Centralized evaluation
    ####################################################################

    def _evaluate_global_model(self, arrays):

        # FIX: reuse the model built once in __init__ instead of
        # reconstructing (and re-initializing pretrained weights) every
        # round. Only the weights change per round now.
        self.eval_model.load_state_dict(arrays.to_torch_state_dict())

        loss, acc = test_model(
            self.eval_model,
            self.device,
            self.test_loader,
        )

        return loss, acc

    ####################################################################
    # Training aggregation
    ####################################################################

    def aggregate_train(self, server_round, replies: Iterable[Message]):

        replies = list(replies)

        entry = self._get_round_entry(server_round)

        ##########################################################
        # Client train metrics
        ##########################################################

        for msg in replies:

            if not msg.has_content():
                continue

            m = msg.content["metrics"]

            cid = m.get("client-id", "unknown")

            entry["clients"].setdefault(str(cid), {})

            entry["clients"][str(cid)]["train_loss"] = m["train_loss"]

            wandb.log(
                {
                    f"client_{cid}/train_loss": m["train_loss"]
                },
                step=server_round,
            )

        ##########################################################
        # Flower aggregation
        ##########################################################

        arrays, agg_metrics = super().aggregate_train(
            server_round,
            replies,
        )

        ##########################################################
        # Save checkpoint
        ##########################################################

        self._save_checkpoint(server_round, arrays)

        ##########################################################
        # Aggregated train metrics
        ##########################################################

        if agg_metrics is not None:

            entry["aggregated_train_loss"] = agg_metrics.get(
                "train_loss"
            )

            wandb.log(
                {
                    "train/aggregated_loss":
                        agg_metrics.get("train_loss")
                },
                step=server_round,
            )

        ##########################################################
        # NEW: Centralized evaluation
        ##########################################################

        test_loss, test_acc = self._evaluate_global_model(arrays)

        entry["global_test_loss"] = test_loss
        entry["global_test_accuracy"] = test_acc

        wandb.log(
            {
                "global_test/loss": test_loss,
                "global_test/accuracy": test_acc,
            },
            step=server_round,
        )

        self._save()

        return arrays, agg_metrics

    ####################################################################
    # Client evaluation aggregation
    ####################################################################

    def aggregate_evaluate(
        self,
        server_round,
        replies: Iterable[Message],
    ):

        replies = list(replies)

        entry = self._get_round_entry(server_round)

        for msg in replies:

            if not msg.has_content():
                continue

            m = msg.content["metrics"]

            cid = m.get("client-id", "unknown")

            entry["clients"].setdefault(str(cid), {})

            entry["clients"][str(cid)]["eval_loss"] = m["eval_loss"]
            entry["clients"][str(cid)]["eval_accuracy"] = m["eval_accuracy"]

            wandb.log(
                {
                    f"client_{cid}/eval_loss": m["eval_loss"],
                    f"client_{cid}/eval_accuracy": m["eval_accuracy"],
                },
                step=server_round,
            )

        agg_metrics = super().aggregate_evaluate(
            server_round,
            replies,
        )

        if agg_metrics is not None:

            entry["aggregated_eval_loss"] = agg_metrics.get(
                "eval_loss"
            )

            entry["aggregated_eval_accuracy"] = agg_metrics.get(
                "eval_accuracy"
            )

            wandb.log(
                {
                    "eval/aggregated_loss":
                        agg_metrics.get("eval_loss"),
                    "eval/aggregated_accuracy":
                        agg_metrics.get("eval_accuracy"),
                },
                step=server_round,
            )

        self._save()

        return agg_metrics