import json
from pathlib import Path
from typing import Iterable
from flwr.app import ArrayRecord, Message, MetricRecord
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg
import wandb

class WandbFedAvg(FedAvg):
    def __init__(self, *args, save_path : str = "results/history.json" ,**kwargs):
        super().__init__(*args, **kwargs)
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.history = []

    # Helper function to create a entry such that agg_train and agg_eval are in the same entry
    def _get_round_entry(self, server_round):
        
        for entry in self.history:
            if entry["round"] == server_round:
                return entry
        
        entry = {"round" : server_round, "clients" : {}}
        self.history.append(entry)
        return entry
    
    # Save the history to a json file
    def _save(self):
        with open(self.save_path, "w") as f:
            json.dump(self.history, f, indent=2)
            
            
    def aggregate_train(self, server_round, replies: Iterable[Message]):
        replies = list(replies)
        entry = self._get_round_entry(server_round)
        
        # Log the data in wandb and add to history
        for msg in replies:
            if msg.has_content():
                m = msg.content["metrics"]
                cid = m.get("client-id", "unknown")
                entry["clients"].setdefault(str(cid), {})["train_loss"] = m["train_loss"]                
                wandb.log(
                    {f"client_{cid}/train_loss": m["train_loss"]},
                    step=server_round,
                )

        # Actually call the aggregator from FedAvg
        arrays, agg_metrics = super().aggregate_train(server_round, replies)
        
        # Log the aggegrated metrics in wand and history
        if agg_metrics is not None:
            entry["aggregated_train_loss"] = agg_metrics.get("train_loss")
            wandb.log({"train/aggregated_loss": agg_metrics.get("train_loss")}, step=server_round)
        self._save()
        return arrays, agg_metrics


    def aggregate_evaluate(self, server_round, replies: Iterable[Message]):
        replies = list(replies)
        entry = self._get_round_entry(server_round)

        for msg in replies:
            if msg.has_content():
                m = msg.content["metrics"]
                cid = m.get("client-id", "unknown")
                entry["clients"].setdefault(str(cid), {})["eval_loss"] = m["eval_loss"]
                entry["clients"][str(cid)]["eval_accuracy"] = m["eval_accuracy"]
                wandb.log({
                    f"client_{cid}/eval_loss": m["eval_loss"],
                    f"client_{cid}/eval_accuracy": m["eval_accuracy"],
                }, step=server_round)

        agg_metrics = super().aggregate_evaluate(server_round, replies)
        if agg_metrics is not None:
            entry["aggregated_eval_loss"] = agg_metrics.get("eval_loss")
            entry["aggregated_eval_accuracy"] = agg_metrics.get("eval_accuracy")
            wandb.log({
                "eval/aggregated_loss": agg_metrics.get("eval_loss"),
                "eval/aggregated_accuracy": agg_metrics.get("eval_accuracy"),
            }, step=server_round)
        self._save()
        return agg_metrics