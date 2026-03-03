"""
Module to train a bioFAM model
"""

from rich import box
from rich.panel import Panel

from mofapy2 import console
from mofapy2.core.BayesNet import BayesNet


def train_model(model: BayesNet) -> None:
    # Sanity check on the Bayesian Network
    if not isinstance(model, BayesNet):
        msg = "'model' has to be a BayesNet class"
        raise TypeError(msg)

    console.print(Panel(f"Training the model with seed {model.options['seed']!s}", box=box.HEAVY, style="bold blue"))

    model.iterate()

    console.print(Panel("Training finished", box=box.HEAVY, style="bold green"))
