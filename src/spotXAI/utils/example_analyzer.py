from spotXAI import spotXAI

from spotpython.data.diabetes import Diabetes
from spotpython.light.regression.netlightregression import NetLightRegression
from spotpython.utils.classes import get_removed_attributes_and_base_net

from torch.nn import ReLU
import torch

import numpy as np


def example_analyzer(seed=123):
    """
    Creates an exemplary instance of the spotXAI class for demonstration and testing purposes.

    Args:
    - seed (int, optional): Seed value for random number generation. Defaults to 123.

    Returns:
    - analyzer (spotXAI): An instance of the spotXAI class initialized with example configurations and data.

    Example:
    ```python
    analyzer = example_analyzer(seed=42)
    ```
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Define example configuration for the model
    example_config = {
        "l1": 64,
        "epochs": 1024,
        "batch_size": 32,
        "act_fn": ReLU(),
        "optimizer": "AdamW",
        "dropout_prob": 0.04938229888019609,
        "lr_mult": 2.3689895017756495,
        "patience": 64,
        "initialization": "Default",
    }

    # Create a sample instance of the model
    model = NetLightRegression(**example_config, _L_in=10, _L_out=1, _torchmetric="mean_squared_error")

    # Get removed attributes and base network
    removed_attributes, torch_net = get_removed_attributes_and_base_net(net=model)

    # Create a sample dataset
    dataset = Diabetes(target_type=torch.float)

    # Initialize spotXAI instance with example configurations and data
    analyzer = spotXAI(model=torch_net, data=dataset, train_split=0.6, training_attributes=removed_attributes, seed=123)

    # Train the model
    analyzer.train_model()

    return analyzer
