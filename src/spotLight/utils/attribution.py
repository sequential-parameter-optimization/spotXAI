import pandas as pd

from captum.attr import IntegratedGradients, DeepLift, GradientShap, FeatureAblation

from spotPython.hyperparameters.optimizer import optimizer_handler

import torchmetrics.functional.regression
from torch.utils.data import DataLoader
import torch

import numpy as np


class spotXAI:
    """
    Spot XAI class capable of performing the following tasks:
    * Training the weights of a PyTorch model.
    * Applying feature attribution methods to the trained model.
    """

    def __init__(self, model, data, training_attributes, train_split, seed=123):
        """
        Initialize the spotLight instance.

        Parameters:
            model (torch.nn.Module): The PyTorch model to be trained.
            data (torch.utils.data.Dataset): A pyTorch dataset.
            training_attributes (dict): Dictionary containing training attributes,
                                         including hyperparameters and optimizer details.
            train_split (float): The proportion of the dataset to be used for training.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.batch_size = training_attributes.get("_hparams_initial", {}).get("batch_size")
        self.optimizer = optimizer_handler(
            optimizer_name=training_attributes.get("_hparams_initial", {}).get("optimizer"),
            params=model.parameters(),
            lr_mult=training_attributes.get("_hparams_initial", {}).get("lr_mult"),
        )
        self.epochs = training_attributes.get("_hparams_initial", {}).get("epochs")
        self.criterion = getattr(torchmetrics.functional.regression, training_attributes.get("_torchmetric", None))
        self.model = model
        self.dataset = data
        self.train_split = train_split
        self.train_set, self.test_set = torch.utils.data.random_split(
            self.dataset, [self.train_split, 1 - self.train_split]
        )
        self.train_loader = DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True, drop_last=True, pin_memory=True
        )
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size)

    def train_model(self):
        """
        Train the weights of a plain PyTorch model.

        This method trains the weights of the provided PyTorch model using the specified
        optimizer and criterion, along with the training data loader.

        Returns:
        None

        Note:
        This method assumes that the following attributes are already initialized:
            - self.epochs: Number of epochs for training.
            - self.model: PyTorch model to be trained.
            - self.optimizer: Optimizer used for updating the model parameters.
            - self.criterion: Loss function used for calculating the loss.
            - self.train_loader: Data loader for the training dataset.
            - self.train_set: Training dataset.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        train_losses = []
        for epoch in range(self.epochs):
            print("epochs {}/{}".format(epoch + 1, self.epochs))
            self.model.train()
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                self.optimizer.zero_grad()
                preds = self.model(inputs)
                labels = labels.view(len(labels), 1)
                loss = self.criterion(preds, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss

            train_loss = running_loss / len(self.train_set)
            print("train loss: ", train_loss.item())
            train_losses.append(train_loss.detach().numpy())

    def get_n_most_sig_features(self, n_rel=20, attr_method="IntegratedGradients", baseline=None):
        """
        Compute the n most significant features for a given model and test set using a specified attribution methods.

        Args:
        - n_rel (int, optional): Number of most significant features to return. Defaults to 20.
        - attr_method (str, optional): Attribution method to use. Choose from 'IntegratedGradients', 'DeepLift', or 'FeatureAblation'. Defaults to 'IntegratedGradients'.
        - plot (bool, optional): Whether to plot the attribution scores. Defaults to True.

        Returns:
        - selected_indices (list): Indices of the n most significant features.
        - selected_importance (list): Importance scores of the n most significant features.
        """
        self.model.eval()
        total_attributions = None

        if attr_method == "IntegratedGradients":
            attr = IntegratedGradients(self.model)
        elif attr_method == "DeepLift":
            attr = DeepLift(self.model)
        elif attr_method == "GradientShap":  # Todo: would need a baseline
            if baseline == None:
                raise ValueError("baseline cannot be 'None' for GradientShap")
            attr = GradientShap(self.model)
        elif attr_method == "FeatureAblation":
            attr = FeatureAblation(self.model)
        else:
            raise ValueError(
                "Unsupported attribution method. Please choose from 'IntegratedGradients', 'DeepLift', 'GradientShap', or 'FeatureAblation'."
            )

        for inputs, labels in self.test_set:
            inputs = inputs.unsqueeze(0)
            attributions = attr.attribute(inputs, return_convergence_delta=False, baselines=baseline)
            if total_attributions is None:
                total_attributions = attributions
            else:
                if len(attributions) == len(total_attributions):
                    total_attributions += attributions

        # Calculation of average attribution across all batches
        avg_attributions = total_attributions.mean(dim=0).detach().numpy()

        # Get indices of the 10 most important features
        top_n_indices = avg_attributions.argsort()[-n_rel:][::-1]

        # Get the importance values for the top 10 features
        top_n_importances = avg_attributions[top_n_indices]

        df = pd.DataFrame({"Feature Index": top_n_indices + 1, attr_method + "Attribution": top_n_importances})
        return df
