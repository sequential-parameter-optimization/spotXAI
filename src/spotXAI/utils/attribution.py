import pandas as pd

from captum.attr import IntegratedGradients, DeepLift, GradientShap, FeatureAblation, LayerConductance, KernelShap

from spotpython.hyperparameters.optimizer import optimizer_handler

import torchmetrics.functional.regression
from torch.utils.data import DataLoader, Subset
import torch

import numpy as np

from scipy.stats import ttest_1samp

from sklearn.preprocessing import StandardScaler


class spotXAI:
    """
    Spot XAI class capable of performing the following tasks:
    * Training the weights of a PyTorch model.
    * Applying feature attribution methods to the trained model.
    """

    def __init__(self, model, data, training_attributes, train_split, scale_data=True, seed=123):
        """
        Initialize the spotLight instance.

        Parameters:
            model (torch.nn.Module): The PyTorch model to be trained.
            data (torch.utils.data.Dataset): A pyTorch dataset.
            training_attributes (dict): Dictionary containing training attributes,
                                         including hyperparameters and optimizer details.
            train_split (float): The proportion of the dataset to be used for training.
            scale_data (bool): if True: use standard scaler to scale the data sets.
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

        if scale_data == True:
            self.scaler = StandardScaler()

            # Extract the features and labels from the datasets
            train_data = np.array([self.train_set[i][0].numpy() for i in range(len(self.train_set))])
            test_data = np.array([self.test_set[i][0].numpy() for i in range(len(self.test_set))])

            self.scaler.fit(train_data)

            scaled_train_data = self.scaler.transform(train_data)
            scaled_test_data = self.scaler.transform(test_data)
            self.train_set = self._replace_dataset(self.train_set, scaled_train_data)
            self.test_set = self._replace_dataset(self.test_set, scaled_test_data)

        self.train_loader = DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True, drop_last=True, pin_memory=True
        )
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size)

    def _replace_dataset(self, dataset, new_data):
        """
        Replace the features in the dataset with the scaled features.
        """
        new_dataset = []
        for i, (features, target) in enumerate(dataset):
            new_features = torch.tensor(new_data[i], dtype=torch.float32)
            new_dataset.append((new_features, target))
        return Subset(new_dataset, range(len(new_dataset)))

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
            # print("epochs {}/{}".format(epoch + 1, self.epochs))
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
            # print("train loss: ", train_loss.item())
            train_losses.append(train_loss.detach().numpy())

    def get_n_most_sig_features(self, n_rel=20, attr_method="IntegratedGradients", baseline=None, abs_attr=True):
        """
        Compute the n most significant features for a given model and test set using a specified attribution methods.

        Args:
        - n_rel (int, optional): Number of most significant features to return. Defaults to 20.
        - attr_method (str, optional): Attribution method to use. Choose from 'IntegratedGradients', 'DeepLift', or 'FeatureAblation'. Defaults to 'IntegratedGradients'.
        - plot (bool, optional): Whether to plot the attribution scores. Defaults to True.
        - baseline (torch.Tensor, optional): Baseline for the attribution methods. Baseline is defined as input features. Defaults to None.
        - abs_attr (bool, optional): Wether the method should sort by the absolute attribution values. Defaults to True.

        Returns:
        - df (pd.DataFrame): A DataFrame containing the indices of the n important features with their corresponding attribution values.
        """
        self.model.eval()
        total_attributions = None

        if attr_method == "IntegratedGradients":
            attr = IntegratedGradients(self.model)
        elif attr_method == "DeepLift":
            attr = DeepLift(self.model)
        elif attr_method == "GradientShap":  # Todo: would need a baseline
            if baseline is None:
                raise ValueError("baseline cannot be 'None' for GradientShap")
            attr = GradientShap(self.model)
        elif attr_method == "FeatureAblation":
            attr = FeatureAblation(self.model)
        elif attr_method == "KernelShap":
            attr = KernelShap(self.model)
        else:
            raise ValueError(
                "Unsupported attribution method. Please choose from 'IntegratedGradients', 'DeepLift', 'GradientShap', 'KernelShap', or 'FeatureAblation'."
            )

        if baseline is not None:
            self.scaled_baseline = torch.Tensor(self.scaler.transform(baseline))

        for inputs, labels in self.test_loader:
            for i in range(len(inputs)):
                attribution = attr.attribute(inputs[i].unsqueeze(0), baselines=self.scaled_baseline)
                if total_attributions is None:
                    total_attributions = attribution
                else:
                    if len(attribution) == len(total_attributions):
                        total_attributions += attribution

        # Calculation of average attribution across all batches
        avg_attributions = total_attributions.mean(dim=0).detach().numpy()

        # Transformation to the absolute attribution values if abs_attr is True
        # Get indices of the n most important features
        if abs_attr is True:
            abs_avg_attributions = abs(avg_attributions)
            top_n_indices = abs_avg_attributions.argsort()[-n_rel:][::-1]
        else:
            top_n_indices = avg_attributions.argsort()[-n_rel:][::-1]

        # Get the importance values for the top n features
        top_n_importances = avg_attributions[top_n_indices]

        df = pd.DataFrame({"Feature Index": top_n_indices, attr_method + "Attribution": top_n_importances})
        return df

    def get_attribution_distribution(self, attr_method="IntegratedGradients", baseline=None, abs_attr=True):
        """
        Compute the feature attribution distribution for a given model and test set using a specified attribution methods.

        Args:
        - attr_method (str, optional): Attribution method to use. Choose from 'IntegratedGradients', 'DeepLift', or 'FeatureAblation'. Defaults to 'IntegratedGradients'.
        - baseline (torch.Tensor, optional): Baseline for the attribution methods. Baseline is defined as input features. Defaults to None.
        - abs_attr (bool, optional): Whether the method should sort by the absolute attribution values. Defaults to True.

        Returns:
        - df (pd.DataFrame): A DataFrame containing the indices of the n important features with their corresponding attribution and target values.
        """
        self.model.eval()

        if attr_method == "IntegratedGradients":
            attr = IntegratedGradients(self.model)
        elif attr_method == "DeepLift":
            attr = DeepLift(self.model)
        elif attr_method == "GradientShap":  # Todo: would need a baseline
            if baseline is None:
                raise ValueError("baseline cannot be 'None' for GradientShap")
            attr = GradientShap(self.model)
        elif attr_method == "FeatureAblation":
            attr = FeatureAblation(self.model)
        elif attr_method == "KernelShap":
            attr = KernelShap(self.model)
        else:
            raise ValueError(
                "Unsupported attribution method. Please choose from 'IntegratedGradients', 'DeepLift', 'GradientShap', 'KernelShap', or 'FeatureAblation'."
            )

        attribution_values = []
        feature_indices = []
        y_values = []

        for inputs, labels in self.test_loader:
            # Batch processing
            attributions = attr.attribute(inputs, baselines=baseline)

            if abs_attr:
                attributions = attributions.abs()

            attributions = attributions.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()

            # Flatten attributions and corresponding labels for DataFrame
            batch_size, num_features = attributions.shape
            for i in range(batch_size):
                for j in range(num_features):
                    attribution_values.append(attributions[i, j])
                    feature_indices.append(j)
                    y_values.append(labels[i])

        df = pd.DataFrame(
            {
                "attribution value": attribution_values,
                "feature index": feature_indices,
                "corresponding y_value": y_values,
            }
        )

        return df

    def get_sig_features_t_test(self, alpha=0.05, attr_method="IntegratedGradients", baseline=None, abs_attr=True):
        """
        Performs a one-tailed t-test to check which feature attributions differ significantly from 0.

        Args:
        - alpha (float, optional): The significance level for the t-test. Defaults to 0.05.
        - attr_method (str, optional): Attribution method to use. Choose from 'IntegratedGradients', 'DeepLift', or 'FeatureAblation'. Defaults to 'IntegratedGradients'.
        - baseline (torch.Tensor, optional): Baseline for the attribution methods. Baseline is defined as input features. Defaults to None.

        Returns:
        - df (pd.DataFrame): A DataFrame containing the indices of the features deemed significant and their corresponding p-values.
        """
        self.model.eval()
        total_attributions = None

        if attr_method == "IntegratedGradients":
            attr = IntegratedGradients(self.model)
        elif attr_method == "DeepLift":
            attr = DeepLift(self.model)
        elif attr_method == "GradientShap":  # Todo: would need a baseline
            if baseline is None:
                raise ValueError("baseline cannot be 'None' for GradientShap")
            attr = GradientShap(self.model)
        elif attr_method == "FeatureAblation":
            attr = FeatureAblation(self.model)
        else:
            raise ValueError(
                "Unsupported attribution method. Please choose from 'IntegratedGradients', 'DeepLift', 'GradientShap', or 'FeatureAblation'."
            )

        all_attributions = []

        # Loop through the test loader and collect attributions
        for inputs, labels in self.test_loader:
            attributions = attr.attribute(inputs, baselines=baseline)
            if abs_attr:
                attributions = torch.abs(attributions)
            all_attributions.append(attributions.cpu().numpy())

        # Concatenate all attributions
        total_attributions = np.concatenate(all_attributions, axis=0)

        print(total_attributions)
        print(len(total_attributions))
        print(total_attributions.shape)

        # Perform t-test
        t_statistic, p_value = ttest_1samp(total_attributions, popmean=0, axis=0)

        # Create a DataFrame with results
        res = [{"Feature Index": i, "p value": p, "Significant": p < alpha} for i, p in enumerate(p_value)]

        df = pd.DataFrame(res)
        return df

    def get_layer_conductance(self, layer_idx):
        """
        Compute the average layer conductance attributions for a specified layer in the model.

        Args:
        - layer_idx (int): Index of the layer for which to compute layer conductance attributions.

        Returns:
        - numpy.ndarray: An array containing the average layer conductance attributions for the specified layer. The shape of the array corresponds to the shape of the attributions.
        """
        self.model.eval()
        total_layer_attributions = None
        layers = self.model.layers
        model = self.model
        print("Conductance analysis for layer: ", layers[layer_idx])
        lc = LayerConductance(model, layers[layer_idx])

        for inputs, labels in self.test_loader:
            lc_attr_test = lc.attribute(inputs, n_steps=10, attribute_to_layer_input=True)
            if total_layer_attributions is None:
                total_layer_attributions = lc_attr_test
            else:
                if len(lc_attr_test) == len(total_layer_attributions):
                    total_layer_attributions += lc_attr_test

        avg_layer_attributions = total_layer_attributions.mean(dim=0).detach().numpy()

        return avg_layer_attributions
