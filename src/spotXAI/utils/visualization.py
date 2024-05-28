import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd


def boxplot_attribution(attribution_distribution):
    """
    Generate a box plot to visualize the attribution values for each feature.

    Parameters:
    attribution_distribution (list): A list of attribution distributions for each feature.

    Returns:
    None
    """
    attribution_matrix = torch.stack(attribution_distribution)

    # Convert the 2D tensor to a NumPy array for easier manipulation
    attribution_matrix_np = attribution_matrix.numpy()

    # Prepare the data for plotting
    data = []
    for feature_index in range(attribution_matrix_np.shape[1]):
        for value in attribution_matrix_np[:, feature_index]:
            data.append([feature_index, value])

    # Convert to a pandas DataFrame
    df = pd.DataFrame(data, columns=["Feature Index", "Attribution Value"])

    # Create the box plot
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="Attribution Value", y="Feature Index", data=df, orient="h")

    # Customize the plot
    plt.title("Box Plot of Attribution Values for Each Feature")
    plt.xlabel("Attribution Value")
    plt.ylabel("Feature Index")
    plt.grid(True)

    # Show the plot
    plt.show()
