import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def boxplot_attribution(attribution_distribution_df, size=(12, 8), attr_filter=None):
    """
    Generate a box plot to visualize the attribution values for each feature.

    Parameters:
    attribution_distribution_df (data frame): A data frame of attribution distributions for each feature.
    size (tuple, optional): The size of the plot (width, height). Default is (12, 8).
    attr_filter (tuple, optional): The y-axis limits for the plot. Default is None.

    Returns:
    None
    """

    # Create the box plot
    fig, ax = plt.subplots(figsize=size)
    sns.boxplot(x="attribution value", y="feature index", data=attribution_distribution_df, orient="h")

    if attr_filter is not None:
        ax.set_ylim(attr_filter)

    # Customize the plot
    plt.title("Box Plot of Attribution Values for Each Feature")
    plt.xlabel("Attribution Value")
    plt.ylabel("Feature Index")
    plt.grid(True)

    plt.tight_layout()

    # Show the plot
    plt.show()


def scatter_attribution(df, size=(12, 8), attr_filter=None, param_list=None):
    """
    Generate a scatter plot to visualize the attribution values for each feature.

    Parameters:
    df (data frame): A data frame containing attribution value, feature index and corresponding y value.
    size (tuple, optional): The size of the plot (width, height). Default is (12, 8).
    attr_filter (tuple, optional): The y-axis limits for the plot. Default is None.
    param_list (list, optional): A list of the feature names.

    Returns:
    None
    """
    if param_list:
        param_df = pd.DataFrame({"Index": range(len(param_list)), "Parameter": param_list})
        attribution_distribution_df = pd.merge(df, param_df, left_on="feature index", right_on="Index")
        fig, ax = plt.subplots(figsize=size)
        scatter = sns.scatterplot(
            x="Parameter",
            y="attribution value",
            hue="corresponding y_value",
            palette="viridis",
            data=attribution_distribution_df,
            ax=ax,
            legend=False,
        )
    else:
        attribution_distribution_df = df
        fig, ax = plt.subplots(figsize=size)
        scatter = sns.scatterplot(
            x="feature index",
            y="attribution value",
            hue="corresponding y_value",
            palette="viridis",
            data=attribution_distribution_df,
            ax=ax,
            legend=False,
        )

    if attr_filter is not None:
        ax.set_ylim(attr_filter)

    # Customize the plot
    plt.title("Scatter Plot of Attribution Values for Each Feature")
    plt.xlabel("Feature")
    plt.ylabel("Attribution Value")
    if param_list:
        plt.xticks(rotation=90)

    # Create a colorbar
    norm = plt.Normalize(
        attribution_distribution_df["corresponding y_value"].min(),
        attribution_distribution_df["corresponding y_value"].max(),
    )
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Target Variable Value")

    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.show()
