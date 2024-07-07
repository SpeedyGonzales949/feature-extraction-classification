from dataloader.build import build_dataloader
from model.build import build_pretrained_model, build_evaluator
from utils.logger import Logger
from sklearn.manifold import TSNE

import traceback
import matplotlib.pyplot as plt
import os


def visualize(config, weights):
    """
    Visualizes the features of the data using the provided configuration and weights.

    Args:
        config (dict): A dictionary containing the configuration parameters.
        weights: The weights to be used for visualization.

    Raises:
        Exception: If an error occurs during visualization.

    Returns:
        None
    """
    try:
        logger = Logger(log_dir=config["save_dir"], log_file="visualize")
        start_visualize(config, logger, weights)
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
        raise e


def start_visualize(config, logger, weights=None):
    train_dataloader = build_dataloader(config["train_dataloader"])
    logger.info(f"<--- Train dataloader successfully loaded --->\n{train_dataloader}\n")

    model = build_pretrained_model(config, weights)
    logger.info(f"<--- Model successfully loaded --->\n{model}\n")

    logger.info("<--- Visuzlization started --->")
    y_pred, y_true, features = model.predict(
        train_dataloader, logger=logger, return_features=True
    )
    tsne = TSNE(n_components=2, perplexity=5.0, random_state=42)
    features_tsne = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    for label in config["model"]["params"]["classes"]:
        plt.scatter(
            features_tsne[y_true == label, 0],
            features_tsne[y_true == label, 1],
            label=label,
            s=30,
        )
    plt.title(f"t-SNE Visualization of Features")
    plt.legend(title="True Class")
    plt.savefig(
        os.path.join(config["save_dir"], "tsne_visualization.png"),
        format="png",
        dpi=300,
    )
    logger.info("<--- Visualization completed --->")
    logger.info(f"Visualization saved at {os.path.join(config['save_dir'], 'tsne_visualization.png')}")