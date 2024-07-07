from dataloader.build import build_dataloader
from model.build import build_pretrained_model, build_evaluator
from utils.logger import Logger

import traceback
import numpy as np
from model.models import ExtractorPredictor


def test(config, weights):
    """
    Run the test process using the given configuration and weights.

    Args:
        config (dict): A dictionary containing the configuration parameters.
        weights (str): The path to the weights file.

    Raises:
        Exception: If an error occurs during the test process.

    Returns:
        None
    """
    try:
        logger = Logger(log_dir=config["save_dir"], log_file="test")
        start_test(config, logger, weights)
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
        raise e


def start_test(config, logger, weights=None):
    test_dataloader = build_dataloader(config["test_dataloader"])
    logger.info(f"<--- Test dataloader successfully loaded --->\n{test_dataloader}\n")

    model = build_pretrained_model(config, weights)
    logger.info(f"<--- Model successfully loaded --->\n{model}\n")
    
    evaluator = build_evaluator(config)
    logger.info(f"<--- Evaluator successfully loaded --->\n{evaluator}\n")

    logger.info("<--- Prediction started --->")
    y_pred, y_true = model.predict(test_dataloader, logger=logger)

    logger.info("<--- Evaluation started --->")
    metrics = evaluator.evaluate(y_true, y_pred)

    logger.info("<--- Evaluation results --->")
    logger.info(f"Reduced Metrics:")
    for metric, value in metrics:
        if isinstance(value, float):
            logger.info(f"{metric}: {value:.4f}")

    logger.info(f"Class Metrics:")
    for class_idx in range(config["num_classes"]):
        class_metrics = ", ".join([f"{metric}: {value[class_idx]}" for metric, value in metrics if isinstance(value, np.ndarray)])
        logger.info(f"Class {class_idx} metrics: {class_metrics}")
