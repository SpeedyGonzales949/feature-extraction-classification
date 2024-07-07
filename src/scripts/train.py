from dataloader.build import build_dataloader
from model.build import build_model, build_evaluator
from utils.logger import Logger

import traceback
import numpy as np
import os



def train(config):
    """
    Trains the model using the provided configuration. Exception wrapper for logging.

    Args:
        config (dict): A dictionary containing the configuration parameters for training.

    Raises:
        Exception: If an error occurs during training.
    """
    try:
        logger = Logger(log_dir=config["save_dir"], log_file="train")
        start_train(config, logger)
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
        raise e

def start_train(config, logger):  
    train_dataloader = build_dataloader(config["train_dataloader"])
    logger.info(f"<--- Train dataloader successfully loaded --->\n{train_dataloader}\n")

    validation_dataloader = build_dataloader(config["validation_dataloader"])
    logger.info(f"<--- Validation dataloader successfully loaded --->\n{validation_dataloader}\n")


    model = build_model(config)
    logger.info(f"<--- Model successfully loaded --->\n{model}\n")


    evaluator = build_evaluator(config)
    logger.info(f"<--- Evaluator successfully loaded --->\n{evaluator}\n")


    logger.info("<--- Training started --->")
    model.train(train_dataloader, logger=logger)

    logger.info("<--- Prediction started --->")
    y_pred, y_true = model.predict(validation_dataloader, logger=logger)

    logger.info("<--- Evaluation started --->")
    metrics = evaluator.evaluate(y_true, y_pred)

    logger.info("<--- Evaluation results --->")
    logger.info(f"Reduced Metrics:")
    for metric, value in metrics:
        if isinstance(value, float):
            logger.info(f"{metric}: {value}")
    
    logger.info(f"Class Metrics:")
    for class_idx in range(config["num_classes"]):
        class_metrics = ", ".join([f"{metric}: {value[class_idx]}" for metric, value in metrics if isinstance(value, np.ndarray)])
        logger.info(f"Class {class_idx} metrics: {class_metrics}")

    logger.info("<--- Saving model --->")
    model.save_model(os.path.join(config["save_dir"], "model_best.pkl"))
    logger.info("Model saved at: " + os.path.join(config["save_dir"], "model_best.pkl"))