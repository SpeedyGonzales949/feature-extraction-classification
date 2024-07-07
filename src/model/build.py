from .extractor_predictor.feature_extractor import (
    FeatureExtractor,
    VGG16FeatureExtractor,
    SIFTFeatureExtractor,
)

from .extractor_predictor.feature_predictor import (
    FeaturePredictor,
    RandomForestPredictor,
    KNNPredictor,
    SVMPredictor,
)

from .models import ExtractorPredictor

from .evaluator import Evaluator, Accuracy, F1Score, Precision, Recall


feature_extractors = {
    "VGG16FeatureExtractor": VGG16FeatureExtractor,
    "SIFTFeatureExtractor": SIFTFeatureExtractor,
}

feature_predictors = {
    "RandomForestPredictor": RandomForestPredictor,
    "KNNPredictor": KNNPredictor,
    "SVMPredictor": SVMPredictor,
}


def build_feature_extractor(cfg) -> FeatureExtractor:
    """
    Builds a feature extractor based on the configuration provided.

    Args:
        cfg (dict): The configuration dictionary containing the model settings.

    Returns:
        FeatureExtractor: The instantiated feature extractor object.

    Raises:
        ValueError: If the specified feature extractor type is not supported.
    """
    feature_extractor_type = cfg["model"]["feature_extractor"]["type"]
    params = cfg["model"]["feature_extractor"]["params"]
    feature_extractor = None
    if feature_extractor_type == "VGG16FeatureExtractor":
        feature_extractor = VGG16FeatureExtractor(**params)
    elif feature_extractor_type == "SIFTFeatureExtractor":
        feature_extractor = SIFTFeatureExtractor(**params)
    else:
        raise ValueError(
            f"feature_extractor {cfg['feature_extractor']['feature_extractor']['type']} not supported"
        )
    return feature_extractor


def build_feature_predictor(cfg) -> FeaturePredictor:
    """
    Builds a feature predictor based on the configuration provided.

    Args:
        cfg (dict): The configuration dictionary containing the model parameters.

    Returns:
        FeaturePredictor: The built feature predictor object.

    Raises:
        ValueError: If the specified feature predictor type is not supported.
    """

    feature_predictor_type = cfg["model"]["feature_predictor"]["type"]
    params = cfg["model"]["feature_predictor"]["params"]
    feature_predictor = None
    if feature_predictor_type == "RandomForestPredictor":
        feature_predictor = RandomForestPredictor(**params)
    elif feature_predictor_type == "KNNPredictor":
        feature_predictor = KNNPredictor(**params)
    elif feature_predictor_type == "SVMPredictor":
        feature_predictor = SVMPredictor(**params)
    else:
        raise ValueError(
            f"feature_predictor {feature_predictor_type} not supported"
        )
    return feature_predictor


def build_model(cfg) -> ExtractorPredictor:
    """
    Builds a model based on the provided configuration.

    Args:
        cfg (dict): The configuration dictionary containing model parameters.

    Returns:
        model: The built model.

    Raises:
        ValueError: If the model type specified in the configuration is not supported.
    """
    if cfg["model"]["type"] != "extract_and_predict":
        raise ValueError(f"model {cfg['model']['type']} not supported")
    return ExtractorPredictor(
        build_feature_extractor(cfg), build_feature_predictor(cfg), **cfg["model"]["params"]
    )


def build_pretrained_model(cfg, weights_file) -> ExtractorPredictor:
    """
    Build a pretrained model based on the provided configuration and weights file.

    Args:
        cfg (dict): Configuration dictionary containing model specifications.
        weights_file (str): Path to the weights file.

    Returns:
        ExtractorPredictor: Pretrained model instance.

    Raises:
        ValueError: If the specified model type is not supported.
    """

    if cfg["model"]["type"] == "extract_and_predict":
        feature_extractor_type = cfg["model"]["feature_extractor"]["type"]
        assert (
            feature_extractor_type in feature_extractors.keys()
        ), "Feature extractor not supported"

        feature_predictor_type = cfg["model"]["feature_predictor"]["type"]
        assert (
            feature_predictor_type in feature_predictors.keys()
        ), "Feature predictor not supported"

        return ExtractorPredictor.load_model(
            weights_file,
            feature_extractors[feature_extractor_type],
            feature_predictors[feature_predictor_type],
        )
    else:
        raise ValueError(f"model {cfg['model']['type']} not supported")


def build_evaluator(cfg) -> Evaluator:
    """
    Build an evaluator object based on the provided configuration.

    Args:
        cfg (dict): The configuration dictionary containing the evaluator settings.

    Returns:
        Evaluator: An instance of the Evaluator class.

    Raises:
        ValueError: If the metric specified in the configuration is not supported.
    """
    metrics = []
    for metric in cfg["evaluator"]["metrics"]:
        name = metric["name"]
        params = metric["params"]
        if name == "Accuracy":
            metrics.append(Accuracy(**params))
        elif name == "F1Score":
            metrics.append(F1Score(**params))
        elif name == "Precision":
            metrics.append(Precision(**params))
        elif name == "Recall":
            metrics.append(Recall(**params))
        else:
            raise ValueError(f"metric {metric} not supported")
    return Evaluator(metrics)
