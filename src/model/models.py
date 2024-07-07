from model.extractor_predictor.feature_extractor import (
    FeatureExtractor,
)

from model.extractor_predictor.feature_predictor import (
    FeaturePredictor,
)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import time
from utils.logger import Logger
import joblib


class Model:
    """
    A base class for machine learning models.
    """

    def train(self, dataloader):
        raise NotImplementedError()

    def evaluate(self, dataloader):
        raise NotImplementedError()

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            + ",\n".join([f"  {key}={value}" for key, value in self.__dict__.items()])
            + "\n)"
        )

    @classmethod
    def load_model(cls, path):
        """
        Load a saved model from the specified path.

        Args:
        - path (str): The path to the saved model file.

        Returns:
        - model (Model): The loaded model.

        Raises:
        - AssertionError: If the loaded model is not an instance of the current class.
        """
        model = joblib.load(path)
        assert isinstance(model, cls)
        return model

    def save_model(self, path):
        """
        Save the model to the specified path.

        Args:
        - path (str): The path to save the model.

        Returns:
        - None
        """
        joblib.dump(self, path)


class ExtractorPredictor(Model):
    """
    A class representing an ExtractorPredictor model.
    """

    def __init__(
        self,
        extractor: FeatureExtractor,
        predictor: FeaturePredictor,
        class_encoder=None,
        classes=None,
    ):
        """
        Initializes an instance of the ExtractorPredictor class.

        Args:
            extractor (FeatureExtractor): The feature extractor object.
            predictor (FeaturePredictor): The feature predictor object.
            class_encoder (str, optional): The type of class encoder to use. Defaults to "label_encoder".
            classes (array-like, optional): The classes to be encoded. Defaults to None.
        """
        super(ExtractorPredictor, self).__init__()
        self.extractor = extractor
        self.predictor = predictor

        assert class_encoder in [
            "label_encoder",
            "one_hot_encoder",
            None,
        ], "Invalid class encoder type."
        if class_encoder == "label_encoder":
            self.class_encoder = LabelEncoder()
        elif class_encoder == "one_hot_encoder":
            self.class_encoder = OneHotEncoder()
        else:
            self.class_encoder = None

        if class_encoder is not None:
            assert classes is not None, "Classes must be provided for class encoding."
            self.class_encoder.classes_ = np.array(classes)

    def train(self, dataloader, logger):
        """
        Trains the model using the provided dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader containing the training data.
            logger (logging.Logger): The logger object for logging training progress.

        """
        steps = len(dataloader)
        for idx, (images, targets) in enumerate(dataloader):

            extractor_start_time = time.time()
            self.extractor.fit(images)
            features = self.extractor.predict(images)
            extractor_end_time = time.time()

            if self.class_encoder:
                targets = self.class_encoder.transform(targets)

            predictor_start_time = time.time()
            self.predictor.fit(features, targets)
            predictor_end_time = time.time()

            logger.info(
                f"[{idx+1}/{steps}]  Extractor-Time:{extractor_end_time - extractor_start_time:.4f}s, Predictor-Time:{predictor_end_time - predictor_start_time:.4f}s, Total:{predictor_end_time - extractor_start_time:.4f}"
            )

    def predict(self, dataloader, logger, return_features=False):
        """
        Makes predictions using the provided dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader containing the data for prediction.
            logger (logging.Logger): The logger object for logging information.
            return_features (bool, optional): Whether to return the extracted features along with predictions.
                Defaults to False.

        Returns:
            predictions (numpy.ndarray): An array of predicted labels.
            true_labels (numpy.ndarray): An array of true labels.
            features_list (numpy.ndarray, optional): An array of extracted features if `return_features` is True.

        """
        predictions = []
        true_labels = []
        steps = len(dataloader)

        if return_features:
            features_list = []
        for idx, (images, targets) in enumerate(dataloader):
            extractor_start_time = time.time()
            features = self.extractor.predict(images)
            extractor_end_time = time.time()

            if return_features:
                features_list.extend(features)

            predictor_start_time = time.time()
            pred = self.predictor.predict(features)
            predictor_end_time = time.time()

            if self.class_encoder:
                pred = self.class_encoder.inverse_transform(pred)
            predictions.extend(pred)
            true_labels.extend(targets)
            logger.info(
                f"[{idx+1}/{steps}] Extractor-Time:{extractor_end_time - extractor_start_time:.4f}s, Predictor-Time:{predictor_end_time - predictor_start_time:.4f}s, Total:{predictor_end_time - extractor_start_time:.4f}s"
            )

        if return_features:
            return np.array(predictions), np.array(true_labels), np.array(features_list)
        else:
            return np.array(predictions), np.array(true_labels)

    def save_model(self, path):
        """
        Save the model to the specified path.

        Args:
        - path (str): The path to save the model.

        Returns:
        - None
        """
        joblib.dump(
            {
                "extractor": self.extractor.serialize(),
                "predictor": self.predictor.serialize(),
                "class_encoder": self.class_encoder,
            },
            path,
        )

    @classmethod
    def load_model(cls, path, extractor: FeatureExtractor, predictor: FeaturePredictor):
        """
        Load a pre-trained model from the specified path.

        Args:
            path (str): The path to the saved model.
            extractor (FeatureExtractor): The feature extractor used by the model and present in the config.
            predictor (FeaturePredictor): The feature predictor used by the model and present in the config.

        Returns:
            cls: An instance of the model class with the loaded feature extractor and predictor.
        """
        params = joblib.load(path)
        assert "extractor" in params, "Extractor not found in model file."
        assert "predictor" in params, "Predictor not found in model file."
        assert "class_encoder" in params, "Class encoder not found in model file."

        extractor = extractor.load_model(params["extractor"])
        predictor = predictor.load_model(params["predictor"])

        model = cls(extractor, predictor)
        model.class_encoder = params["class_encoder"]
        return model
