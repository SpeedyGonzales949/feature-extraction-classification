from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib


class FeaturePredictor:
    """
    A base class that represents a feature predictor. Usually, it represent a wrapper over an sklearn model. Sklearn models can be serialized and deserialized using joblib.
    """

    def __init__(self):
        self.model = None

    def fit(self, feature, target):
        raise NotImplementedError()

    def predict(self, feature):
        raise NotImplementedError()

    @classmethod
    def load_model(cls, params):
        """
        Load a trained model from the given parameters.

        Args:
            params (dict): A dictionary containing the model to be loaded.

        Returns:
            predictor (cls): The loaded model.

        Raises:
            AssertionError: If the loaded model is not an instance of cls.
        """
        predictor = params["model"]
        assert isinstance(
            predictor, cls
        ), f"Loaded model {predictor.__name__} is not an instance of {cls}"
        return predictor

    def serialize(self):
        """
        Serializes the feature predictor object.

        Returns:
            dict: A dictionary containing the serialized feature predictor object.
        """
        return {"model": self}

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            + ",\n".join([f"  {key}={value}" for key, value in self.__dict__.items()])
            + "\n)"
        )


class RandomForestPredictor(FeaturePredictor):
    """
    A predictor that uses a Random Forest classifier for feature prediction.

    Parameters:
    - n_estimators (int): The number of trees in the forest. Default is 100.
    - max_depth (int): The maximum depth of the tree. Default is None.
    - random_state (int): The seed used by the random number generator. Default is 42.
    """

    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )

    def fit(self, feature, target):
        """
        Fit the Random Forest classifier to the given feature and target data.

        Parameters:
        - feature (array-like): The input feature data.
        - target (array-like): The target data.
        """
        self.model.fit(feature, target)

    def predict(self, feature):
        """
        Predict the target labels for the given feature data.

        Parameters:
        - feature (array-like): The input feature data.

        Returns:
        - predictions (array-like): The predicted target labels.
        """
        return self.model.predict(feature)


class KNNPredictor(FeaturePredictor):
    """
    KNNPredictor is a class that implements a k-nearest neighbors (KNN) predictor for feature classification.

    Parameters:
    - n_neighbors (int): The number of neighbors to consider for classification. Default is 5.
    - weights (str): The weight function used in prediction. Possible values are 'uniform' or 'distance'. Default is 'uniform'.
    - algorithm (str): The algorithm used to compute the nearest neighbors. Possible values are 'auto', 'ball_tree', 'kd_tree', or 'brute'. Default is 'auto'.
    - metric (str): The distance metric used for computing the neighbors. Possible values are 'euclidean', 'manhattan', 'chebyshev', or 'minkowski'. Default is 'euclidean'.
    """

    def __init__(
        self, n_neighbors=5, weights="uniform", algorithm="auto", metric="euclidean"
    ):
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, metric=metric
        )

    def fit(self, feature, target):
        """
        Fit the KNN model to the given feature and target data.

        Parameters:
        - feature (array-like): The input feature data.
        - target (array-like): The target data for classification.
        """
        self.model.fit(feature, target)

    def predict(self, feature):
        """
        Predict the class labels for the given feature data.

        Parameters:
        - feature (array-like): The input feature data.

        Returns:
        - predictions (array-like): The predicted class labels.
        """
        return self.model.predict(feature)


class SVMPredictor(FeaturePredictor):
    """
    Support Vector Machine (SVM) predictor for feature classification.

    Parameters:
    - kernel (str): Specifies the kernel type to be used in the SVM algorithm. Default is 'rbf'.
    - C (float): Regularization parameter. Default is 1.0.
    - gamma (str): Kernel coefficient for 'rbf', 'poly', and 'sigmoid'. Default is 'scale'.

    Attributes:
    - model (SVC): The SVM model used for prediction.
    """

    def __init__(self, kernel="rbf", C=1.0, gamma="scale"):
        self.model = SVC(kernel=kernel, C=C, gamma=gamma)

    def fit(self, feature, target):
        """
        Fit the SVM model to the given feature and target data.

        Parameters:
        - feature (array-like): The input feature data.
        - target (array-like): The target labels corresponding to the feature data.

        """
        self.model.fit(feature, target)

    def predict(self, feature):
        """
        Predict the target labels for the given feature data.

        Parameters:
        - feature (array-like): The input feature data.

        Returns:
        - predictions (array-like): The predicted target labels.

        """
        return self.model.predict(feature)
