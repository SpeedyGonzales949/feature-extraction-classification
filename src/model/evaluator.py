import numpy as np


class Metric:
    """
    A base class for defining evaluation metrics.
    """

    def compute(self, y_true, y_pred):
        """
        Abstract method to compute the metric value.

        Args:
            y_true: The true labels.
            y_pred: The predicted labels.

        Returns:
            The computed metric value.
        """
        raise NotImplementedError()

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            + ", ".join([f"{key}={value}" for key, value in self.__dict__.items()])
            + ")"
        )

    def __repr__(self) -> str:
        return self.__str__()


class Precision(Metric):
    """
    A class representing the precision metric.

    This class provides methods to compute precision for different reduce methods,
    such as micro, macro, and weighted.

    Args:
        reduce (str, optional): The reduce method to use. Defaults to None.
    """

    def __init__(self, reduce=None):
        """
        Initialize the Precision metric.

        Args:
            reduce (str, optional): The reduce method to use. Defaults to None.
        """
        self.reduce = reduce

    def compute(self, y_true, y_pred):
        """
        Computes the precision metric based on the specified reduce method.

        Args:
            y_true (array-like): The true labels.
            y_pred (array-like): The predicted labels.

        Returns:
            float or array-like: The computed precision metric.

        Raises:
            ValueError: If an unknown reduce method is specified.
        """
        if self.reduce is None:
            return self.precision(y_true, y_pred)
        elif self.reduce == "micro":
            return self.micro_precision(y_true, y_pred)
        elif self.reduce == "macro":
            return self.macro_precision(y_true, y_pred)
        elif self.reduce == "weighted":
            return self.weighted_precision(y_true, y_pred)
        else:
            raise ValueError(f"Unknown reduce method: {self.reduce}")

    def precision(self, y_true, y_pred):
        """
        Computes the precision metric for each class.

        Args:
            y_true (array-like): The true labels.
            y_pred (array-like): The predicted labels.

        Returns:
            array-like: The precision for each class.
        """
        classes = np.unique(y_true)
        precisions = np.zeros_like(classes, dtype=float)

        for idx, cls in enumerate(classes):
            true_positives = np.sum((y_true == cls) & (y_pred == cls))
            false_positives = np.sum((y_true != cls) & (y_pred == cls))
            precision = (
                true_positives / (true_positives + false_positives)
                if (true_positives + false_positives) != 0
                else 0
            )
            precisions[idx] = precision

        return precisions

    def micro_precision(self, y_true, y_pred):
        """
        Computes the micro-averaged precision metric.

        Args:
            y_true (array-like): The true labels.
            y_pred (array-like): The predicted labels.

        Returns:
            float: The micro-averaged precision.
        """
        true_positives = np.sum(y_true == y_pred)
        false_positives = np.sum(y_true != y_pred)
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) != 0
            else 0
        )
        return precision

    def macro_precision(self, y_true, y_pred):
        """
        Computes the macro-averaged precision metric.

        Args:
            y_true (array-like): The true labels.
            y_pred (array-like): The predicted labels.

        Returns:
            float: The macro-averaged precision.
        """
        return np.mean(self.precision(y_true, y_pred))

    def weighted_precision(self, y_true, y_pred):
        """
        Computes the weighted precision metric.

        Args:
            y_true (array-like): The true labels.
            y_pred (array-like): The predicted labels.

        Returns:
            float: The weighted precision.
        """
        return np.average(
            self.precision(y_true, y_pred),
            weights=np.unique(y_true, return_counts=True)[1],
        )


class Recall(Metric):
    """
    A Class to compute recall metric for multi-class classification.

    Attributes:
        reduce (str, optional): The reduce method to use. Defaults to None.
    """

    def __init__(self, reduce=None):
        """
        Initialize the Recall metric.

        Args:
            reduce (str, optional): The reduce method to use. Defaults to None.
        """
        self.reduce = reduce

    def compute(self, y_true, y_pred):
        """
        Compute the recall metric.

        Parameters:
            y_true (array-like): True labels.
            y_pred (array-like): Predicted labels.
            reduce (str, optional): The reduce method to use. Defaults to None.

        Returns:
            recall (float or array): Computed recall metric.

        Raises:
            ValueError: If an unknown reduce method is provided.
        """
        if self.reduce is None:
            return self.recall(y_true, y_pred)
        elif self.reduce == "micro":
            return self.micro_recall(y_true, y_pred)
        elif self.reduce == "macro":
            return self.macro_recall(y_true, y_pred)
        elif self.reduce == "weighted":
            return self.weighted_recall(y_true, y_pred)
        else:
            raise ValueError(f"Unknown reduce method: {self.reduce}")

    def recall(self, y_true, y_pred):
        """
        Compute the recall metric for each class.

        Parameters:
            y_true (array-like): True labels.
            y_pred (array-like): Predicted labels.

        Returns:
            recalls (array): Computed recall metric for each class.
        """
        classes = np.unique(y_true)
        recalls = np.zeros_like(classes, dtype=float)

        for idx, cls in enumerate(classes):
            true_positives = np.sum((y_true == cls) & (y_pred == cls))
            false_negatives = np.sum((y_true == cls) & (y_pred != cls))
            recall = (
                true_positives / (true_positives + false_negatives)
                if (true_positives + false_negatives) != 0
                else 0
            )
            recalls[idx] = recall

        return recalls

    def micro_recall(self, y_true, y_pred):
        """
        Compute the micro-averaged recall metric.

        Parameters:
            y_true (array-like): True labels.
            y_pred (array-like): Predicted labels.

        Returns:
            recall (float): Computed micro-averaged recall metric.
        """
        true_positives = np.sum(y_true == y_pred)
        false_negatives = np.sum(y_true != y_pred)

        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) != 0
            else 0
        )
        return recall

    def macro_recall(self, y_true, y_pred):
        """
        Compute the macro-averaged recall metric.

        Parameters:
            y_true (array-like): True labels.
            y_pred (array-like): Predicted labels.

        Returns:
            recall (float): Computed macro-averaged recall metric.
        """
        return np.mean(self.recall(y_true, y_pred))

    def weighted_recall(self, y_true, y_pred):
        """
        Compute the weighted recall metric.

        Parameters:
            y_true (array-like): True labels.
            y_pred (array-like): Predicted labels.

        Returns:
            recall (float): Computed weighted recall metric.
        """
        return np.average(
            self.recall(y_true, y_pred),
            weights=np.unique(y_true, return_counts=True)[1],
        )


class Accuracy(Metric):
    """
    A class representing accuracy metrics for classification models.

    Methods:
        - compute(y_true, y_pred, reduce=None): Compute the accuracy metric based on the specified reduce method.
        - accuracy(y_true, y_pred): Compute the accuracy for each class.
        - micro_accuracy(y_true, y_pred): Compute the micro-averaged accuracy.
        - macro_accuracy(y_true, y_pred): Compute the macro-averaged accuracy.
        - weighted_accuracy(y_true, y_pred): Compute the weighted accuracy.
    """

    def __init__(self, reduce=None):
        """
        Initialize the Accuracy metric.

        Args:
            reduce (str, optional): The reduce method to use. Defaults to None.
        """
        self.reduce = reduce

    def compute(self, y_true, y_pred):
        """
        Parameters:
        - y_true (array-like): The true labels.
        - y_pred (array-like): The predicted labels.

        Returns:
        - accuracy (float or array): Computed accuracy metric.

        Raises:
        - ValueError: If an unknown reduce method is specified.
        """
        if self.reduce is None:
            return self.accuracy(y_true, y_pred)
        elif self.reduce == "micro":
            return self.micro_accuracy(y_true, y_pred)
        elif self.reduce == "macro":
            return self.macro_accuracy(y_true, y_pred)
        elif self.reduce == "weighted":
            return self.weighted_accuracy(y_true, y_pred)
        else:
            raise ValueError(f"Unknown reduce method: {self.reduce}")

    def accuracy(self, y_true, y_pred):
        classes = np.unique(y_true)
        accuracies = []

        for cls in classes:
            correct = np.sum((y_true == cls) & (y_pred == cls))
            total = np.sum(y_true == cls)
            accuracy = correct / total if total != 0 else 0
            accuracies.append(accuracy)

        return np.array(accuracies)

    def micro_accuracy(self, y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true) if len(y_true) != 0 else 0

    def macro_accuracy(self, y_true, y_pred):
        return np.mean(self.accuracy(y_true, y_pred))

    def weighted_accuracy(self, y_true, y_pred):
        return np.average(
            self.accuracy(y_true, y_pred),
            weights=np.unique(y_true, return_counts=True)[1],
        )


class F1Score(Metric):
    """
    Class representing the F1 Score metric.

    Methods:
        - compute(y_true, y_pred, reduce=None): Computes the F1 Score based on the specified reduction method.
        - f1score(y_true, y_pred): Computes the F1 Score for each class.
        - micro_f1score(y_true, y_pred): Computes the micro-averaged F1 Score.
        - macro_f1score(y_true, y_pred): Computes the macro-averaged F1 Score.
        - weighted_f1score(y_true, y_pred): Computes the weighted F1 Score.

    """

    def __init__(self, reduce=None):
        """
        Initializes the F1 Score metric.

        Args:
            reduce (str, optional): The reduce method to use. Defaults to None.

        """
        self.reduce = reduce

    def compute(self, y_true, y_pred):
        """
        Computes the F1 Score based on the specified reduction method.

        Args:
            y_true (array-like): The true labels.
            y_pred (array-like): The predicted labels.

        Returns:
            float or array-like: The computed F1 Score(s).

        Raises:
            ValueError: If an unknown reduce method is specified.

        """
        if self.reduce is None:
            return self.f1score(y_true, y_pred)
        elif self.reduce == "micro":
            return self.micro_f1score(y_true, y_pred)
        elif self.reduce == "macro":
            return self.macro_f1score(y_true, y_pred)
        elif self.reduce == "weighted":
            return self.weighted_f1score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown reduce method: {self.reduce}")

    def f1score(self, y_true, y_pred):
        precisions = Precision().compute(y_true, y_pred)
        recalls = Recall().compute(y_true, y_pred)

        assert len(precisions) == len(recalls)

        f1_scores = np.zeros_like(precisions)

        for idx in range(len(precisions)):
            precision = precisions[idx]
            recall = recalls[idx]
            f1_score = (
                2 * precision * recall / (precision + recall)
                if precision + recall != 0
                else 0
            )
            f1_scores[idx] = f1_score

        return f1_scores

    def micro_f1score(self, y_true, y_pred):
        precision = Precision().micro_precision(y_true, y_pred)
        recall = Recall().micro_recall(y_true, y_pred)
        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall != 0
            else 0
        )
        return f1_score

    def macro_f1score(self, y_true, y_pred):
        return np.mean(self.f1score(y_true, y_pred))

    def weighted_f1score(self, y_true, y_pred):
        return np.average(
            self.f1score(y_true, y_pred),
            weights=np.unique(y_true, return_counts=True)[1],
        )


class Evaluator:
    """
    Class to evaluate model performance using a list of metrics.

    Attributes:
        metrics (list[Metric]): A list of Metric objects representing the evaluation metrics.
    """

    def __init__(self, metrics: list[Metric]):
        self.metrics = metrics

    def evaluate(self, y_true, y_pred):
        return [
            (metric.__class__.__name__, metric.compute(y_true, y_pred))
            for metric in self.metrics
        ]

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            + ",\n".join([f"  {key}={value}" for key, value in self.__dict__.items()])
            + "\n)"
        )
