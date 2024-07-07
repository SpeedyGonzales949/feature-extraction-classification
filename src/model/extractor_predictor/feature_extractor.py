import numpy as np
from sklearn.cluster import KMeans


from .sift import SIFT, RootSIFT, PCASIFT
from keras.applications.vgg16 import VGG16


class FeatureExtractor:
    """
    This class represents a feature extractor for image data. It provides an interface for fitting the model to the data and extracting features from the data.
    """

    def fit(self, images):
        raise NotImplementedError()

    def predict(self, images):
        raise NotImplementedError()

    def serialize(self):
        raise NotImplementedError()

    def load_model(cls, params):
        raise NotImplementedError()

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            + ",\n".join([f"  {key}={value}" for key, value in self.__dict__.items()])
            + "\n)"
        )


class VGG16FeatureExtractor(FeatureExtractor):
    """
    A class representing a feature extractor based on the VGG16 model.

    Attributes:
        model: The VGG16 model used for feature extraction.
        input_shape: The input shape of the images to be processed.
        weights: The weights used by the model.
        pooling: The pooling strategy used by the model.
    """

    def __init__(
        self, input_shape=(224, 224, 3), weights="imagenet", pooling="max"
    ) -> None:
        super(VGG16FeatureExtractor, self).__init__()

        # these are saved for serialization
        self.weights = weights
        self.pooling = pooling
        self.input_shape = input_shape

        self.model = VGG16(
            weights=self.weights,
            include_top=False,
            input_shape=self.input_shape,
            pooling=self.pooling,
        )
        self.model.evaluate_mode = True

    def fit(self, images=None):
        """
        Fits the feature extractor to the given images.
        Since we are only using the imagenet weights for feature extraction, this method does not do anything.
        However, it is included for compatibility with the ExtractorPredictor class.
        In future, we can add code here to fine-tune the model on the given images.
        """
        pass

    def predict(self, images):
        """
        Predicts the features of the given images using the fitted model.

        Args:
            images (numpy.ndarray): An array of images to extract features from.

        Returns:
            numpy.ndarray: An array of extracted features.

        Raises:
            Exception: If the model has not been fitted.
        """
        if self.model is None:
            raise Exception("Model has not been fitted")
        features = self.model.predict(images, verbose=0)
        return features

    def serialize(self):
        """
        Serializes the feature extractor model.

        Returns:
            dict: A dictionary containing the serialized information of the feature extractor.
                - "weights": The weights used by the model (e.g., "imagenet").
                - "input_shape": The shape of the model's input.
                - "pooling": The pooling method used by the model (e.g., "max").
        """
        return {
            "weights": self.weights,
            "input_shape": self.input_shape,
            "pooling": self.pooling,
        }

    @classmethod
    def load_model(cls, params):
        assert "weights" in params
        assert "input_shape" in params
        assert "pooling" in params

        return cls(**params)


class SIFTFeatureExtractor(FeatureExtractor):
    """
    SIFTFeatureExtractor is a class that extracts SIFT features from images and computes visual words.

    Attributes:
        visual_words (ndarray): The computed visual words.
        n_visual_words (int): The number of visual words to compute.
        detector (SIFT): The SIFT detector object. Defaults to SIFT. Other options are RootSIFT and PCASIFT.

    Raises:
        NotImplementedError: If the specified detector_name is not implemented.

    """

    def __init__(self, detector_name="SIFT", n_visual_words=100) -> None:
        super().__init__()

        self.visual_words = None
        self.n_visual_words = n_visual_words

        self.detector: SIFT = None

        sift_detectors = {"SIFT": SIFT, "RootSIFT": RootSIFT, "PCASIFT": PCASIFT}

        if detector_name not in sift_detectors:
            raise NotImplementedError(f"{detector_name} not implemented!!")

        self.detector = sift_detectors[detector_name]()

    def fit(self, images):
        """
        Fits the SIFTFeatureExtractor by computing the visual words from the given images.

        Args:
            images (List[ndarray]): The list of images to fit the extractor on.

        """

        self.visual_words = self.compute_visual_words(
            descriptors=np.concatenate(self.transform(images), axis=0)
        )

    def predict(self, images):
        """
        Predicts the features of the given images using the computed visual words.

        Args:
            images (List[ndarray]): The list of images to predict the features for.

        Returns:
            List[ndarray]: The predicted features for each image.

        Raises:
            Exception: If the model has not been fitted.

        """

        if self.visual_words is None:
            raise Exception("Model has not been fitted")

        descriptors = self.transform(images)
        features = []
        for desc in descriptors:
            histogram = np.zeros(len(self.visual_words))
            for d in desc:
                distances = np.linalg.norm(self.visual_words - d, axis=1)
                histogram[np.argmin(distances)] += 1
            features.append(histogram)

        return features

    def compute_visual_words(self, descriptors):
        """
        Computes the visual words using K-means clustering.

        Args:
            descriptors (ndarray): The descriptors of the images.

        Returns:
            ndarray: The computed visual words.

        """

        kmeans = KMeans(n_clusters=self.n_visual_words, n_init=10)
        kmeans.fit(descriptors)
        return kmeans.cluster_centers_

    def transform(self, images):
        """
        Transforms the images into descriptors using the type of SIFT detector.

        Args:
            images (List[ndarray]): The list of images to transform.

        Returns:
            List[ndarray]: The descriptors of the images.

        """

        descriptors = []
        for img in images:
            kp = self.detector.detectKeypoints(img)
            kp, desc = self.detector.computeDescriptors(img, kp)
            descriptors.append(desc)
        return descriptors

    def serialize(self):
        """
        Serializes the SIFTFeatureExtractor object.

        Returns:
            dict: A dictionary containing the serialized representation of the object.

        """
        return {
            "detector_name": self.detector.__class__.__name__,
            "n_visual_words": self.n_visual_words,
            "visual_words": self.visual_words,
        }

    @classmethod
    def load_model(cls, params):
        """
        Loads a serialized SIFTFeatureExtractor object.

        Args:
            params (dict): A dictionary containing the serialized representation of the object.

        Returns:
            SIFTFeatureExtractor: The loaded SIFTFeatureExtractor object.

        """
        assert "detector_name" in params
        assert "n_visual_words" in params
        assert "visual_words" in params

        model = cls(
            detector_name=params["detector_name"],
            n_visual_words=params["n_visual_words"],
        )
        model.visual_words = params["visual_words"]
        return model
