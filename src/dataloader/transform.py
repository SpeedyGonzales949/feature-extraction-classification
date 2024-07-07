import numpy as np
import cv2



class Transform:
    """
    A base class for data transformations.

    This class provides a template for implementing custom data transformations.
    Subclasses should override the `__call__` method to define the specific transformation logic.
    """

    def __init__(self):
        pass

    def __call__(self, image):
        raise NotImplementedError()

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return f"{self.__class__.__name__}({', '.join([f'{key}={value}' for key, value in self.__dict__.items()])})"




class RandomFlip(Transform):
    """
    Randomly flips the input image horizontally or vertically with a given probability.

    Attributes:
        probability (float): The probability of flipping the image. Defaults to 0.5.

    Returns:
        numpy.ndarray: The flipped image.
    """

    def __init__(self, probability=0.5):
        super().__init__()
        self.probability = probability

    def __call__(self, image):
        if np.random.rand() > self.probability:
            image = cv2.flip(image, 1)  # Horizontal flip
        if np.random.rand() > self.probability:
            image = cv2.flip(image, 0)  # Vertical flip
        return image

class RandomRotation(Transform):
    """
    Randomly rotates an image by a specified angle or a random angle within a range.

    Attributes:
        probability (float): The probability of rotating the image. Defaults to 0.5.
        angle (float): The angle of rotation in degrees. If None, a random angle between -180 and 180 will be used.
    
    Returns:
        numpy.ndarray: The rotated image.

    """

    def __init__(self, probability=0.5, angle=None):
        super().__init__()
        self.probability = probability
        self.angle = angle

    def __call__(self, image):
        if np.random.rand() > self.probability:
            if self.angle is None:
                angle = np.random.uniform(-180, 180)
            else:
                angle = self.angle
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h))
        return image

class Normalize(Transform):
    """
    A transformation class to normalize an image.

    Attributes:
        mean (float or tuple): The mean value(s) for normalization.
        std (float or tuple): The standard deviation value(s) for normalization.
    """

    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def __call__(self, image):
        """
        Normalize the input image.

        Args:
            image (numpy.ndarray): The input image to be normalized.

        Returns:
            numpy.ndarray: The normalized image.
        """
        image = image.astype(np.float32) / 255
        image = (image - self.mean) / self.std
        return image

class Resize(Transform):
    """
    A transformation class to resize an image to a specified size.

    Attributes:
        size (tuple): The desired size of the image after resizing.

    """

    def __init__(self, size):
        super().__init__()
        self.size = size

    def __call__(self, image):
        """
        Resize the input image to the specified size.

        Args:
            image (numpy.ndarray): The input image to be resized.

        Returns:
            numpy.ndarray: The resized image.

        """
        image = cv2.resize(image, self.size)
        return image


class Compose(Transform):
    """
    A class that represents a composition of multiple transforms.

    Attributes:
        transforms (list): A list of transforms to be applied sequentially.

    """

    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def __call__(self, image):
        for transform in self.transforms:
            image = transform(image)
        return image
    
    def __repr__(self):
        return "Compose(" + ", ".join([str(transform) for transform in self.transforms]) + ")"
    
    def __str__(self):
        return "Compose(\n" + ",\n".join(["  " + str(transform) for transform in self.transforms]) + "\n)"
    
class GammaCorrection(Transform):
    """
    Apply gamma correction to an image.

    Attributes:
        gamma (float): The gamma value to be used for gamma correction. If None, a random value between 0.5 and 1.5 will be used.

    Returns:
        numpy.ndarray: The gamma-corrected image.
    """

    def __init__(self, gamma=None):
        super().__init__()
        self.gamma = gamma

    def __call__(self, image):
        if self.gamma is None:
            gamma = np.random.uniform(0.5, 1.5)
        else:
            gamma = self.gamma
        invGamma = 1.0 / gamma
        table = [((i / 255.0) ** invGamma) * 255 for i in range(256)]
        table = np.array(table).astype("uint8")
        return cv2.LUT(image, table)

class GrayScale(Transform):
    """
    Transform class to convert an image to grayscale.

    Returns:
        Grayscale image.

    """

    def __init__(self):
        super().__init__()

    def __call__(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

