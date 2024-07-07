import cv2
import numpy as np
from sklearn.cluster import KMeans

class SIFT:
    """
    Class representing the Scale-Invariant Feature Transform (SIFT) algorithm.

    Attributes:
        sift: SIFT object from OpenCV library.
    """

    def __init__(self):
        self.sift = cv2.SIFT_create()

    def detectKeypoints(self, grayscale_image, mask=None) -> list:
        """
        Detects keypoints in the grayscale image.

        Args:
            grayscale_image: Grayscale image to detect keypoints from.
            mask: Optional mask specifying where to look for keypoints.

        Returns:
            List of keypoints detected in the image.
        """
        return self.sift.detect(grayscale_image, mask)

    def computeDescriptors(self, grayscale_image, kp) -> tuple:
        """
        Computes descriptors for the keypoints.

        Args:
            grayscale_image: Grayscale image to compute descriptors from.
            kp: List of keypoints.

        Returns:
            Tuple containing the computed descriptors and the list of keypoints.
        """
        return self.sift.compute(grayscale_image, kp)

    def drawKeypoints(
        self, grayscale_image, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    ) -> np.ndarray:
        """
        Draws keypoints on the image.

        Args:
            grayscale_image (numpy.ndarray): Grayscale image.
            kp (list): List of keypoints.
            img (numpy.ndarray): Image to draw keypoints on.
            flags (int, optional): Drawing flags. Possible values are:
                - cv2.DRAW_MATCHES_FLAGS_DEFAULT: Default flag.
                - cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG: Draw keypoints on top of the input image.
                - cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS: For each keypoint, the circle around the keypoint with keypoint size and orientation will be drawn.
                - cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS: Do not draw single points.

        Returns:
            numpy.ndarray: Image with keypoints drawn on it.
        """
        return cv2.drawKeypoints(grayscale_image, kp, img, flags=flags)
    
    def __str__(self):
        return f"{self.__class__.__name__}(\n" + ",\n".join([f"  {key}={value}" for key, value in self.__dict__.items()]) + "\n)"


class RootSIFT(SIFT):
    def computeDescriptors(self, grayscale_image, kp, eps=1e-7):
        kp, desc = self.sift.compute(grayscale_image, kp)
        if desc is None:
            return None
        desc /= desc.sum(axis=1, keepdims=True) + eps
        desc = np.sqrt(desc)
        return kp, desc


class PCASIFT(SIFT):
    """
    Class for performing PCA-SIFT feature extraction.

    Attributes:
        n_components (int): Number of principal components to keep.
        pca (cv2.PCACompute): PCA object for dimensionality reduction.

    Methods:
        computeDescriptors(grayscale_image, kp): Computes PCA-SIFT descriptors for the given grayscale image and keypoints.
    """

    def __init__(self, n_components=2):
        """
        Initializes a PCA_SIFT object.

        Args:
            n_components (int, optional): Number of principal components to keep. Defaults to 2.
        """
        super().__init__()
        self.n_components = n_components
        self.pca = cv2.PCACompute

    def computeDescriptors(self, grayscale_image, kp):
        """
        Computes PCA-SIFT descriptors for the given grayscale image and keypoints.

        Args:
            grayscale_image (numpy.ndarray): Grayscale image.
            kp (list): List of keypoints.

        Returns:
            tuple: Tuple containing the updated keypoints and computed descriptors.
        """
        kp, desc = self.sift.compute(grayscale_image, kp)
        if desc is None:
            return None
        _, desc = self.pca(data=desc, mean=None, maxComponents=self.n_components)
        return kp, desc


class BFMatcher:
    """
    Class representing a Brute-Force Matcher for feature matching.
    """

    def __init__(self):
        self.matcher = cv2.BFMatcher

    def matchL1Norm(self, desc1, desc2) -> list:
        """
        Matches feature descriptors using L1 norm.

        Args:
            desc1: The first set of feature descriptors.
            desc2: The second set of feature descriptors.

        Returns:
            A list of matches between the two sets of descriptors.
        """
        return self.matcher(cv2.NORM_L1, crossCheck=True).match(desc1, desc2)

    def matchL2Norm(self, desc1, desc2) -> list:
        """
        Matches feature descriptors using L2 norm.

        Args:
            desc1: The first set of feature descriptors.
            desc2: The second set of feature descriptors.

        Returns:
            A list of matches between the two sets of descriptors.
        """
        return self.matcher(cv2.NORM_L2, crossCheck=True).match(desc1, desc2)

    def knnMatchL1Norm(self, desc1, desc2, k=2) -> list:
        """
        Matches feature descriptors using L1 norm with k-nearest neighbors.

        Args:
            desc1: The first set of feature descriptors.
            desc2: The second set of feature descriptors.
            k: Number of nearest neighbors to return (default is 2).

        Returns:
            A list of k-nearest matches between the two sets of descriptors.
        """
        return self.matcher(cv2.NORM_L1).knnMatch(desc1, desc2, k=k)

    def knnMatchL2Norm(self, desc1, desc2, k=2) -> list:
        """
        Matches feature descriptors using L2 norm with k-nearest neighbors.

        Args:
            desc1: The first set of feature descriptors.
            desc2: The second set of feature descriptors.
            k: Number of nearest neighbors to return (default is 2).

        Returns:
            A list of k-nearest matches between the two sets of descriptors.
        """
        return self.matcher(cv2.NORM_L2).knnMatch(desc1, desc2, k=k)

    def drawMatches(self, img1, kp1, img2, kp2, matches, flags=2) -> np.ndarray:
        """
        Draws matches between two images.

        Args:
            img1: The first image.
            kp1: Keypoints detected in the first image.
            img2: The second image.
            kp2: Keypoints detected in the second image.
            matches: List of matches between the keypoints.
            flags: Drawing options (default is 2).
                - 0: Only draw keypoints without lines connecting them.
                - 1: Draw keypoints with lines connecting them.
                - 2: Draw keypoints with lines connecting them and show matches.

        Returns:
            An image with the matches drawn on it.
        """
        return cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=flags)
