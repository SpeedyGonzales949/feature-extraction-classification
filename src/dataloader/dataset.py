import numpy as np
import os
import cv2


class CustomDataset:
    """
    A custom dataset class for loading and processing image data.

    Attributes:
        data_dir (str): The directory path where the image data is stored.
        transform (callable): A function/transform that takes in an image and returns a transformed version.
        image_paths (list): A list of paths to the image files in the dataset.
    """

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = self._get_image_paths()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)
        target = image_path.split("_")[1]

        if self.transform:
            image = self.transform(image)

        return image, target

    def _get_image_paths(self):
        image_paths = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def __str__(self) :
        return (
            f"{self.__class__.__name__}(\n"
            + ",\n".join(
                [
                    f"  {key}={value}"
                    for key, value in self.__dict__.items()
                    if key not in ["image_paths"]
                ]
            )
            + "\n)"
        )


class DataLoader:
    """
    A class for loading data in batches from a dataset.

    Attributes:
        dataset (list): The dataset to load.
        batch_size (int, optional): The size of each batch. Defaults to 1.
        shuffle (bool, optional): Whether to shuffle the dataset before each epoch. Defaults to True.
        indices (list): A list of indices used to shuffle the dataset.
    """

    def __init__(self, dataset, batch_size=1, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return np.ceil(len(self.dataset) / self.batch_size).astype(int)

    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[i : i + self.batch_size]
            batch = [self.dataset[i] for i in batch_indices]
            images, targets = zip(*batch)
            yield np.array(images), np.array(targets)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            + ",\n".join(
                [
                    f"  {key}={value}"
                    for key, value in self.__dict__.items()
                    if key not in ["indices"]
                ]
            )
            + "\n)"
        )
