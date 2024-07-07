# from dataset import CustomDataset, DataLoader
from .transform import Compose, Normalize, Resize, GrayScale, RandomFlip, RandomRotation
from .dataset import CustomDataset, DataLoader


def build_transform(cfg) -> Compose:
    """
    Build a transformation pipeline based on the provided configuration.

    Args:
        cfg (dict): A dictionary containing the configuration for the transformations.

    Returns:
        Compose: A composition of transformations.

    Raises:
        ValueError: If a transform specified in the configuration is not supported.
    """
    transformations = []
    for transform_name, params in cfg.items():
        if transform_name == "Resize":
            transformations.append(Resize(**params))
        elif transform_name == "Normalize":
            transformations.append(Normalize(**params))
        elif transform_name == "GrayScale":
            transformations.append(GrayScale(**params))
        elif transform_name == "RandomFlip":
            transformations.append(RandomFlip(**params))
        elif transform_name == "RandomRotation":
            transformations.append(RandomRotation(**params))
        else:
            raise ValueError(f"Transform {transform_name} not supported")

    return Compose(transformations)


def build_dataset(cfg) -> CustomDataset:
    """
    Build a custom dataset based on the provided configuration.

    Args:
        cfg (dict): The configuration dictionary containing the following keys:
            - data_dir (str): The directory path where the data is stored.
            - transform (dict): The configuration for data transformation.

    Returns:
        CustomDataset: The custom dataset object.

    """
    return CustomDataset(
        data_dir=cfg["data_dir"], transform=build_transform(cfg["transform"])
    )


def build_dataloader(cfg) -> DataLoader:
    """
    Build and return a PyTorch DataLoader object.

    Args:
        cfg (dict): Configuration dictionary containing the following keys:
            - "dataset": The dataset to be used for building the DataLoader.
            - "batch_size": The batch size for the DataLoader.
            - "shuffle": Whether to shuffle the data in the DataLoader.

    Returns:
        DataLoader: The built DataLoader object.

    """
    return DataLoader(
        dataset=build_dataset(cfg["dataset"]),
        batch_size=cfg["batch_size"],
        shuffle=cfg["shuffle"],
    )
