import numpy as np
import torch.utils.data as data


class Sphere(data.Dataset):
    def __init__(
        self,
        R=3.0,
        N=2000,
        seed=None,
    ):
        """
        Args:
            R (float): radius of the sphere
            N (int): number of points
            seed (int, optional): seed for the random number generator
        """

        self.R = R
        self.N = N
        self.seed = seed
        self.connectivity = None

        self.generate_data()

    def generate_data(self):
        """Generate data."""

        np.random.seed(self.seed)
        self.data = np.random.randn(self.N, 3)
        self.data = self.data / np.linalg.norm(self.data, axis=1)[:, None]
        self.data = self.data * self.R

    def save_dataset(self, dataset_dir):
        np.save(dataset_dir / "sphere.npy", self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
