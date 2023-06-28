import torch
from torchvision import datasets
from utils.utils_algo import create_partial_label
import torch.nn.functional as F
from PIL import Image


class PMNIST(datasets.MNIST):
    def __init__(self, root, train, transform=None, target_transform=None, download=False, partial_type="binomial",
                 partial_rate=0.1):
        super(PMNIST, self).__init__(root, train, transform, target_transform, download)
        one_hot_targets = F.one_hot(self.targets, len(self.classes))
        if self.train and partial_rate != 0.:
            self.partial_targets, self.Avg_Cls = create_partial_label(one_hot_targets, self.targets,
                                                                      partial_type, partial_rate)
        else:
            self.partial_targets = one_hot_targets

    def __getitem__(self, idx):
        img, target, partial_target = self.data[idx], self.targets[idx], self.partial_targets[idx]

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        img = img.reshape(28 * 28)
        if self.train:
            return img, target, partial_target, idx
        else:
            return img, target, partial_target, idx
