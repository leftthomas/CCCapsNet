import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


def get_iterator(mode, batch_size=64):
    data = MNIST(root='data/', train=mode, transform=transforms.ToTensor(), download=True)
    return DataLoader(dataset=data, batch_size=batch_size, shuffle=mode, num_workers=4)
