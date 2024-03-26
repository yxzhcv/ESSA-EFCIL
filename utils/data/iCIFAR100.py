from torchvision.datasets import CIFAR100
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
import os
import sys

class iCIFAR100(CIFAR100):
    def __init__(self, root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False):
        super(iCIFAR100, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.TrainData = []
        self.TrainLabels = []
        self.TestData = []
        self.TestLabels = []
       
    def _map_new_class_index(self, y, order):
        """Transforms targets for new class order."""
        return list(map(lambda x: order.index(x), y))

    def concatenate(self, datas, labels):
        con_data = datas[0]
        con_label = labels[0]
        for i in range(1, len(datas)):
            con_data = np.concatenate((con_data, datas[i]), axis=0)
            con_label = np.concatenate((con_label, labels[i]), axis=0)
        return con_data, con_label

    def getTestData(self, classes):
        datas, labels = [], []
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        datas, labels = self.concatenate(datas, labels)
        self.TestData = datas if self.TestData == [] else np.concatenate((self.TestData, datas), axis=0)
        self.TestLabels = labels if self.TestLabels == [] else np.concatenate((self.TestLabels, labels), axis=0)
        print("the size of test set is %s" % (str(self.TestData.shape)))
        print("the size of test label is %s" % str(self.TestLabels.shape))

    def getTestData_up2now(self, classes):
        datas, labels = [], []
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        datas, labels = self.concatenate(datas, labels)
        self.TestData = datas
        self.TestLabels = labels
        print("the size of test set is %s" % (str(datas.shape)))
        print("the size of test label is %s" % str(labels.shape))

    def getTrainData(self, classes):
        datas, labels = [], []
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        self.TrainData, self.TrainLabels = self.concatenate(datas, labels)
        print("the size of train set is %s" % (str(self.TrainData.shape)))
        print("the size of train label is %s" % str(self.TrainLabels.shape))

    def getTrainData_sub(self, classes):
        datas, labels = [], []
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label][:20]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        self.TrainData, self.TrainLabels = self.concatenate(datas, labels)
        print("the size of train set is %s" % (str(self.TrainData.shape)))
        print("the size of train label is %s" % str(self.TrainLabels.shape))

    def getTrainItem(self, index):
        img, target = Image.fromarray(self.TrainData[index]), self.TrainLabels[index]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return index, img, target

    def getTestItem(self, index):
        img, target = Image.fromarray(self.TestData[index]), self.TestLabels[index]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return index, img, target

    def __getitem__(self, index):
        if self.TrainData != []:
            return self.getTrainItem(index)
        elif self.TestData != []:
            return self.getTestItem(index)

    def __len__(self):
        if self.TrainData != []:
            return len(self.TrainData)
        elif self.TestData != []:
            return len(self.TestData)

    def get_image_class(self, label):
        return self.data[np.array(self.targets) == label]

def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    """
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)








