import torchvision, torch
import numpy as np

def image_subset(dataname, subclass_list, grayscale=False):
    if dataname == "mnist":
        transform = [torchvision.transforms.Resize(size=(28, 28),
                                                   interpolation=torchvision.transforms.InterpolationMode.BICUBIC)]
        if grayscale:
            transform += [torchvision.transforms.Grayscale()]
        transform += [torchvision.transforms.ToTensor()]
        mnist = torchvision.datasets.MNIST(
            root="mnist", train=True, download=True,
            transform=torchvision.transforms.Compose(transform))
        len_data = len(mnist)

        idx_per_digit = dict()
        for digit in subclass_list:
            idx = mnist.train_labels == digit
            idx_per_digit.update({digit: np.arange(len_data)[idx]})

        return mnist, idx_per_digit

    elif dataname == "cifar10":
        transform = [torchvision.transforms.Resize(size=(32, 32),
                                                   interpolation=torchvision.transforms.InterpolationMode.BICUBIC)]
        if grayscale:
            transform += [torchvision.transforms.Grayscale()]
        transform += [torchvision.transforms.ToTensor()]
        cifar = torchvision.datasets.CIFAR10(
            root="cifar10", train=True, download=True,
            transform=torchvision.transforms.Compose(transform))

        idx_per_image = dict()
        for c in subclass_list:
            idx = [t == c for t in cifar.targets]
            idx_per_image.update({c: np.where(idx)[0]})

        return cifar, idx_per_image

    elif dataname == "cifar100":
        transform = [torchvision.transforms.Resize(size=(32, 32),
                                                   interpolation=torchvision.transforms.InterpolationMode.BICUBIC)]
        if grayscale:
            transform += [torchvision.transforms.Grayscale()]
        transform += [torchvision.transforms.ToTensor()]
        cifar = torchvision.datasets.CIFAR100(
            root="cifar100", train=True, download=True,
            transform=torchvision.transforms.Compose(transform))

        idx_per_image = dict()
        for c in subclass_list:
            idx = [t == c for t in cifar.targets]
            idx_per_image.update({c: np.where(idx)[0]})

        return cifar, idx_per_image

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_size, image_dataset, idx_dict, image_patches, bsc_p, batch_size, return_label=False, nuisance=0):
        super().__init__()
        
        self.image_dataset = image_dataset
        self.image_size = img_size
        self.idx_dict = idx_dict
        self.image_patches = image_patches
        self.bsc_p = bsc_p
        self.batch_size = batch_size
        self.return_label = return_label
        self.original_image_size = self.image_dataset[0][0].shape
        self.nuisance = nuisance
        self.background, _ = image_subset("cifar10", np.arange(10), grayscale=False)
        self.to_image = torchvision.transforms.ToPILImage()

        self.resize = torchvision.transforms.Compose([
            torchvision.transforms.Resize(
                size=(self.image_size, self.image_size),
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC
            ),
            torchvision.transforms.ToTensor()
        ])
        
        self.bg_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(
                size=(self.image_size, self.image_size),
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC
            ),
            torchvision.transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.bsc_p) * self.batch_size

    def __getitem__(self, idx):
        torch.manual_seed(idx)

        batch_idx = idx // self.batch_size
        x1, x2, y1, y2 = self._generate_image(bsc_p=self.bsc_p[batch_idx])

        if (x1.size(0) == 1) and (self.image_patches[0] == 3):
            x1 = torch.tile(x1, (3, 1, 1))
            x2 = torch.tile(x2, (3, 1, 1))

        y1 = y1.numpy().reshape([-1]).tolist()
        y1 = [int(y) for y in y1]
        y2 = y2.numpy().reshape([-1]).tolist() 
        y2 = [int(y) for y in y2]

        label1 = int("".join(map(str, y1)), 2)
        label2 = int("".join(map(str, y2)), 2)

        if self.return_label:
            return x1, x2, label1, label2
        else:
            return x1, x2

    def _generate_image(self, bsc_p):
        idx = torch.bernoulli(torch.full(size=self.image_patches, fill_value=0.5))
        bsc_idx = torch.bernoulli(torch.full(size=self.image_patches, fill_value=bsc_p))
        idx_2 = torch.abs(idx - bsc_idx)

        # Pre-allocate tensors based on original image size and patches
        _, h, w = self.original_image_size
        channels, rows, cols = self.image_patches
        full_h = h * rows
        full_w = w * cols
        
        # Create empty tensors for the full images
        img1 = torch.empty((channels, full_h, full_w))
        img2 = torch.empty((channels, full_h, full_w))

        for p1 in range(channels):
            for p2 in range(rows):
                for p3 in range(cols):
                    class_1 = list(self.idx_dict.keys())[int(idx[p1, p2, p3])]
                    class_2 = list(self.idx_dict.keys())[int(idx_2[p1, p2, p3])]

                    img_idx_1 = np.random.choice(self.idx_dict[class_1])
                    img_idx_2 = np.random.choice(self.idx_dict[class_2])
                    while img_idx_2 == img_idx_1:
                        img_idx_2 = np.random.choice(self.idx_dict[class_2])

                    img1[p1, p2*h:(p2+1)*h, p3*w:(p3+1)*w] = self.image_dataset[img_idx_1][0]
                    img2[p1, p2*h:(p2+1)*h, p3*w:(p3+1)*w] = self.image_dataset[img_idx_2][0]

        im1 = self.resize(self.to_image(img1))
        im2 = self.resize(self.to_image(img2))

        if self.nuisance > 0:
            im1 = torch.tile(im1, (1, 3, 1, 1))
            im2 = torch.tile(im2, (1, 3, 1, 1))
            im1, im2 = self.apply_background(im1, im2)

        return im1, im2, idx, idx_2
    
    def apply_background(self, x1, x2):
        bg_idx = np.random.choice(len(self.background), 2)
        background_images = [
            self.bg_transform(self.background.data[bg_idx[0]])[None, :, :, :],
            self.bg_transform(self.background.data[bg_idx[1]])[None, :, :, :]
        ]

        z1 = torch.clip(x1 + background_images[0] * self.nuisance, 0, 1)
        z2 = torch.clip(x2 + background_images[1] * self.nuisance, 0, 1)
        
        if self.image_patches[0] == 1:
            return (z1.mean(1, keepdims=True), z2.mean(1, keepdims=True)) #(z1, z2)
        else:
            return (z1, z2)

    @property
    def dimensionality(self):
        return self.image_size * self.image_size * self.image_patches[0]