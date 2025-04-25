import torch, torchvision
    
class MixtureDataset(torch.utils.data.Dataset):
    def __init__(self, image_dataset, text_dataset, return_label=False):
        super().__init__()
        assert len(image_dataset) == len(text_dataset)

        self.image_dataset = image_dataset
        self.text_dataset = text_dataset
        self.return_label = return_label

        if self.text_dataset.dimensionality == 7680:
            self.image_dataset.resize = torchvision.transforms.Compose([
                torchvision.transforms.Resize(
                    size=(80, 96), 
                    interpolation=torchvision.transforms.InterpolationMode.BICUBIC
                ),
                torchvision.transforms.ToTensor()
            ])
        
    def __len__(self):
        return len(self.image_dataset)
    
    def __getitem__(self, idx):
        # seeding is done within the dataset __getitem__ to ensure consistency across views
        img1, _ = self.image_dataset[idx]
        _, text2 = self.text_dataset[idx]
        return img1, text2