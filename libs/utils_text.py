import os, glob, torch
import numpy as np


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 ds,
                 bsc_p,
                 batch_size,
                 dataname="imdb.bert-imdb-finetuned",
                 root="dataset",
                 get_label=False,
                 n_sample=None):
        super().__init__()

        self.classes = [0, 1]

        self.data = []
        self.counts = dict()
        self.label = []
        self.ds = ds
        self.bsc_p = bsc_p
        self.batch_size = batch_size
        self.get_label = get_label
        self.n_sample = n_sample

        for idx, subclass in enumerate(self.classes):
            file_list = glob.glob(os.path.join(root, dataname, str(subclass), '*.npy'))
            file_list.sort()

            if (not self.n_sample in [None, "None"]):
                if self.n_sample > 0:
                    file_list = file_list[:self.n_sample]

            self.counts[idx] = len(self.data)
            self.data += [(filename, idx) for filename in file_list]
            self.label += [idx] * len(file_list)
        self.counts[len(self.classes)] = len(self.data)

    def __len__(self):
        return len(self.bsc_p) * self.batch_size

    def _load_file(self, index):
        filename, target_idx = self.data[index]
        data = np.load(filename)

        return data, target_idx

    def _generate_text(self, idx_dict, bsc_p=0):
        idx = torch.bernoulli(torch.full(size=(self.ds,), fill_value=0.5))
        bsc_idx = torch.bernoulli(torch.full(size=(self.ds,), fill_value=bsc_p))
        idx_2 = torch.abs(idx - bsc_idx)

        x1, x2 = [], []
        for t in range(self.ds):
            class1 = int(idx[t])
            class2 = int(idx_2[t])

            text1_idx = np.random.choice(np.arange(idx_dict[class1], idx_dict[class1+1]))
            text2_idx = np.random.choice(np.arange(idx_dict[class2], idx_dict[class2+1]))

            while text2_idx == text1_idx:
                text2_idx = np.random.choice(np.arange(idx_dict[class2], idx_dict[class2+1]))

            x1.append(torch.Tensor(self._load_file(text1_idx)[0]))
            x2.append(torch.Tensor(self._load_file(text2_idx)[0]))

        return torch.cat(x1, dim=-1), torch.cat(x2, dim=-1), idx, idx_2

    def __getitem__(self, idx):
        torch.manual_seed(idx)

        if self.n_sample is None:
            idx_dict = self.counts
        else:
            idx_dict = {0: 0, 1: self.ds, 2: self.counts[1]+self.ds}

        batch_idx = idx // self.batch_size
        x1, x2, y1, y2 = self._generate_text(idx_dict, bsc_p=self.bsc_p[batch_idx])

        y1 = y1.numpy().reshape([-1]).tolist()
        y1 = [int(y) for y in y1]

        y2 = y2.numpy().reshape([-1]).tolist()
        y2 = [int(y) for y in y2]

        label1 = int("".join(map(str, y1)), 2)
        label2 = int("".join(map(str, y2)), 2)

        x1 = torch.Tensor(x1).view(1, -1)
        x2 = torch.Tensor(x2).view(1, -1)
        if self.get_label:
            return x1, x2, label1, label2
        else:
            return x1, x2
        
    @property
    def dimensionality(self):
        return self._load_file(0)[0].shape[1] * self.ds