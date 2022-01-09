import numpy as np
import math


class Dataset:
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=False):
        if not isinstance(dataset, Dataset):
            raise TypeError(f'type(dataset) must be Dataset')
        
        self.dataset = dataset
        self.total_sample = len(dataset)
        self.sample_yield = 0
        self.batch_szie = batch_size
        self.shuffle = shuffle

        self.indices = np.arange(0, self.total_sample, 1)
    
    def __len__(self):
        return math.ceil(self.total_sample / self.batch_szie)
    
    def __iter__(self):
        self.sample_yield = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self
    
    def __next__(self):
        if self.sample_yield < self.total_sample:
            start_index = self.sample_yield
            end_index = min(self.sample_yield + self.batch_szie, self.total_sample)
            self.sample_yield = end_index

            temp1 = []   # List[Tuple(...)]
            for i in range(start_index, end_index):
                temp1.append(self.dataset[self.indices[i]])
            temp2 = list(zip(*temp1))
            
            res = []
            for i in range(len(temp2)):
                res.append(np.array(temp2[i]))

            return res   # type(res): List[ndarray]
        else:
            raise StopIteration
