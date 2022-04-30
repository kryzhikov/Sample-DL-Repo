import numpy as np
import pandas as pd


class FlattenDataSet:
    def __init__(self, df):
        self.image_size = 28
        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __convert_array(self, x):
        answer = np.zeros((1, 10))
        answer[0][x] = 1
        return answer

    def __getitem__(self, item):
        label = self.df['label'].iloc[item]
        img = np.zeros((1, self.image_size, self.image_size))
        for i in range(self.image_size):
            for j in range(self.image_size):
                num_pixel = f'pixel{i * self.image_size + j}'
                img[0][i][j] = self.df[num_pixel].iloc[item]
        return img, self.__convert_array(label)


class DataLoader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __getitem__(self, item):
        answer = None
        labels = None
        for i in range(item * self.batch_size, min(len(self.dataset), self.batch_size * (item + 1))):
            if answer is None:
                answer, labels = self.dataset[i]
            else:
                cur_img, cur_label = self.dataset[i]
                answer = np.concatenate((answer, cur_img), axis=0)
                labels = np.concatenate((labels, cur_label), axis=0)
        return answer, labels

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class DataLoaderMNIST:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset.reset_index()
        self.batch_size = batch_size

    def __convert_array(self, x):
        answer = np.zeros((x.shape[0], 10))
        for i, value in enumerate(x):
            answer[i][value] = 1
        return answer

    def __getitem__(self, item):
        answer = None
        labels = None
        subdata = self.dataset[(item * self.batch_size):min(self.dataset.shape[0], (item + 1)* self.batch_size)]
        pixels = [f'pixel{i}' for i in range(784)]
        labels = subdata['label']
        answer = subdata[pixels].to_numpy()
        return answer, self.__convert_array(labels)

    def __len__(self):
        return (self.dataset.shape[0] + self.batch_size - 1) // self.batch_size


class DataLoaderMNIST2D:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset.reset_index()
        self.batch_size = batch_size
        self.image_size = 28

    def __convert_array(self, x):
        answer = np.zeros((x.shape[0], 10))
        for i, value in enumerate(x):
            answer[i][value] = 1
        return answer

    def __getitem__(self, item):
        answer = None
        labels = None
        subdata = self.dataset[(item * self.batch_size):min(self.dataset.shape[0], (item + 1)* self.batch_size)]
        pixels = [f'pixel{i}' for i in range(784)]
        labels = subdata['label']
        answer = subdata[pixels].to_numpy()
        return answer.reshape((subdata.shape[0], self.image_size, self.image_size)), self.__convert_array(labels)

    def __len__(self):
        return (self.dataset.shape[0] + self.batch_size - 1) // self.batch_size