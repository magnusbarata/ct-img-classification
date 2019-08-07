import pydicom as dcm
import numpy as np
import keras
from sklearn.preprocessing import MultiLabelBinarizer

def generator(list_IDs, labels=None, n_class=10, batch_size=32, dim=(512,512), n_channels=1):
    'Simple generator function for multi-class classification'
    b_features = np.zeros(shape=(batch_size, *dim, n_channels))
    b_labels = [None] * batch_size

    while True:
        indices = np.random.choice(len(list_IDs), batch_size)
        for i, index in enumerate(indices):
            ds = dcm.dcmread(list_IDs[index])
            pixel_array = ds.pixel_array
            b_features[i] = pixel_array.reshape(pixel_array.shape+(n_channels,))
            if labels is not None: b_labels[i] = labels[index]

        if labels is None: yield b_features
        else: yield b_features, keras.utils.to_categorical(b_labels, n_class)

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels=None, batch_size=32, dim=(512,512), n_channels=1,
                 n_class=10, shuffle=True, multi=False):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.multi = multi
        if multi:
            self.mlb = MultiLabelBinarizer()
            self.mlb.fit(labels)
            self.classes = self.mlb.classes_
            self.n_class = len(self.classes)
        else:
            self.n_class = n_class
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        b_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        if self.labels is None:
            if index == self.__len__()-1:
                b_indices = self.indices[index*self.batch_size:]
            X = np.empty((len(b_indices), *self.dim, self.n_channels))
            for i, index in enumerate(b_indices):
                fname = self.list_IDs[index]
                X[i,] = self.getPixelData(fname)
            return X
        else:
            X, y = self.__data_generation(b_indices)
            return X, y

    def on_epoch_end(self):
        'Updates indices after each epoch'
        self.indices = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __data_generation(self, indices):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        #y = np.empty((self.batch_size))
        y = [None] * self.batch_size

        # Generate data
        for i, index in enumerate(indices):
            fname = self.list_IDs[index]
            X[i,] = self.getPixelData(fname)
            y[i] = self.labels[index]

        if self.multi: return X, self.mlb.transform(y)
        else: return X, keras.utils.to_categorical(y, num_classes=self.n_class)

    def getPixelData(self, fname):
        ds = dcm.dcmread(fname)
        pixel_array = ds.pixel_array
        return pixel_array.reshape(pixel_array.shape+(self.n_channels,))
