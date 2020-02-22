import pydicom as dcm
import numpy as np
import keras
from sklearn.preprocessing import MultiLabelBinarizer
from skimage.transform import resize

def expand_channel(img, ranges=None, data_format='channels_last'):
    out_arr = []
    if ranges is None: ranges = [(np.amin(img), np.amax(img))]
    for r in ranges:
        assert len(r) is 2, 'ranges values must be a tuple of (min, max)'
        assert r[0] < r[1], 'ranges\'s min value must be lower than max'
        c = np.where((img<r[0]) | (img>r[1]), 0, img)
        out_arr.append(c)
    if data_format == 'channels_last': return np.stack(out_arr, axis=-1)
    elif data_format == 'channels_first': return np.array(out_arr)
    else: return out_arr

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels=None, batch_size=32, dim=(512,512), n_channels=1,
                 n_class=10, shuffle=True, normalize=None, aug=None, multi=False):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.normalize = normalize
        self.aug = aug
        self.data_shape = dim + (n_channels,)
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
            if self.aug is not None:
                X = next(self.aug.flow(X, batch_size=X.shape[0], shuffle=False))
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
        pixel_array = ds.pixel_array.astype('float64')
        pixel_array = resize(pixel_array, self.dim)
        if self.normalize is not None:
            pixel_array = self.normalize['method'](pixel_array, self.normalize['args'])
        return expand_channel(pixel_array)#, ranges=[(-70,50), (-10,110), (50,170)]) #pixel_array.reshape(pixel_array.shape+(self.n_channels,))
