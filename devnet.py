"""
@author: Guansong Pang (with major refactor by Jakub Karczewski)
The algorithm was implemented using Python 3.6.6, Keras 2.2.2 and TensorFlow 1.10.1.
More details can be found in our KDD19 paper.
Guansong Pang, Chunhua Shen, and Anton van den Hengel. 2019.
Deep Anomaly Detection with Deviation Networks.
In The 25th ACM SIGKDDConference on Knowledge Discovery and Data Mining (KDD ’19),
August4–8, 2019, Anchorage, AK, USA.ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3292500.3330871
"""

from os.path import join

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as K
from keras import regularizers

# set seed values for reproducibility
SEED = 1
np.random.seed(SEED)
tf.set_random_seed(SEED)
sess = tf.Session()

MAX_INT = np.iinfo(np.int32).max


class DevNet:
    """Deviation Network for credit card fraud detection."""
    def __init__(self, epochs, batch_size, num_batches, num_runs, known_outliers, contamination_rate, seed,
                 dataset_path='./dataset/creditcard.csv', output_dir='./output'):
        self.output_dir = output_dir
        self.dataset_path = dataset_path

        self.epochs = epochs
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.num_runs = num_runs
        self.known_outliers = known_outliers
        self.contamination_rate = contamination_rate
        self.seed = seed

    @staticmethod
    def deviation_loss(y_true, y_pred):
        """Z-score based deviation loss"""
        confidence_margin = 5.
        ref = K.variable(np.random.normal(loc=0., scale=1.0, size=5000), dtype='float32')
        dev = (y_pred - K.mean(ref)) / K.std(ref)
        inlier_loss = K.abs(dev)
        outlier_loss = K.abs(K.maximum(confidence_margin - dev, 0.))
        return K.mean((1 - y_true) * inlier_loss + y_true * outlier_loss)

    @classmethod
    def deviation_network(cls, input_shape):
        """Construct the deviation network-based detection model."""
        x_input = Input(shape=input_shape)
        intermediate = Dense(20, activation='relu', kernel_regularizer=regularizers.l2(0.01), name='hl1')(x_input)
        intermediate = Dense(1, activation='linear', name='score')(intermediate)
        model = Model(x_input, intermediate)
        rms = RMSprop(clipnorm=1.)
        model.compile(loss=cls.deviation_loss, optimizer=rms)
        return model

    def batch_generator_sup(self, x, outlier_indices, inlier_indices, rng):
        """Generator """
        rng = np.random.RandomState(rng.randint(MAX_INT, size=1))
        counter = 0
        while True:
            ref, training_labels = self.input_batch_generation_sup(x, outlier_indices, inlier_indices, rng)
            counter += 1
            yield ref, training_labels

            if counter > self.num_batches:
                counter = 0

    def input_batch_generation_sup(self, x_train, outlier_indices, inlier_indices, rng):
        """Returns batches of samples. Alternates between positive and negative pairs."""
        dim = x_train.shape[1]
        ref = np.empty((self.batch_size, dim))
        training_labels = []
        n_inliers = len(inlier_indices)
        n_outliers = len(outlier_indices)
        for i in range(self.batch_size):
            if i % 2 == 0:
                sid = rng.choice(n_inliers, 1)
                ref[i] = x_train[inlier_indices[sid]]
                training_labels += [0]
            else:
                sid = rng.choice(n_outliers, 1)
                ref[i] = x_train[outlier_indices[sid]]
                training_labels += [1]
        return np.array(ref), np.array(training_labels)

    def inject_noise(self, seed, n_out):
        """
        Add anomalies to training data to replicate anomaly contaminated data sets.
        We randomly swap 5% features of anomalies to avoid duplicate contaminated anomalies.
        """
        rng = np.random.RandomState(self.seed)
        n_sample, dim = seed.shape
        swap_ratio = 0.05
        n_swap_feat = int(swap_ratio * dim)
        noise = np.empty((n_out, dim))
        for i in np.arange(n_out):
            outlier_idx = rng.choice(n_sample, 2, replace=False)
            o1 = seed[outlier_idx[0]]
            o2 = seed[outlier_idx[1]]
            swap_feats = rng.choice(dim, n_swap_feat, replace=False)
            noise[i] = o1.copy()
            noise[i, swap_feats] = o2[swap_feats]
        return noise

    @staticmethod
    def auc_performance(mse, labels):
        roc_auc = roc_auc_score(labels, mse)
        avg_precision = average_precision_score(labels, mse)
        print(f"AUC-ROC: {roc_auc}, AUC-PR: {avg_precision}")
        return roc_auc, avg_precision

    def run_devnet(self):
        # create placeholder variables for scores in each run
        rauc = np.zeros(self.num_runs)
        ap = np.zeros(self.num_runs)
        # read data
        dataset = pd.read_csv(self.dataset_path)
        y = dataset['Class'].values
        x = dataset.drop(columns=['Class']).values
        # inspect number of frauds (outliers)
        outlier_indices = np.where(y == 1)[0]
        outliers = x[outlier_indices]
        # log original outlier number
        print(f'There are {len(outliers)} original frauds (outliers) in the dataset.')
        # run training several times
        for run_id in np.arange(self.num_runs):
            # split data into train/test
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=self.seed, stratify=y)
            print(f'Run number: {run_id}')
            outlier_indices = np.where(y_train == 1)[0]
            n_outliers = len(outlier_indices)
            print(f"Original training size for run {run_id}: {x_train.shape[0]}, No. outliers: {n_outliers}")
            # todo: figure out what is going on here - removal of excess outliers?

            # add noise to data
            n_noise = len(np.where(y_train == 0)[0]) * self.contamination_rate / (1. - self.contamination_rate)
            n_noise = int(n_noise)

            rng = np.random.RandomState(self.seed)
            if n_outliers > self.known_outliers:
                mn = n_outliers - self.known_outliers
                remove_idx = rng.choice(outlier_indices, mn, replace=False)
                x_train = np.delete(x_train, remove_idx, axis=0)
                y_train = np.delete(y_train, remove_idx, axis=0)

            noises = self.inject_noise(outliers, n_noise)
            x_train = np.append(x_train, noises, axis=0)
            y_train = np.append(y_train, np.zeros((noises.shape[0], 1)))

            # new outlier indices
            outlier_indices = np.where(y_train == 1)[0]
            inlier_indices = np.where(y_train == 0)[0]
            n_outliers = len(outlier_indices)
            print(f"Post-transformations training size for run {run_id}: {x_train.shape[0]}, No. outliers: {n_outliers}")

            input_shape = x_train.shape[1:]

            # create model
            model = self.deviation_network(input_shape)
            print(model.summary())
            model_name = f'devnet_cr:{self.contamination_rate}_bs:{self.batch_size}_ko:{self.known_outliers}.h5'
            checkpointer = ModelCheckpoint(join(self.output_dir, model_name), monitor='loss', verbose=0,
                                           save_best_only=True)

            generator = self.batch_generator_sup(x_train, outlier_indices, inlier_indices, rng)
            model.fit_generator(generator, steps_per_epoch=self.num_batches, epochs=self.epochs,
                                callbacks=[checkpointer])

            scores = model.predict(x_test)
            rauc[run_id], ap[run_id] = self.auc_performance(scores, y_test)

        mean_auc = np.mean(rauc)
        std_auc = np.std(rauc)
        mean_aucpr = np.mean(ap)
        std_aucpr = np.std(ap)
        print(f'AUC:\nmean: {mean_auc}\nstd: {std_auc}\nAUC_PR\nmean: {mean_aucpr}\nstd: {std_aucpr}')


if __name__ == '__main__':
    dev_net_conf = {
        'batch_size': 512,
        'num_batches': 20,  # todo: check, I would rather get rid of it
        'epochs': 250,
        'num_runs': 5,
        'known_outliers': 30,
        'contamination_rate': 0.02,
        'seed': SEED
    }
    dev_net = DevNet(**dev_net_conf)
    dev_net.run_devnet()
