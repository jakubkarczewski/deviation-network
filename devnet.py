"""
@author: Guansong Pang (with major refactor by Jakub Karczewski)
The algorithm was implemented using Python 3.6.6, Keras 2.2.2 and TensorFlow 1.10.1.
More details can be found in our KDD19 paper.
Guansong Pang, Chunhua Shen, and Anton van den Hengel. 2019.
Deep Anomaly Detection with Deviation Networks.
In The 25th ACM SIGKDDConference on Knowledge Discovery and Data Mining (KDD ’19),
August4–8, 2019, Anchorage, AK, USA.ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3292500.3330871
"""

from os.path import join, isdir, isfile
import argparse

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.backend import mean, std, abs, maximum
from tensorflow.keras.regularizers import l2

SEED = 7
np.random.seed(SEED)
tf.random.set_seed(SEED)

MAX_INT = np.iinfo(np.int32).max


class DevNet:
    """Deviation Network for credit card fraud detection."""
    def __init__(self, epochs, batch_size, num_runs, known_outliers, contamination_rate, seed,
                 dataset_path='./dataset/creditcard.csv', output_path='./output'):
        self.output_path = output_path
        self.dataset_path = dataset_path

        self.epochs = epochs
        self.batch_size = batch_size
        self.num_batches = None
        self.num_runs = num_runs
        self.known_outliers = known_outliers
        self.contamination_rate = contamination_rate
        self.seed = seed

        self.scaler = MinMaxScaler()

    @staticmethod
    @tf.function
    def deviation_loss(y_true, y_pred):
        """Z-score based deviation loss"""
        confidence_margin = 5.
        ref = tf.cast(np.random.normal(loc=0., scale=1.0, size=5000), dtype=tf.float32)
        dev = (y_pred - mean(ref)) / std(ref)
        inlier_loss = abs(dev)
        outlier_loss = abs(maximum(confidence_margin - dev, 0.))
        return mean((1 - y_true) * inlier_loss + y_true * outlier_loss)

    def deviation_network(self, input_shape, l2_coef=0.01):
        """Construct the deviation network-based detection model."""
        x_input = Input(shape=input_shape)
        intermediate = Dense(20, activation='relu', kernel_regularizer=l2(l2_coef), name='hl1')(x_input)
        intermediate = Dense(1, activation='linear', name='score')(intermediate)
        model = Model(x_input, intermediate)
        rms = RMSprop(clipnorm=1.)
        model.compile(loss=self.deviation_loss, optimizer=rms)
        return model

    def get_training_data_generator(self, x, outlier_indices, inlier_indices, rng):
        """Generates batches of training data."""
        rng = np.random.RandomState(rng.randint(MAX_INT, size=1))
        counter = 0
        while True:
            ref, training_labels = self.get_one_training_batch(x, outlier_indices, inlier_indices, rng)
            counter += 1
            yield ref.astype('float32'), training_labels.astype('float32')

            if counter > self.num_batches:
                counter = 0

    def get_one_training_batch(self, x_train, outlier_indices, inlier_indices, rng):
        """Returns a single batch of training data. Alternates between positive and negative pairs."""
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
        """Returns performance metrics."""
        roc_auc = roc_auc_score(labels, mse)
        avg_precision = average_precision_score(labels, mse)
        print(f"AUC-ROC: {roc_auc}, AUC-PR: {avg_precision}")
        return roc_auc, avg_precision

    @staticmethod
    def plot_loss(history):
        # Plot loss values
        plt.plot(history.history['loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()

    def run_devnet(self):
        # create placeholder variables for scores in each run
        rauc = np.zeros(self.num_runs)
        ap = np.zeros(self.num_runs)
        # read data
        dataset = pd.read_csv(self.dataset_path)
        # scale data
        scaled_data = self.scaler.fit_transform(dataset)
        scaled_dataset = pd.DataFrame(scaled_data, columns=dataset.columns, index=dataset.index)
        y = scaled_dataset['Class'].values
        x = scaled_dataset.drop(columns=['Class']).values
        print(f'X size {x.shape}')
        print(f'Number of batches per epoch: {len(x) // self.batch_size}')
        self.num_batches = len(x) // self.batch_size
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

            # add noise to data
            n_noise = len(np.where(y_train == 0)[0]) * self.contamination_rate / (1. - self.contamination_rate)
            n_noise = int(n_noise)

            # todo: figure out what is going on here - removal of excess outliers?
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
            model_name = f'devnet_cr:{self.contamination_rate}_bs:{self.batch_size}_ko:{self.known_outliers}.h5'
            checkpointer = ModelCheckpoint(join(self.output_path, model_name), monitor='loss', verbose=0,
                                           save_best_only=True)

            generator = self.get_training_data_generator(x_train, outlier_indices, inlier_indices, rng)
            history = model.fit(generator, steps_per_epoch=self.num_batches, epochs=self.epochs,
                                callbacks=[checkpointer])

            scores = model.predict(x_test)
            rauc[run_id], ap[run_id] = self.auc_performance(scores, y_test)

        # show training history
        self.plot_loss(history)

        mean_auc = np.mean(rauc)
        std_auc = np.std(rauc)
        mean_aucpr = np.mean(ap)
        std_aucpr = np.std(ap)
        print(f'AUC:\nmean: {mean_auc}\nstd: {std_auc}\nAUC_PR\nmean: {mean_aucpr}\nstd: {std_aucpr}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default='dataset/', help="Path to the csv with creditcard dataset")
    parser.add_argument("--output_path", type=str, default='output/', help="Path to the output directory.")
    args = parser.parse_args()

    dataset_path = args.dataset_path if isfile(args.dataset_path) else None
    output_path = args.output_dir if isdir(args.output_dir) else None

    is_gpu = tf.test.is_gpu_available()
    print(f"GPU is{'' if is_gpu else ' not'} available.")

    dev_net_conf = {
        'batch_size': 512,
        'epochs': 250,
        'num_runs': 5,
        'known_outliers': 500,
        'contamination_rate': 0.02,
        'seed': SEED
    }
    if dataset_path:
        dev_net_conf['dataset_path'] = dataset_path

    if output_path:
        dev_net_conf['output_path'] = output_path

    dev_net = DevNet(**dev_net_conf)
    dev_net.run_devnet()
