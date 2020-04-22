"""
@author: Guansong Pang (with major refactor by Jakub Karczewski)
The algorithm was implemented using Python 3.6.6, Keras 2.2.2 and TensorFlow 1.10.1.
More details can be found in our KDD19 paper.
Guansong Pang, Chunhua Shen, and Anton van den Hengel. 2019.
Deep Anomaly Detection with Deviation Networks.
In The 25th ACM SIGKDDConference on Knowledge Discovery and Data Mining (KDD ’19),
August4–8, 2019, Anchorage, AK, USA.ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3292500.3330871
"""
import pickle
from os.path import join, isdir, isfile
import argparse
from collections import defaultdict
import json

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, precision_recall_curve, auc, \
    average_precision_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
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
    def __init__(self, epochs, batch_size, num_runs, seed, dataset_path='./dataset/creditcard.csv',
                 output_path='./output', contamination_rate=0.02, limit_of_anomalies=None, noise_ratio=None):
        self.output_path = output_path
        self.dataset_path = dataset_path

        self.epochs = epochs
        self.batch_size = batch_size
        self.num_runs = num_runs

        self.limit_of_anomalies = limit_of_anomalies
        self.contamination_rate = contamination_rate
        self.noise_ratio = noise_ratio
        self.seed = seed

        self.scaler = MinMaxScaler()
        self.random_state = np.random.RandomState(self.seed)
        # 95% confidence threshold score value
        self.conf_threshold = 1.96

    @staticmethod
    @tf.function
    def deviation_loss(y_true, y_pred):
        """Z-score based deviation loss"""
        confidence_margin = 5.
        ref = tf.cast(np.random.normal(loc=0., scale=1.0, size=5000), dtype=tf.float32)
        dev = (y_pred - mean(ref)) / std(ref)
        normal_loss = abs(dev)
        anomaly_loss = abs(maximum(confidence_margin - dev, 0.))
        return mean((1 - y_true) * normal_loss + y_true * anomaly_loss)

    def deviation_network(self, input_shape, l2_coef=0.01):
        """Construct the deviation network-based detection model."""
        x_input = Input(shape=input_shape)
        intermediate = Dense(20, activation='relu', kernel_regularizer=l2(l2_coef), name='hl1')(x_input)
        intermediate = Dense(1, activation='linear', name='score')(intermediate)
        model = Model(x_input, intermediate)
        rms = RMSprop(clipnorm=1.)
        model.compile(loss=self.deviation_loss, optimizer=rms)
        return model

    def get_data_generator(self, x, y):
        """Generates batches of training data."""
        anomaly_indexes = np.where(y == 1)[0]
        normal_indexes = np.where(y == 0)[0]
        while True:
            ref, training_labels = self.get_batch(x, anomaly_indexes, normal_indexes)
            yield ref.astype('float32'), training_labels.astype('float32')

    def get_batch(self, x_train, anomaly_indexes, normal_indexes):
        """Preprocess training set by alternating between negative and positive pairs."""
        preprocessed_x_train = np.empty((self.batch_size, x_train.shape[-1]))
        training_labels = []
        n_normal = len(normal_indexes)
        n_anomaly = len(anomaly_indexes)
        for i in range(len(preprocessed_x_train)):
            if i % 2 == 0:
                selected_idx = self.random_state.choice(n_normal, 1)
                preprocessed_x_train[i] = x_train[normal_indexes[selected_idx]]
                training_labels += [0]
            else:
                selected_idx = self.random_state.choice(n_anomaly, 1)
                preprocessed_x_train[i] = x_train[anomaly_indexes[selected_idx]]
                training_labels += [1]
        return np.array(preprocessed_x_train), np.array(training_labels)

    def _inject_noise(self, anomalies, num_noise_samples, x_train, y_train):
        """
        Add anomalies to training data to replicate anomaly contaminated data sets.
        We randomly swap 5% features of anomalies to avoid duplicate contaminated anomalies.
        """
        n_sample, feat_num = anomalies.shape
        n_swap_feat = int(self.noise_ratio * feat_num)
        noise = np.empty((num_noise_samples, feat_num))
        for i in np.arange(num_noise_samples):
            anomaly_idx = self.random_state.choice(n_sample, 2, replace=False)
            anomaly_1 = anomalies[anomaly_idx[0]]
            anomaly_2 = anomalies[anomaly_idx[1]]
            swap_feats = self.random_state.choice(feat_num, n_swap_feat, replace=False)
            noise[i] = anomaly_1.copy()
            noise[i, swap_feats] = anomaly_2[swap_feats]

        x_train = np.append(x_train, noise, axis=0)
        y_train = np.append(y_train, np.zeros((noise.shape[0], 1)))
        return x_train, y_train

    @staticmethod
    def round_top_perc(preds, perc):
        return np.where(preds > np.percentile(preds, perc, interpolation='nearest'), 1, 0)

    def get_metrics(self, gt, preds, convert_to_proba=True):
        """Returns performance metrics."""
        pred_3_perc = self.round_top_perc(np.abs(preds), 97)
        pred_1_perc = self.round_top_perc(np.abs(preds), 99)
        if convert_to_proba:
            preds = np.array([1 if abs(x) >= self.conf_threshold else 0 for x in preds])
        roc_auc = roc_auc_score(gt, preds)
        _, rec_1, _, _ = precision_recall_fscore_support(gt, pred_1_perc)
        _, rec_3, _, _ = precision_recall_fscore_support(gt, pred_3_perc)
        precision, recall, _ = precision_recall_curve(gt, preds)
        prec_rec_auc = auc(recall, precision)
        avg_precision = average_precision_score(gt, preds)
        return {
            'roc_auc': np.around(roc_auc, decimals=4),
            'prec_rec_auc': np.around(prec_rec_auc, decimals=4),
            'recall_1%': np.around(rec_1[-1], decimals=5),
            'recall_3%': np.around(rec_3[-1], decimals=5),
            'avg_precision': np.around(avg_precision, decimals=5)
        }

    @staticmethod
    def plot_loss(history):
        """Plot training & validation loss values."""
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    def _limit_num_anomalies(self, num_anomalies, x_train, y_train):
        """Limit anomalies to certain number."""
        if num_anomalies > self.limit_of_anomalies:
            excess_anomalies = num_anomalies - self.limit_of_anomalies
            remove_idx = self.random_state.choice(num_anomalies, excess_anomalies, replace=False)
            x_train = np.delete(x_train, remove_idx, axis=0)
            y_train = np.delete(y_train, remove_idx, axis=0)
        return x_train, y_train

    def run_devnet(self):
        # create placeholder variables for scores in each run
        all_metrics = dict()
        for metrics_type in ('authors', 'mine'):
            all_metrics[metrics_type] = {name: np.zeros(self.num_runs) for name in
                                         ('roc_auc', 'prec_rec_auc', 'recall_1%', 'recall_3%', 'avg_precision')}
        # read data
        dataset = pd.read_csv(self.dataset_path)
        # scale data
        scaled_data = self.scaler.fit_transform(dataset)
        scaled_dataset = pd.DataFrame(scaled_data, columns=dataset.columns, index=dataset.index)
        y = scaled_dataset['Class'].values
        x = scaled_dataset.drop(columns=['Class']).values
        print(f'X shape {x.shape}')
        # inspect number of frauds (anomalies)
        anomalies = x[np.where(y == 1)[0]]
        print(f'There are {len(anomalies)} original frauds (outliers) in the dataset.')
        # run training several times
        for run_id in np.arange(self.num_runs):
            # split data into train/test
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=self.seed, stratify=y)
            print(f'Run number: {run_id}')
            original_train_anomaly_indices = np.where(y_train == 1)[0]
            num_train_anomalies = len(original_train_anomaly_indices)
            print(f"Original training size for run {run_id}: {x_train.shape[0]}, No. outliers: {num_train_anomalies}")

            if self.limit_of_anomalies:
                x_train, y_train = self._limit_num_anomalies(num_train_anomalies, x_train, y_train)

            if self.noise_ratio:
                # add noise to data
                n_noise = int(len(np.where(y_train == 0)[0]) * self.contamination_rate / (1. - self.contamination_rate))
                train_anomalies = x_train[np.where(y_train == 1)[0]]
                x_train, y_train = self._inject_noise(train_anomalies, n_noise, x_train, y_train)

            print(f"Post-transformations training size for run {run_id}: {x_train.shape[0]},"
                  f" No. outliers: {len(np.where(y_train == 1)[0])}")

            input_shape = x_train.shape[1:]

            # create model
            model = self.deviation_network(input_shape)
            model_name = f'devnet_run_{run_id}.h5'
            callbacks = [
                ModelCheckpoint(join(self.output_path, model_name), monitor='loss', verbose=0, save_best_only=True),
                EarlyStopping(monitor='val_loss', min_delta=0, patience=25, verbose=0, mode='auto', baseline=None,
                              restore_best_weights=True)
            ]

            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=self.seed,
                                                              stratify=y_train)
            train_generator = self.get_data_generator(x_train, y_train)
            val_generator = self.get_data_generator(x_val, y_val)

            history = model.fit(train_generator, validation_data=val_generator,  epochs=self.epochs,
                                steps_per_epoch=len(x_train)//self.batch_size,
                                validation_steps=len(x_val)//self.batch_size, callbacks=callbacks, verbose=0)

            preds = model.predict(x_test)

            for metrics_type, fix in zip(all_metrics, (False, True)):
                for metric_name, value in self.get_metrics(y_test, preds, convert_to_proba=fix).items():
                    all_metrics[metrics_type][metric_name][run_id] = value

            with open(join(self.output_path, f'history_run:{run_id}.pkl'), 'wb') as f:
                pickle.dump(history.history, f)

        metrics_report = dict()
        for metrics_type in all_metrics:
            partial_metrics_report = defaultdict(lambda: {'avg': None, 'std': None})
            for metric_label, values in all_metrics[metrics_type].items():
                print(f'For {metrics_type} metric: {metric_label} -> avg: {values.mean()}, std: {values.std()}')
                partial_metrics_report[metric_label]['avg'] = values.mean()
                partial_metrics_report[metric_label]['std'] = values.std()
            metrics_report[metrics_type] = partial_metrics_report

        with open(join(self.output_path, 'metrics.json'), 'w') as f:
            json.dump(dict(metrics_report), f)

        return {
            'model': model,
            'predictions': preds,
            'ground_truth': y_test,
            'history': history.history,
            'metrics': metrics_report
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default='dataset/', help="Path to the csv with creditcard dataset")
    parser.add_argument("--output_path", type=str, default='output/', help="Path to the output directory.")
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs in trainings.")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of training to run.")
    args = parser.parse_args()

    dataset_path = args.dataset_path if isfile(args.dataset_path) else None
    output_path = args.output_path if isdir(args.output_path) else None

    is_gpu = tf.test.is_gpu_available()
    print(f"GPU is{'' if is_gpu else ' not'} available.")

    dev_net_conf = {
        'batch_size': 512,
        'num_runs': args.num_runs,
        'seed': SEED,
        'epochs': args.epochs
    }
    if dataset_path:
        dev_net_conf['dataset_path'] = dataset_path

    if output_path:
        dev_net_conf['output_path'] = output_path

    dev_net = DevNet(**dev_net_conf)
    results = dev_net.run_devnet()
    DevNet.plot_loss(results['history'])
