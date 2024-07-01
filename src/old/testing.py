import time
import keras
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import constants as c
import tensorflow as tf
import matplotlib.pyplot as plt

from torch import nn
from joblib import Memory
from data_process import *
from matplotlib.ticker import AutoMinorLocator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


# Caching
cachedir = './cache'
memory = Memory(cachedir, verbose=0)


def zero_pad_util(group, max_len):
  event_length = len(group)
  padding = max_len - event_length
  if padding > 0:
      padded = group.reindex(group.index.tolist() + list(range(group.index[-1] + 1, group.index[-1] + 1 + padding)))
  else:
      padded = group
  return padded.fillna(0)


def zero_pad(pulse):
    """
    Zero pads events to make each event the same length. 
    """

    max_len = pulse.groupby('event_no').size().max()
    padded = pd.concat([zero_pad_util(group, max_len) for _, group in pulse.groupby('event_no')])

    padded = padded.reset_index(drop=True)

    mask = padded['event_no'] != 0
    padded['event_no'] = padded['event_no'].mask(~mask).ffill().astype(int)

    return padded


def sublist(pulse):
    """
    Creates a list of events with each event having a list of data. 
    """

    events = [
        [
            [
                row['dom_x'],
                row['dom_y'],
                row['dom_z'],
                # row['dom_time'],
                row['charge']
            ]
            for _, row in group.iterrows()
        ]
        for _, group in pulse.groupby('event_no')
    ]
    
    return events


@memory.cache
def get_data():

    X, y = feature_adding()
    
    X = zero_pad(X)

    X = sublist(X)

    return np.asarray(X), y


def pre_process():
    # Loading low energy database and attempting to clean it
    path = '/home/bread/Documents/projects/neutrino/data/db/oscNext_genie_level5_v02.00_pass2.141122.000000.db'
    pulse, truth = get_db(path)

    # Begin processing data
    process = PulseDataProcessing((pulse,truth))

    # Zero pad events to make them the same length then sublist by
    # event number.
    process.zero_pad()
    process.sublist()

    print(type(process.pulse_shape))
    print(process.pulse_shape)

    # Finalizing pulse and truth data
    pulse = process.pulse
    truth = np.array(truth['inelasticity'])

    # Set up CNN model
    model = process.get_model()
    model.summary()

    x_train = pulse[:-250]


def sublist(data) -> None:
        """
        Creates a list of events with each event having a list of data. 
        """

        events = [
            [
                [
                    row['dom_x'],
                    row['dom_y'],
                    row['dom_z'],
                    # row['dom_time'],
                    row['charge']
                ]
                for _, row in group.iterrows()
            ]
            for _, group in data.groupby('event_no')
        ]
        
        return events


@dataclass
class DOMAttributes():
    zenith: float
    azimuth: float
    time: float
    track: float


class DOMGraph():

    def __init__(self) -> None:
        self.node = None
        self.edge = None

    def add_node(self, x, y, z):
        self.node.append((x, y, z))

    def add_edge(self, x, y, z):
        self.edge.append((x, y, z))


def k_nearest_neighbors():
    # Loading low energy database
    path = '/home/bread/Documents/projects/neutrino/data/db/oscNext_genie_level5_v02.00_pass2.141122.000000.db'
    pulse, truth = get_db(path)
    
    events = sublist(pulse)

    event = events[1]

    print(len(event))

    nodes = {}

    for dom in event:
        r = 0
        for position in dom:
            r += position**2

        if np.sqrt(r) < distances[-1]:
            distances[-1] = np.sqrt(r)
            distances = np.sort(distances)
            nodes.append(dom)


    print(distances)
    print(nodes)



def check_energy_distribution():
    # Reading simulation data from hdf5 file
    hdf5 = '/home/bread/Documents/projects/neutrino/data/hdf5/NuMu_genie_149999_030000_level6.zst_cleanedpulses_transformed_IC19.hdf5'
    hdf5_df = get_hdf5(hdf5, 'output_label_names', 'labels')

    plt.title("Energy Distribution")
    plt.hist(hdf5_df['Energy'], bins=50, label='Truth', alpha=0.6, color='deepskyblue')
    plt.legend()
    plt.show()


def feature_adding():
    # Loading low energy database 
    path = '/home/bread/Documents/projects/neutrino/data/db/oscNext_genie_level5_v02.00_pass2.141122.000000.db'
    pulse,truth = get_db(path)

    # print(pulse)

    # print(pulse['charge'])

    change_array = pulse['charge'].pct_change(periods=1).dropna()

    pulse = pulse[1:]
    pulse['charge change'] = change_array

    return pulse, truth

class DataAnalysis():

    def __init__(self, data, save):
        self.data = data
        self.save = save

    def get_correlation(self):

        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes()
        ax.set_facecolor('#faf0e6')
        fig.patch.set_facecolor('#faf0e6')
        cor = self.data.corr()
        sns.heatmap(cor, annot=True, cmap="Reds")
        if self.save:
            plt.savefig('data/fig/correlation_map.png')
        plt.show()


# TODO fill out this framework
class NeuralNetwork():

    def __init__(self) -> None:
        pass

    
    def build(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def eval(self):
        pass

    def plot(self):
        pass

# TODO: Another option to do following 
# https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.sigmoid_stack = nn.Sequential(
            nn.Sigmoid(),
            nn.Sigmoid()
        )

    def forward(self, x):
        logits = self.sigmoid_stack
        return logits


def train_loop(X, y, model, loss_fn):
    return

def test_loop(X, y, model, loss_fn):
    return


# TODO ===============================================================


# TODO move this into a class with more functionality and readability
def neural_network(
    X, 
    y, 
    epochs, 
    save,
    overwrite,
    load,
    batch,
    learning_rate,
    dropout_rate,
    optimizer=None,
    model_eval=False,
    early=False, 
    save_fig=False,
    show=False, 
    log=False
):

    # Scale input data
    scaler=RobustScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

    if load:
        model = keras.saving.load_model('model.keras')

    else:
        model = keras.Sequential()
        # relu, exponential, sigmoid -- options
        model.add(keras.Input(shape=np.shape(X[0]), name="Input"))
        # model.add(layers.Dense(32000, activation='relu', name="Layer07"))
        # model.add(layers.Dense(16284, activation='relu', name="Layer06"))
        # model.add(layers.Dense(8192, activation='relu', name="Layer05"))
        # model.add(layers.Dense(4096, activation='relu', name="Layer04"))
        # model.add(layers.Dense(2048, activation='relu', name="Layer03"))
        # model.add(layers.Dense(1024, activation='relu', name="Layer02"))
        # model.add(layers.Dense(512, activation='relu', name="Layer01"))
        # model.add(layers.Dense(256, activation='relu', name="Layer0"))
        # model.add(layers.Dense(128, activation='relu', name="Layer1"))
        # model.add(layers.Dense(64, activation='relu', name="Layer2"))
        # model.add(layers.Dropout(rate=dropout_rate))
        # model.add(layers.Dense(32, activation='relu', name="Layer3"))
        model.add(layers.Dense(16, activation='relu', name="Layer4"))
        model.add(layers.Dense(16, activation='relu', name="Layer04"))
        model.add(layers.Dense(16, activation='relu', name="Layer004"))
        model.add(layers.Dense(16, activation='relu', name="Layer0004"))
        model.add(layers.Dense(16, activation='relu', name="Layer00004"))
        model.add(layers.Dense(16, activation='relu', name="Layer000004"))
        model.add(layers.Dense(8, activation='relu', name="Layer5"))
        model.add(layers.Dense(8, activation='relu', name="Layer6"))
        model.add(layers.Dense(4, activation='relu', name="Layer7"))
        model.add(layers.Dense(1, activation='linear', name="Layer8"))

    callbacks = []

    if early:
        
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            min_delta=1e-20, 
            patience=10, 
            verbose=1, 
            mode='min', 
            restore_best_weights=True
        ) 

        callbacks.append(early_stop)

    if optimizer is None:
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        opt = optimizer
    model.compile(loss='mae', optimizer=opt, metrics=['mae'])

    history = model.fit(
        X_train, 
        y_train, 
        validation_split=0.30, 
        epochs=epochs, 
        batch_size=batch, 
        verbose=2,
        callbacks=callbacks
    )

    if save:
        model.save('model.keras', overwrite=overwrite)

    print(f"Model Eval: {model.evaluate(X_test, y_test)[0]} ============\n")

    if model_eval:
        return (min(history.history['val_mae']))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    if log:
        ax.set_yscale('log')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    plt.title('Model Mean Absolute Error')
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    if save_fig:
        plt.savefig('/home/bread/Documents/projects/neutrino/data/fig/nn_mae_history.png')
    if show:
        plt.show()

    prediction = model.predict(X_test)
    prediction = [i[0] for i in prediction]

    error = prediction - y_test

    plt.hist2d(y_test, prediction, bins=30)
    plt.xlabel('True Inelasticity')
    plt.ylabel('Predicted Inelasticity')
    plt.title('Using Neural Network to Predict Neutrino Ineslasticity - 2d Histogram')
    plt.xlim([0,1])
    plt.ylim([0,1])  
    if save_fig:
        plt.savefig('/home/bread/Documents/projects/neutrino/data/fig/nn_inelasticity_2d.png')
    if show:
        plt.show()

    plt.title("Using Neural Network to Predict Neutrino Ineslasticity - 1d Histogram")
    plt.hist(y_test, bins=50, label='True Inelasticity', alpha=0.6, color='olive')
    plt.hist(prediction, bins=50, label='Predicted Inelasticity', alpha=0.6, color='deepskyblue')
    plt.legend()
    plt.savefig('/home/bread/Documents/projects/neutrino/data/fig/nn_inelasticity_1d_histogram.png')
    if show:
        plt.show()


    # true_e = np.array(y_test['Energy'])
    # y_train = np.array(y_train['Cascade'])
    # y_test = np.array(y_test['Cascade'])

    # plt.hist2d(true_e, error, bins=30)
    # plt.xlabel('True Neutrino Energy')
    # plt.ylabel('Predicted - True Inelasticity (Error)')
    # plt.title('True Neutrino Energy vs Inelasticity Error')
    # if save_fig:
    #     plt.savefig('/home/bread/Documents/projects/neutrino/data/fig/true_e_vs_inelasticity_error.png')
    # plt.xlim([0,1])
    # if show:
    #     plt.show()

    # hist, bin_edges = np.histogram(error, bins=30)
    # quantiles = np.quantile(hist, q=[0.1, 0.5, 0.9])
    # print(quantiles[-1])
    # plt.fill_between(error, quantiles[0], quantiles[-1])
    # # plt.plot(bin_edges, quantiles[1])
    # plt.show()

    return min(history.history['val_mae'])


if __name__ == "__main__":

    # Is correlation analysis being run
    analysis = False
    
    # Is NN being run
    nn = True

    # Are hyperparameters being tuned
    optimizing=False
    
    # Which hyperparameters?
    learning_rate=False
    batch_size=False
    drouput=True


    # Otimizers
    # nn_evals = pd.read_csv('data/csv/hyperparameter/batch_size.csv')
    # print(nn_evals)
    # print(nn_evals.min(axis=0).sort_values())
    # quit()

    df = cat_hdf5("/home/bread/Documents/projects/neutrino/data/hdf5", 'output_label_names', 'labels')

    # TODO - KEEP: Track!!, Time, Z, IsTrack, IsAntiNeutrino, IsCC, ZenithCos, ZenithSin
    # TODO - DROP: Zenith, X, Y, Azimuth, Charge
    # TODO - OPTIMIZER RANK: LION, NADAM, ADAM, ADAMW
    # TODO - BATCH RANK: 1024, 2048, 64, 128, 4000, 6000, 256, 512

    # Insert inelasticity
    inelasticity = df['Cascade']/df['Energy']
    df.insert(0, 'Inelasticity', inelasticity)

    # Trying angles with zenith and azumith
    # zenithCos = np.cos(df['Zenith']) * df['Track']
    # zenithSin = np.sin(df['Zenith']) * df['Track']
    # df['ZenithCos'] = zenithCos
    # df['ZenithSin'] = zenithSin

    # # Trying charge * track
    # chargeTrack = -df['Charge'] * df['Track']
    # df['ChargeTrack'] = chargeTrack


    if analysis:

        # Run correlation
        data = DataAnalysis(df, True)
        data.get_correlation()
    
    if nn:
        y = pd.DataFrame()

        y['Cascade'] = df['Cascade']
        y['Energy'] = df['Energy']

        X = df.drop(['Inelasticity', 'Energy', 'Cascade'], axis=1)

        if not optimizing:
            min_mae = neural_network(
                X, y, 
                epochs=750, 
                save=False, 
                overwrite=False, 
                load=False,
                optimizer='adam',
                batch=3000,
                learning_rate=0.001, 
                dropout_rate=0.05,
                show=True,
                save_fig=True
            )
            print(min_mae)

        # Tuning hyperparameters
        else:
            
            if learning_rate:

                nn_evals = pd.DataFrame()

                for lr in [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1]:

                    evals = []

                    for i in range(5):
                        evals.append(
                            neural_network(
                            X, y, 
                            epochs=75, 
                            save=False, 
                            overwrite=False, 
                            load=False, 
                            optimizer=None,
                            batch=1024,
                            learning_rate = lr,
                            model_eval=True,
                            show=False,
                            )
                        )
                        print(evals)
                    
                    nn_evals[f"{lr}"] = evals

                nn_evals.to_csv("data/csv/hyperparameter/learning_rate.csv", index=False)

                print(nn_evals)
                print(nn_evals.min(axis=0).sort_values())

            if drouput:

                nn_evals = pd.DataFrame()

                for drop in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

                    evals = []

                    for i in range(5):
                        evals.append(
                            neural_network(
                            X, y, 
                            epochs=75, 
                            save=False, 
                            overwrite=False, 
                            load=False, 
                            optimizer=None,
                            batch=1024,
                            learning_rate = 0.001,
                            dropout_rate=drop,
                            model_eval=True,
                            show=False,
                            )
                        )
                        print(evals)
                    
                    nn_evals[f"{drop}"] = evals

                nn_evals.to_csv("data/csv/hyperparameter/dropout.csv", index=False)

                print(nn_evals)
                print(nn_evals.min(axis=0).sort_values())

