import os
import keras
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from data_process import *
from matplotlib.ticker import AutoMinorLocator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

import time
start = time.time()

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
        plt.clf()

        sns.pairplot(self.data, kind='reg')
        plt.show()

@dataclass
class Data:
    train: np.array
    test: np.array
    neurons: tuple[int]
    activations: tuple[str]
    epochs: int
    batch: int
    label: str


class NeuralNetwork():

    def __init__(
        self,
        train,
        test,
        true_energy, 
        args: Optional[dict] = None
    ):
        
        _args = {
            'early'         : False,
            'learn_rate'    : 0.001,
            'load'          : False,
            'log'           : True,
            'loss'          : 'mae',
            'metrics'       : 'mae',
            'optimizer'     : 'adam',
            'overwrite'     : True,
            'save'          : False,
            'save_fig'      : False,
            'scaler'        : StandardScaler(),
            'show'          : True,
            'true_e'        : None,
            'txt_title'     : 'Non-Ensemble Model'
        }

        if args is not None:
            for key in args.keys():
                _args[key] = args[key]
        
        self.args = _args

        self.X_train = self.args['scaler'].fit_transform(train)
        self.X_test = self.args['scaler'].fit_transform(test)

        self.true_energy = true_energy
        
        self.is_built = False

        self.callbacks = {}
        self.models = {}
        self.histories = {}
        self.min_mae = {}
        self.predictions = {}

        if self.args['save_fig']:
            directory = '/home/bread/Documents/projects/neutrino/data/fig/plots'
            pathdir = pathlib.Path(directory)
            labels = []

            for path in pathdir.iterdir():
                labels.append(int(str(path).split('/')[-1]))

            new_dir = f"{max(labels)+1}"
            parent_dir = "/home/bread/Documents/projects/neutrino/data/fig/plots/"

            self.path = os.path.join(parent_dir, new_dir)

            os.mkdir(self.path)

    def build(self, truth):

        assert len(truth.neurons) == len(truth.activations), "neuron and activation layers must be same size"

        if self.args['early']:

            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                min_delta=1e-20, 
                patience=10, 
                verbose=1, 
                mode='min', 
                restore_best_weights=True
            ) 

            self.callbacks[truth.label] = early_stop


        if self.args['optimizer'] is None:
            opt = keras.optimizers.Adam(
                learning_rate=self.args['learning_rate']
            )
        else:
            opt = self.args['optimizer']

        if self.args['load']:
            self.model = keras.saving.load_model(f'src/models/model_{truth.label}.keras')

        else:
            model = keras.Sequential()

            model.add(keras.Input(shape=np.shape(self.X_train[0]), name='Input'))

            for i in range(len(truth.neurons)):

                model.add(
                    layers.Dense(
                        truth.neurons[i], 
                        activation=truth.activations[i], 
                        name=f"Layer{i}"
                    )
                )

            self.models[truth.label] = model
        
        self.models[truth.label].compile(
            optimizer=opt,
            loss=[self.args['loss']],
            metrics=[self.args['metrics']]
        )

        self.is_built = True

    def train(self, truth):

        if self.args['early']:
            callbacks = self.callbacks[truth.label]
        else:
            callbacks = []
        
        self.histories[truth.label] = self.models[truth.label].fit(
            self.X_train,
            truth.train,
            validation_split=0.20,
            epochs=truth.epochs,
            batch_size=truth.batch,
            verbose=2,
            callbacks=callbacks
        )

        if self.args['save']:
            self.model.save(
                f'src/models/model_{truth.label}.keras', 
                overwrite=self.args['overwrite']
            )

        if self.args['save_fig']:
            self.models[truth.label].save(
                f'{self.path}/model_{truth.label}.keras'
            )

        self.min_mae[truth.label] = min(self.histories[truth.label].history['val_mae'])

        return self.min_mae[truth.label]

    def test(self, truth):

        prediction = self.models[truth.label].predict(self.X_test)

        self.predictions[truth.label] = [i[0] for i in prediction]

        if self.args['save_fig']:
            epochs = truth.epochs
            batch = truth.batch
            opt = self.args['optimizer']
            lr = self.args['learn_rate']

            with open(f'{self.path}/model_{truth.label}.txt', 'w') as f:
                title = self.args['txt_title']
                f.write(f'{title}\nmin mae, neurons, activations, epochs, batch, opt, lr\n{self.min_mae[truth.label]}\n{truth.neurons}\n{truth.activations}\n{epochs}\n{batch}\n{opt}\n{lr}')

    def eval(self):
        pass

    def plot(self, truth):

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.plot(self.histories[truth.label].history['mae'])
        plt.plot(self.histories[truth.label].history['val_mae'])
        if self.args['log']:
            ax.set_yscale('log')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        plt.title(f'Neural Network Mean Absolute Error - Training For {truth.label}')
        plt.ylabel('Mean Absolute Error')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper left')
        if self.args['save_fig']:
            plt.savefig(f'{self.path}/nn_mae_history.png')
        if self.args['show']:
            plt.show()
        plt.clf()
        
        plt.hist2d(truth.test, self.predictions[truth.label], bins=30)
        plt.xlabel(f'True {truth.label}')
        plt.ylabel(f'Predicted {truth.label}')
        plt.title(f'Using Neural Network to Predict Neutrino {truth.label} - 2d Histogram')
        # plt.xlim([0,1])
        # plt.ylim([0,1])  
        if self.args['save_fig']:
            plt.savefig(f'{self.path}/nn_{truth.label.lower()}_2d.png')
        if self.args['show']:
            plt.show()
        plt.clf()

        plt.hist(truth.test, bins=50, label=f'True {truth.label}', alpha=0.6, color='olive')
        plt.hist(self.predictions[truth.label], bins=50, label=f'Predicted {truth.label}', alpha=0.6, color='deepskyblue')
        plt.title(f"Using Neural Network to Predict Neutrino {truth.label} - 1d Histogram")
        plt.legend()
        if self.args['save_fig']:
            plt.savefig(f'{self.path}/nn_{truth.label.lower()}_1d.png')
        if self.args['show']:
            plt.show()
        plt.clf()

        error = np.array(self.predictions[truth.label]) - np.array(truth.test)

        plt.hist2d(self.true_energy, error, bins=30)
        plt.xlabel(f'True {truth.label} Energy')
        plt.ylabel(f'Model Error')
        plt.title(f'True Neutrino Energy vs {truth.label} Error')
        plt.xlim([0,3])
        if self.args['save_fig']:
            plt.savefig(f'{self.path}/true_e_vs_{truth.label.lower()}_error.png')
        if self.args['show']:
            plt.show()
        plt.clf()

        return


if __name__ == "__main__":

    ensemble =  True
    neural = True
    feature = False
    analysis = False

    txt_title = 'Training Energy Working Towards Energy/Cascade Ensemble Model'

    df = pd.read_csv('/home/bread/Documents/projects/neutrino/data/csv/built.csv')

    inelasticity = df['Cascade']/df['Energy']
    df.insert(0, 'Inelasticity', inelasticity)

    if feature:
        zenithCos = np.cos(df['Zenith']) * df['Track']
        zenithSin = np.sin(df['Zenith']) * df['Track']
        df['ZenithCos'] = zenithCos
        df['ZenithSin'] = zenithSin

    y = df[['Inelasticity', 'Energy', 'Cascade']]

    X = np.array(df.drop(['Inelasticity', 'Energy', 'Cascade'], axis=1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    true_energy = y_test['Energy']

    model = NeuralNetwork(
                X_train, 
                X_test,
                true_energy,
                args={
                    'save_fig'      : True,
                    'show'          : True,
                    'txt_title'     : txt_title
                }
            )

    energy = Data(
        train=y_train['Energy'], 
        test=y_test['Energy'],  
        neurons=[256, 128, 64, 32, 16, 8, 4, 2, 1], 
        activations=['tanh', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'linear'],
        epochs=50,
        batch=3000,
        label='Energy'
    )

    cascade = Data(
        train=y_train['Cascade'], 
        test=y_test['Cascade'],  
        neurons=[64, 32, 16, 8, 4, 2, 1], 
        activations=['tanh', 'relu', 'relu', 'relu', 'relu', 'relu', 'linear'],
        epochs=50,
        batch=3000,
        label='Cascade'
    )

    model.build(energy)
    model.train(energy)
    model.test(energy)
    model.plot(energy)

    end = time.time()
    print(f"{end-start:.4f}s")    
