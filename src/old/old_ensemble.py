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


class NeuralNetwork():

    def __init__(
        self,
        features,
        truth,
        neurons,
        activations,
        title: str, 
        args: Optional[dict] = None
    ):
        
        _args = {
            'batch'         : 3000,
            'epochs'        : 100,
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
            'title'         : 'Non-Ensemble Model'
        }

        if args is not None:
            for key in args.keys():
                _args[key] = args[key]
        
        self.args = _args

        assert len(neurons) == len(activations), 'neurons and activation function lists must be the same length'

        self.neurons = neurons
        self.activations = activations
        self.title = title

        X = self.args['scaler'].fit_transform(np.array(features))
        y = truth 

        self.X_train, self.X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        self.true_e  = np.array(y_test['Energy'])

        y_train = np.array(y_train.drop(['Energy'], axis=1))
        self.y_train = np.array([i[0] for i in y_train])
        
        y_test = np.array(y_test.drop(['Energy'], axis=1))
        self.y_test = np.array([i[0] for i in y_test])

        self.callbacks = []
        self.is_built = False

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


    def __repr__(self):
        if self.is_built:
            return str(f"{self.model_cascade.summary()}\n{self.model_inelasticity.summary()}")
        else:
            return str(f"""
Cascade Feature Shape: {np.shape(self.X_cascade)}
Inelasticity Feature Shape: {np.shape(self.X_inelasticity)}
Testing Feature Shape: {np.shape(self.X_test)}

Cascade Truth Shape: {np.shape(self.y_cascade)}
Inelasticity Truth Shape: {np.shape(self.y_inelasticity)}
Testing Truth Shape: {np.shape(self.y_test)}

Model Params: 
  Neurons - {self.neurons}
  Activations - {self.activations}
            """)

    def build(self):

        if self.args['early']:

            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                min_delta=1e-20, 
                patience=10, 
                verbose=1, 
                mode='min', 
                restore_best_weights=True
            ) 

            self.callbacks.append(early_stop)


        if self.args['optimizer'] is None:
            opt = keras.optimizers.Adam(
                learning_rate=self.args['learning_rate']
            )
        else:
            opt = self.args['optimizer']

        if self.args['load']:
            self.model_cascade = keras.saving.load_model('src/models/model.keras')

        else:
            model = keras.Sequential()

            model.add(keras.Input(shape=np.shape(self.X_train[0]), name='Input'))

            for i in range(len(self.neurons)):

                model.add(
                    layers.Dense(
                        self.neurons[i], 
                        activation=self.activations[i], 
                        name=f"Layer{i}"
                    )
                )

            self.model = model
        
        self.model.compile(
            optimizer=opt,
            loss=[self.args['loss']],
            metrics=[self.args['metrics']]
        )

        self.is_built = True

    def train(self):
        
        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            validation_split=0.20,
            epochs=self.args['epochs'],
            batch_size=self.args['batch'],
            verbose=2,
            callbacks=self.callbacks
        )

        if self.args['save']:
            self.model.save(
                'src/models/model.keras', 
                overwrite=self.args['overwrite']
            )

        if self.args['save_fig']:
            self.model.save(
                f'{self.path}/model.keras'
            )

        self.min_mae = min(self.history.history['val_mae'])

        return self.min_mae

    def test(self):

        prediction = self.model.predict(self.X_test)

        self.prediction = [i[0] for i in prediction]

        if self.args['save_fig']:
            epochs = self.args['epochs']
            batch = self.args['batch']
            opt = self.args['optimizer']
            lr = self.args['learn_rate']

            with open(f'{self.path}/model.txt', 'w') as f:
                title = self.args['title']
                f.write(f'{title}\nmin mae, neurons, activations, epochs, batch, opt, lr\n{self.min_mae}\n{self.neurons}\n{self.activations}\n{epochs}\n{batch}\n{opt}\n{lr}')

    def eval(self):
        pass

    def plot(self):

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.plot(self.history.history['mae'])
        plt.plot(self.history.history['val_mae'])
        if self.args['log']:
            ax.set_yscale('log')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        plt.title(f'Neural Network Mean Absolute Error - Training For {self.title}')
        plt.ylabel('Mean Absolute Error')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper left')
        if self.args['save_fig']:
            plt.savefig(f'{self.path}/nn_mae_history.png')
        if self.args['show']:
            plt.show()
        plt.clf()
        
        plt.hist2d(self.y_test, self.prediction, bins=30)
        plt.xlabel(f'True {self.title}')
        plt.ylabel(f'Predicted {self.title}')
        plt.title(f'Using Neural Network to Predict Neutrino {self.title} - 2d Histogram')
        plt.xlim([0,1])
        plt.ylim([0,1])  
        if self.args['save_fig']:
            plt.savefig(f'{self.path}/nn_{self.title}_2d.png')
        if self.args['show']:
            plt.show()
        plt.clf()

        plt.hist(self.y_test, bins=50, label=f'True {self.title}', alpha=0.6, color='olive')
        plt.hist(self.prediction, bins=50, label=f'Predicted {self.title}', alpha=0.6, color='deepskyblue')
        plt.title(f"Using Neural Network to Predict Neutrino {self.title} - 1d Histogram")
        plt.legend()
        if self.args['save_fig']:
            plt.savefig(f'{self.path}/nn_{self.title}_1d.png')
        if self.args['show']:
            plt.show()
        plt.clf()

        error = np.array(self.prediction) - np.array(self.y_test)

        plt.hist2d(self.true_e, error, bins=30)
        plt.xlabel(f'True {self.title} Energy')
        plt.ylabel(f'Model Error')
        plt.title(f'True Neutrino Energy vs {self.title} Error')
        plt.xlim([0,3])
        if self.args['save_fig']:
            plt.savefig(f'{self.path}/true_e_vs_{self.title}_error.png')
        if self.args['show']:
            plt.show()
        plt.clf()

        return

class EnsembleNeuralNetwork():

    def __init__(
        self,
        features,
        truth,
        neurons,
        activations,
        title: str, 
        args: Optional[dict] = None
    ):
        
        _args = {
            'batch'         : 3000,
            'epochs'        : 100,
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
            'title'         : 'Ensemble Model'
        }

        if args is not None:
            for key in args.keys():
                _args[key] = args[key]
        
        self.args = _args

        assert len(neurons) == len(activations), 'neurons and activation function lists must be the same length'

        self.neurons = neurons
        self.activations = activations
        self.title = title

        # if len(features) % 2 != 0:
        #     features = features[1:]
        #     truth = truth[1:]

        # features = self.args['scaler'].fit_transform(np.array(features))

        # zeros = np.array(np.zeros(len(features))).reshape(-1, 1)
        # features = np.concatenate((zeros, features), axis=1)

        # self.X_cascade = features[:-int(len(features)/2 + 1000)]
        # self.X_inelasticity = features[int(len(features)/2 - 1000):-2000]
        # self.X_test = features[-2000:]

        # cascade = np.array(truth['Cascade'])
        # self.y_cascade = cascade[:-int(len(features)/2 + 1000)]
        # inelasticity = np.array(truth['Inelasticity'])
        # self.y_inelasticity = inelasticity[int(len(features)/2 - 1000):-2000]
        # self.y_test = inelasticity[-2000:]

        # self.callbacks = []
        # self.is_built = False

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


    def __repr__(self):
        if self.is_built:
            return str(f"{self.model_cascade.summary()}\n{self.model_inelasticity.summary()}")
        else:
            return str(f"""
Cascade Feature Shape: {np.shape(self.X_cascade)}
Inelasticity Feature Shape: {np.shape(self.X_inelasticity)}
Testing Feature Shape: {np.shape(self.X_test)}

Cascade Truth Shape: {np.shape(self.y_cascade)}
Inelasticity Truth Shape: {np.shape(self.y_inelasticity)}
Testing Truth Shape: {np.shape(self.y_test)}

Model Params: 
  Neurons - {self.neurons}
  Activations - {self.activations}
            """)

    def build(self):

        if self.args['early']:

            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                min_delta=1e-20, 
                patience=10, 
                verbose=1, 
                mode='min', 
                restore_best_weights=True
            ) 

            self.callbacks.append(early_stop)


        if self.args['optimizer'] is None:
            opt = keras.optimizers.Adam(
                learning_rate=self.args['learning_rate']
            )
        else:
            opt = self.args['optimizer']


        if self.args['load']:
            self.model_cascade = keras.saving.load_model('src/models/model_cascade.keras')

        else:
            model = keras.Sequential()

            model.add(keras.Input(shape=np.shape(self.X_cascade[0]), name='Input'))

            for i in range(len(self.neurons)):

                model.add(
                    layers.Dense(
                        self.neurons[i], 
                        activation=self.activations[i], 
                        name=f"Layer{i}"
                    )
                )

            self.model_cascade = model
        
        
        self.model_cascade.compile(
            optimizer=opt,
            loss=[self.args['loss']],
            metrics=[self.args['metrics']]
        )
        
        if self.args['load']:
            self.model_inelasticity = keras.saving.load_model('src/models/model_inelasticity.keras')

        else:
            model = keras.Sequential()

            model.add(keras.Input(shape=np.shape(self.X_inelasticity[0]), name='Input'))

            for i in range(len(self.neurons)):

                model.add(
                    layers.Dense(
                        self.neurons[i], 
                        activation=self.activations[i], 
                        name=f"Layer{i}"
                    )
                )

            self.model_inelasticity = model
        
        self.model_inelasticity.compile(
            optimizer=opt,
            loss=[self.args['loss']],
            metrics=[self.args['metrics']]
        )

        self.is_built = True

    def train_cascade(self):
        
        self.history_cascade = self.model_cascade.fit(
            self.X_cascade,
            self.y_cascade,
            validation_split=0.20,
            epochs=self.args['epochs'],
            batch_size=self.args['batch'],
            verbose=2,
            callbacks=self.callbacks
        )

        if self.args['save']:
            self.model_cascade.save(
                'src/models/model_cascade.keras', 
                overwrite=self.args['overwrite']
            )
        
        if self.args['save_fig']:
            self.model_cascade.save(
                f'{self.path}/model_cascade.keras'
            )

        cascade_pred = self.model_cascade.predict(
            self.X_inelasticity
        )

        self.X_inelasticity = np.delete(self.X_inelasticity, 0, 1)
        self.X_inelasticity = np.concatenate((cascade_pred, self.X_inelasticity), axis=1)

        self.min_cascade = min(self.history_cascade.history['val_mae'])

        return self.min_cascade

    def train_inelasticity(self):

        self.history_inelasticity = self.model_inelasticity.fit(
            self.X_inelasticity,
            self.y_inelasticity,
            validation_split=0.20,
            epochs=self.args['epochs'],
            batch_size=self.args['batch'],
            verbose=2,
            callbacks=self.callbacks
        )
        
        if self.args['save']:
            model.save(
                'src/models/model_inelasticity.keras', 
                overwrite=self.args['overwrite']
            )

        if self.args['save_fig']:
            self.model_inelasticity.save(
                f'{self.path}/model_inelasticity.keras'
            )

        self.min_inelasticity = min(self.history_inelasticity.history['val_mae'])

        return self.min_inelasticity

    def test(self):

        cascade = self.model_cascade.predict(self.X_test)

        self.X_test = np.delete(self.X_test, 0, 1)
        self.X_test = np.concatenate((cascade, self.X_test), axis=1)

        prediction = self.model_inelasticity.predict(self.X_test)

        self.prediction = [i[0] for i in prediction]

        if self.args['save_fig']:
            epochs = self.args['epochs']
            batch = self.args['batch']
            opt = self.args['optimizer']
            lr = self.args['learn_rate']

            with open(f'{self.path}/model.txt', 'w') as f:
                title = self.args['title']
                f.write(f'{title}\nmin mae, neurons, activations, epochs, batch, opt, lr\n{self.min_inelasticity}\n{self.neurons}\n{self.activations}\n{epochs}\n{batch}\n{opt}\n{lr}')

    def eval(self):
        pass

    def plot(self):

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.plot(self.history_inelasticity.history['mae'])
        plt.plot(self.history_inelasticity.history['val_mae'])
        if self.args['log']:
            ax.set_yscale('log')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        plt.title(f'Neural Network Mean Absolute Error - Training For {self.title}')
        plt.ylabel('Mean Absolute Error')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper left')
        if self.args['save_fig']:
            plt.savefig(f'{self.path}/nn_mae_history.png')
        if self.args['show']:
            plt.show()
        plt.clf()
        
        plt.hist2d(self.y_test, self.prediction, bins=30)
        plt.xlabel(f'True {self.title}')
        plt.ylabel(f'Predicted {self.title}')
        plt.title(f'Using Neural Network to Predict Neutrino {self.title} - 2d Histogram')
        plt.xlim([0,1])
        plt.ylim([0,1])  
        if self.args['save_fig']:
            plt.savefig(f'{self.path}/nn_{self.title}_2d.png')
        if self.args['show']:
            plt.show()
        plt.clf()

        plt.hist(self.y_test, bins=50, label=f'True {self.title}', alpha=0.6, color='olive')
        plt.hist(self.prediction, bins=50, label=f'Predicted {self.title}', alpha=0.6, color='deepskyblue')
        plt.title(f"Using Neural Network to Predict Neutrino {self.title} - 1d Histogram")
        plt.legend()
        if self.args['save_fig']:
            plt.savefig(f'{self.path}/nn_{self.title}_1d.png')
        if self.args['show']:
            plt.show()
        plt.clf()

        return


if __name__ == "__main__":

    ensemble = False
    neural = True
    feature = True
    analysis = False

    title = 'Non-Ensemble Model With Lowest Correlation Drop and Added Features'

    df = build_files("/home/bread/Documents/projects/neutrino/data/hdf5", 'output_label_names', 'labels')

    inelasticity = df['Cascade']/df['Energy']
    df.insert(0, 'Inelasticity', inelasticity)

    if feature:
        zenithCos = np.cos(df['Zenith']) * df['Track']
        zenithSin = np.sin(df['Zenith']) * df['Track']
        df['ZenithCos'] = zenithCos
        df['ZenithSin'] = zenithSin

    if analysis:
        data = DataAnalysis(df, True)
        data.get_correlation()

    if ensemble:
        y = df[['Inelasticity', 'Cascade']]
    else:
        y = df[['Inelasticity', 'Energy']]

    X = df.drop(['Inelasticity', 'Energy', 'Cascade', 'Zenith', 'X', 'Y', 'Z', 'Azimuth', 'Flavor', 'Charge'], axis=1)

    neurons = [
        [1028, 512, 256, 128, 64, 32, 16, 8, 4, 1],
        [2056, 1028, 256, 256, 128, 64, 32, 16, 8, 4, 1],
        # [5112, 2056, 1028, 256, 256, 128, 64, 32, 16, 8, 4, 1],
        # [5112, 2056, 1028, 256, 256, 128, 64, 32, 16, 8, 4, 1],
        [128, 128, 128, 128, 64, 32, 16, 8, 4, 1],
        [128, 128, 128, 128, 64, 32, 16, 8, 4, 1],
    ]

    activations = [
        ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'linear'],
        ['tanh', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'linear'],
        # ['tanh', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'linear'],
        # ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'linear'],
        ['tanh', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'linear'],
        ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'linear'],
    ]

    assert len(activations) == len(neurons), "neurons and activations don't have the same number of layers"

    for i in range(len(neurons)):
        assert len(activations[i]) == len(neurons[i]), f"layer {i} doesn't have the same number of neural layers and activation functions"

    for i in range(len(neurons)):

        if neural:
            model = NeuralNetwork(
                X, 
                y,
                neurons[i],
                activations[i],
                'Inelasticity',
                args={
                    'epochs'    : 100,
                    'save_fig'  : True,
                    'show'      : False,
                    'title'     : title
                }
            )

            model.build()

            model.train()

            model.test()

            model.plot()

        if ensemble:
            model = EnsembleNeuralNetwork(
                X, 
                y,
                neurons[i],
                activations[i],
                'Inelasticity',
                args={
                    'epochs'    : 100,
                    'save_fig'  : True,
                    'show'      : False,
                    'title'     : title
                }
            )

            model.build()

            while True:
                model.train_cascade()
                model.train_inelasticity()

                if model.min_inelasticity < 0.20:
                    break


            model.test()

            model.plot()
