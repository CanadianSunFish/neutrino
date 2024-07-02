import os
import keras
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from plotting import *
from keras import layers
from data_process import *
from matplotlib.ticker import AutoMinorLocator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


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
            'txt_title'     : 'Ensemble Model'
        }

        if args is not None:
            for key in args.keys():
                _args[key] = args[key]
        
        self.args = _args

        assert len(neurons) == len(activations), 'neurons and activation function lists must be the same length'

        self.neurons = neurons
        self.activations = activations
        self.title = title
        self.callbacks=[]

        features = self.args['scaler'].fit_transform(features)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(features, truth, test_size=0.2, random_state=42)

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

            model.add(keras.Input(shape=np.shape(self.X_train[0]), name='Input'))

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
            self.model_energy = keras.saving.load_model('src/models/model_energy.keras')

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

            self.model_energy = model
        
        self.model_energy.compile(
            optimizer=opt,
            loss=[self.args['loss']],
            metrics=[self.args['metrics']]
        )

        self.is_built = True

    def train_cascade(self):
        
        self.history_cascade = self.model_cascade.fit(
            self.X_train,
            self.y_train['Cascade'],
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

        self.min_cascade = min(self.history_cascade.history['val_mae'])

        return self.min_cascade

    def train_energy(self):

        self.history_energy = self.model_energy.fit(
            self.X_train,
            self.y_train['Energy'],
            validation_split=0.20,
            epochs=self.args['epochs'],
            batch_size=self.args['batch'],
            verbose=2,
            callbacks=self.callbacks
        )
        
        if self.args['save']:
            model.save(
                'src/models/model_energy.keras', 
                overwrite=self.args['overwrite']
            )

        if self.args['save_fig']:
            self.model_energy.save(
                f'{self.path}/model_energy.keras'
            )

        self.min_energy = min(self.history_energy.history['val_mae'])

        return self.min_energy

    def test(self):

        pred_energy = self.model_energy.predict(self.X_test)
        pred_cascade = self.model_cascade.predict(self.X_test)

        pred_energy = np.array([i[0] for i in pred_energy])
        pred_cascade = np.array([i[0] for i in pred_cascade])
            
        prediction = pred_cascade/pred_energy

        self.prediction = [i if i <= 1 else 1 for i in prediction]

        if self.args['save_fig']:
            epochs = self.args['epochs']
            batch = self.args['batch']
            opt = self.args['optimizer']
            lr = self.args['learn_rate']

            with open(f'{self.path}/model.txt', 'w') as f:
                title = self.args['txt_title']
                f.write(f'{title}\nmin energy, min cascade, neurons, activations, epochs, batch, opt, lr\n{self.min_energy}\n{self.min_cascade}\n{self.neurons}\n{self.activations}\n{epochs}\n{batch}\n{opt}\n{lr}')

    def eval(self):
        pass

    def plot(self):

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.plot(self.history_energy.history['mae'])
        plt.plot(self.history_energy.history['val_mae'])
        ax.grid('on', alpha=0.1)
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
        
        print(len(self.y_test['Inelasticity']))
        print(len(self.prediction))

        plt.hist2d(self.y_test['Inelasticity'], self.prediction, bins=30)
        plt.xlabel(f'True {self.title}')
        plt.ylabel(f'Predicted {self.title}')
        plt.title(f'Using Neural Network to Predict Low Energy\n Neutrino {self.title} - 2d Histogram')
        plt.xlim([0,1])
        plt.ylim([0,1])  
        if self.args['save_fig']:
            plt.savefig(f'{self.path}/nn_{self.title}_2d.png')
        if self.args['show']:
            plt.show()
        plt.clf()

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(self.y_test['Inelasticity'], bins=200, label=f'True {self.title}', alpha=0.6, color='olive')
        ax.hist(self.prediction, bins=200, label=f'Predicted {self.title}', alpha=0.6, color='deepskyblue')
        ax.set_title(f"Using Neural Network to Predict Low Energy\n Neutrino {self.title} - 1d Histogram")
        ax.set_xlabel(f'{self.title} Range')
        ax.set_ylabel('Count')
        ax.grid('on', alpha=0.2, linestyle='-')
        plt.legend()
        if self.args['save_fig']:
            plt.savefig(f'{self.path}/nn_{self.title}_1d.png')
        if self.args['show']:
            plt.show()
        plt.clf()

        truth = np.array(self.y_test['Inelasticity'])

        prediction_norm = []
        truth_norm = []

        for i in range(len(self.prediction)):
            if self.prediction[i] > 0.6 and self.prediction[i] < 1:
                prediction_norm.append(self.prediction[i])
                truth_norm.append(truth[i])

        plt.hist2d(truth_norm, prediction_norm, bins=30)
        plt.xlabel(f'True {self.title}')
        plt.ylabel(f'Predicted {self.title}')
        plt.title(f'Using Neural Network to Predict Low Energy\n Neutrino {self.title} - 2d Histogram')
        if self.args['save_fig']:
            plt.savefig(f'{self.path}/nn_{self.title}_high_values_2d.png')
        if self.args['show']:
            plt.show()
        plt.clf()

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(truth_norm, bins=200, label=f'True {self.title}', alpha=0.6, color='olive')
        ax.hist(prediction_norm, bins=200, label=f'Predicted {self.title}', alpha=0.6, color='deepskyblue')
        ax.set_title(f"Using Neural Network to Predict Low Energy\n Neutrino {self.title} - 1d Histogram")
        ax.set_xlabel(f'{self.title} Range')
        ax.set_ylabel('Count')
        ax.grid('on', alpha=0.2, linestyle='-')
        plt.legend()
        if self.args['save_fig']:
            plt.savefig(f'{self.path}/nn_{self.title}_high_values_1d.png')
        if self.args['show']:
            plt.show()
        plt.clf()

        prediction_norm = []
        truth_norm = []

        for i in range(len(self.prediction)):
            if self.prediction[i] > 0 and self.prediction[i] < 0.4:
                prediction_norm.append(self.prediction[i])
                truth_norm.append(truth[i])

        plt.hist2d(truth_norm, prediction_norm, bins=30)
        plt.xlabel(f'True {self.title}')
        plt.ylabel(f'Predicted {self.title}')
        plt.title(f'Using Neural Network to Predict Low Energy\n Neutrino {self.title} - 2d Histogram')
        if self.args['save_fig']:
            plt.savefig(f'{self.path}/nn_{self.title}_low_values_2d.png')
        if self.args['show']:
            plt.show()
        plt.clf()

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(truth_norm, bins=200, label=f'True {self.title}', alpha=0.6, color='olive')
        ax.hist(prediction_norm, bins=200, label=f'Predicted {self.title}', alpha=0.6, color='deepskyblue')
        ax.set_title(f"Using Neural Network to Predict Low Energy\n Neutrino {self.title} - 1d Histogram")
        ax.set_xlabel(f'{self.title} Range')
        ax.set_ylabel('Count')
        ax.grid('on', alpha=0.2, linestyle='-')
        plt.legend()
        if self.args['save_fig']:
            plt.savefig(f'{self.path}/nn_{self.title}_low_values_1d.png')
        if self.args['show']:
            plt.show()
        plt.clf()
        
        prediction_norm = []
        truth_norm = []

        for i in range(len(self.prediction)):
            if self.prediction[i] > 0.4 and self.prediction[i] < 0.6:
                prediction_norm.append(self.prediction[i])
                truth_norm.append(truth[i])

        plt.hist2d(truth_norm, prediction_norm, bins=30)
        plt.xlabel(f'True {self.title}')
        plt.ylabel(f'Predicted {self.title}')
        plt.title(f'Using Neural Network to Predict Low Energy\n Neutrino {self.title} - 2d Histogram')
        if self.args['save_fig']:
            plt.savefig(f'{self.path}/nn_{self.title}_middle_values_2d.png')
        if self.args['show']:
            plt.show()
        plt.clf()

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(truth_norm, bins=200, label=f'True {self.title}', alpha=0.6, color='olive')
        ax.hist(prediction_norm, bins=200, label=f'Predicted {self.title}', alpha=0.6, color='deepskyblue')
        ax.set_title(f"Using Neural Network to Predict Low Energy\n Neutrino {self.title} - 1d Histogram")
        ax.set_xlabel(f'{self.title} Range')
        ax.set_ylabel('Count')
        ax.grid('on', alpha=0.2, linestyle='-')
        plt.legend()
        if self.args['save_fig']:
            plt.savefig(f'{self.path}/nn_{self.title}_middle_values_1d.png')
        if self.args['show']:
            plt.show()
        plt.clf()
        
        return
    

if __name__ == "__main__":

    ensemble =  False
    neural = True
    feature = True
    analysis = False

    title = 'Non-Ensemble Model With Added Features (ZenithCos, ZenithSin, Distance, DistanceCos, TrackArcCos, TrackTimeCosh, TrackTime)'

    df = pd.read_csv('/home/bread/Documents/projects/neutrino/data/model/events.csv')

    # df = build_files("/home/bread/Documents/projects/neutrino/data/archive", 'output_label_names', 'labels')

    # df.to_csv('/home/bread/Documents/projects/neutrino/data/model/events.csv', index=False)

    inelasticity = df['Cascade']/df['Energy']
    df.insert(0, 'Inelasticity', inelasticity)

    if feature:
        zenithCos = np.cos(df['Zenith']) * df['Track']
        zenithSin = np.sin(df['Zenith']) * df['Track']
        df['ZenithCos'] = zenithCos
        df['ZenithSin'] = zenithSin

        df['Distance'] = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2)

        # df['TrackArcCos'] = np.arccos(df['Track'])
        df['TrackTimeCosh'] = np.cosh(df['Track'] / df['Time'])
        df['TrackTime'] = np.cos(df['Track'] * np.cos(df['Time']))

    if ensemble:
        y = df[['Inelasticity', 'Cascade', 'Energy']]
    else:
        y = df[['Inelasticity', 'Energy']]

    # X = df.drop(['Inelasticity', 'Energy', 'Cascade', 'Zenith', 'X', 'Y', 'Z', 'Azimuth', 'Flavor', 'Charge'], axis=1)
    X = np.array(df.drop(['Inelasticity', 'Energy', 'Cascade'], axis=1))

    neurons = [
        [24, 16, 12, 8, 6, 4, 2, 1],
        [1028, 512, 256, 128, 64, 32, 16, 8, 4, 1],
        [2056, 1028, 256, 256, 128, 64, 32, 16, 8, 4, 1],
        [5012, 2056, 1028, 256, 256, 128, 64, 32, 16, 8, 4, 1],
        [2056, 2056, 1028, 256, 256, 128, 64, 32, 16, 8, 4, 1],
        # [5112, 2056, 1028, 256, 256, 128, 64, 32, 16, 8, 4, 1],
        # [5112, 2056, 1028, 256, 256, 128, 64, 32, 16, 8, 4, 1],
        [128, 128, 128, 128, 64, 32, 16, 8, 4, 1],
        [128, 128, 128, 128, 64, 32, 16, 8, 4, 1],
    ]

    activations = [
        ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu'],
        ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'sigmoid'],
        ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'sigmoid'],
        ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'sigmoid'],
        # ['tanh', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'linear'],
        # ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'linear'],
        ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'linear'],
        ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'linear'],
    ]


    assert len(activations) == len(neurons), "neurons and activations don't have the same number of layers"

    for i in range(len(neurons)):
        assert len(activations[i]) == len(neurons[i]), f"layer {i} doesn't have the same number of neural layers and activation functions"

    for i in range(len(neurons)):

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