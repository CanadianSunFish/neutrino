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
            'txt_title'     : 'Non-Ensemble Model'
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
            directory = '/home/bread/Documents/projects/neutrino/data/figures/review/figures'
            pathdir = pathlib.Path(directory)
            labels = []

            for path in pathdir.iterdir():
                labels.append(int(str(path).split('/')[-1]))

            new_dir = f"{max(labels)+1}"
            parent_dir = "/home/bread/Documents/projects/neutrino/data/figures/review/figures/"

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
                title = self.args['txt_title']
                f.write(f'{title}\nmin mae, neurons, activations, epochs, batch, opt, lr\n{self.min_mae}\n{self.neurons}\n{self.activations}\n{epochs}\n{batch}\n{opt}\n{lr}')

    def eval(self):
        pass

    def plot(self):

        plot_loss(self.history.history, self.path, self.title, self.args['loss'], self.args['save_fig'], self.args['show'])

        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # plt.plot(self.history.history['mae'])
        # plt.plot(self.history.history['val_mae'])
        # if self.args['log']:
        #     ax.set_yscale('log')
        # ax.xaxis.set_minor_locator(AutoMinorLocator())
        # plt.title(f'Neural Network Mean Absolute Error - Training For {self.title}')
        # plt.ylabel('Mean Absolute Error')
        # plt.xlabel('Epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        # if self.args['save_fig']:
        #     plt.savefig(f'{self.path}/nn_mae_history.png')
        # if self.args['show']:
        #     plt.show()
        # plt.close()

        reconstructed_hist_2d(self.y_test, self.prediction, self.path, self.title, self.args['save_fig'], self.args['show'])
        
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
        plt.close()

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
        plt.close()

        true_e_cut = []
        prediction_cut = []

        for i in range(len(self.true_e)):
            if self.prediction[i] > 0.4 and self.prediction[i] < 0.6:
                true_e_cut.append(self.true_e[i])
                prediction_cut.append(self.prediction[i])

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(true_e_cut, bins=50, label=f'True Energy', alpha=0.6, color='olive')
        ax.hist(prediction_cut, bins=50, label=f'Reconstructed Inelasticity (0.4, 0.6)', alpha=0.6, color='deepskyblue')
        ax.set_title(f"Reconstructed Inelasticity Cut in Range (0.4, 0.6)\n And Associated Energy Values - Histogram")
        ax.grid('on', alpha=0.2, linestyle='-')
        plt.legend()
        if self.args['save_fig']:
            plt.savefig(f'{self.path}/nn_{self.title}_true_energy_cut_1d.png')
        if self.args['show']:
            plt.show()
        plt.close()

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.hist2d(true_e_cut, prediction_cut, bins=30)
        ax.set_ylabel("Reconstructed Inelasticity Cut")
        ax.set_xlabel("Associated True Energy")
        ax.set_title(f"Reconstructed Inelasticity Cut in Range (0.4, 0.6)\n And Associated Energy Values - 2d Histogram")
        ax.grid('on', alpha=0.2, linestyle='-')
        if self.args['save_fig']:
            plt.savefig(f'{self.path}/nn_{self.title}_true_energy_cut_2d.png')
        if self.args['show']:
            plt.show()
        plt.close()

        truth = np.array(self.y_test)

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
        plt.close()

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(1, 1, 1)
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
        plt.close()

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
        plt.close()

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
        plt.close()
        
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
        plt.close()

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
        plt.close()
        

        return


if __name__ == "__main__":

    feature = True

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

    y = df[['Inelasticity', 'Energy']]

    # X = df.drop(['Inelasticity', 'Energy', 'Cascade', 'Zenith', 'X', 'Y', 'Z', 'Azimuth', 'Flavor', 'Charge'], axis=1)
    X = np.array(df.drop(['Inelasticity', 'Energy', 'Cascade'], axis=1))

    neurons = [
        [24, 16, 12, 8, 6, 4, 2, 1],
        [1028, 512, 256, 128, 64, 32, 16, 8, 4, 1],
        [2056, 1028, 256, 256, 128, 64, 32, 16, 8, 4, 1],
        [5012, 2056, 1028, 256, 256, 128, 64, 32, 16, 8, 4, 1],
        [2056, 2056, 1028, 256, 256, 128, 64, 32, 16, 8, 4, 1],
    ]

    activations = [
        ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu'],
        ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'sigmoid'],
        ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'sigmoid'],
        ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'sigmoid'],
    ]

    model = NeuralNetwork(
                X, 
                y,
                neurons[3],
                activations[3],
                'Inelasticity',
                args={
                    'epochs'    : 45,
                    'save_fig'  : True,
                    'show'      : False,
                    'show'      : True,
                    'txt_title' : title
                }
            )

    model.build()

    model.train()

    model.test()

    model.plot()
