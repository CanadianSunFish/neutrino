import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_process import *
from matplotlib.ticker import AutoMinorLocator

def dom_plot():

    # Reading original dom positions from dat file
    dat = '/home/bread/Documents/projects/neutrino/data/DOM_Position_GeoCalibDetectorStatus_2020.Run134142.Pass2_V0.dat'
    df = pd.read_csv(dat, delimiter=" ", header=None, names=['String', 'Count', 'X', 'Y', 'Z'], skiprows=[0])

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(df['X'], df['Y'], df['Z'], s=1, alpha=1)
    plt.savefig('/home/bread/Documents/projects/neutrino/data/fig/save/dom_positions.png')
    plt.show()

def true_energy_distribution():

    # df = pd.read_csv('/home/bread/Documents/projects/neutrino/data/model/events.csv')
    df = build_files("/home/bread/Documents/projects/neutrino/data/archive/data_colletion", 'output_label_names', 'labels')

    inelasticity = np.array(df['Cascade']/df['Energy'])

    energy = np.array(df['Energy'])

    energy_dist = []
    inelasticity_dist = []

    for i in range(len(energy)):
        if inelasticity[i] > 0.4 and inelasticity[i] <= 0.6:
            energy_dist.append(energy[i])
            inelasticity_dist.append(inelasticity[i])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(energy, bins=100, label=f'Energy', alpha=0.6, color='olive')
    ax.hist(inelasticity_dist, bins=np.arange(0, 3, 0.03), label=f'Inelasticity (0.4 - 0.6]', alpha=0.6, color='deepskyblue')
    ax.set_title(f"Energy Distribution When Associacted\n Inelasticity is in Range (0.4, 0.6] - 1d Histogram")
    ax.set_xlabel(f'Distribution')
    ax.set_ylabel('Count')
    # ax.set_ylim([0,3250])
    ax.grid('on', alpha=0.2, linestyle='-')
    plt.legend()
    plt.savefig(f'/home/bread/Documents/projects/neutrino/data/fig/save/energy_inelasticity_mid_1d.png')
    plt.show()
    plt.clf()

    plt.hist2d(energy_dist, inelasticity_dist, bins=30)
    plt.xlabel(f'Energy')
    plt.ylabel(f'Inelasticity')
    plt.title(f'Energy Distribution When Associacted\n Inelasticity is in Range (0.4, 0.6] - 2d Histogram')
    plt.savefig(f'/home/bread/Documents/projects/neutrino/data/fig/save/energy_inelasticity_mid_2d.png')
    plt.show()
    plt.clf()

def correlation_mapping():

    import seaborn as sns

    data = pd.read_csv('/home/bread/Documents/projects/neutrino/data/model/events.csv')

    inelasticity = np.array(data['Cascade']/data['Energy'])
    data.insert(0, 'Inelasticity', inelasticity)

    # data['TrackArcCos'] = np.arccos(data['Track'])
    data['TrackTimeCosh'] = np.cosh(data['Track'] / data['Time'])
    data['TrackTime'] = np.cos(data['Track'] * np.cos(data['Time']))

    data['Distance'] = np.sqrt(data['X']**2 + data['Y']**2 + data['Z']**2)

    # data['DistanceCos'] = np.cos(data['Distance']) * np.arccos(1/data['Zenith'])

    # data['AzimuthSin'] = 2 ** np.sin(data['Azimuth'])

    data = data.drop(['Flavor', 'Azimuth', 'IsTrack', 'IsAntineutrino', 'IsCC', 'Charge'], axis=1)

    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes()
    ax.set_facecolor('#faf0e6')
    fig.patch.set_facecolor('#faf0e6')
    cor = data.corr()
    sns.heatmap(cor, annot=True, cmap="Reds")
    plt.savefig('/home/bread/Documents/projects/neutrino/data/figures/temp/correlation_map.png')
    plt.show()
    plt.clf()

def plot_loss(history, path, title, log, save, show):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(history['mae'])
    ax.plot(history['val_mae'])
    if log:
        ax.set_yscale('log')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_title(f'Deep Neural Network Mean Absolute Error Training History\n For Reconstructing Muon Neutrino {title}', fontsize='x-small')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_xlabel('Epoch')
    ax.grid('on', alpha=0.2, linestyle='-')  
    ax.legend(['Training', 'Validating'], loc='upper left')
    if save:
        plt.savefig(f'{path}/nn_mae_history.png')
    if show:
        plt.show()
    plt.close()


def reconstructed_hist_2d(truth, reconstructed, path, title, save, show):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist2d(truth, reconstructed, bins=30)
    ax.set_xlabel(f'True {title}')
    ax.set_ylabel(f'Reconstructed {title}')
    ax.set_title(f'Using Deep Neural Network to Reconstruct\n Low Energy Muon Neutrino {title} - 2d Histogram', fontsize='small')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.grid('on', alpha=0.2, linestyle='-')  
    if save:
        plt.savefig(f'{path}/nn_{title}_2d.png')
    if show:
        plt.show()
    plt.close()

if __name__ == "__main__":

    # dom_plot()

    correlation_mapping()

    # true_energy_distribution()