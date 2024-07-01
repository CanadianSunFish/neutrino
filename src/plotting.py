import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_process import *

def dom_plot():
    # Reading simulation data from hdf5 file
    hdf5 = '/home/bread/Documents/projects/neutrino/data/hdf5/NuMu_genie_149999_030000_level6.zst_cleanedpulses_transformed_IC19.hdf5'
    hdf5_df = get_hdf5(hdf5, 'output_label_names', 'labels')

    # Reading original dom positions from dat file
    dat = '/home/bread/Documents/projects/neutrino/data/DOM_Position_GeoCalibDetectorStatus_2020.Run134142.Pass2_V0.dat'
    df = pd.read_csv(dat, delimiter=" ", header=None, names=['String', 'Count', 'X', 'Y', 'Z'], skiprows=[0])

    # Reading low energy database file
    db = '/home/bread/Documents/projects/neutrino/data/db/oscNext_genie_level5_v02.00_pass2.141122.000000.db'
    pulse, truth = get_db(db)

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(df['X'], df['Y'], df['Z'], s=1, alpha=1)
    ax.scatter(pulse['dom_x'], pulse['dom_y'], pulse['dom_z'], s=1, alpha=1)
    plt.savefig('/home/bread/Documents/projects/neutrino/data/fig/DomPositionAndHits.png')
    plt.show()

def true_energy_distribution():

    df = pd.read_csv('/home/bread/Documents/projects/neutrino/data/csv/built.csv')

    inelasticity = np.array(df['Cascade']/df['Energy'])

    energy = np.array(df['Energy'])

    energy_dist = []
    inelasticity_dist = []

    for i in range(len(energy)):
        if inelasticity[i] > 0.4 and inelasticity[i] <= 0.6:
            energy_dist.append(energy[i])
            inelasticity_dist.append(inelasticity[i])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(energy_dist, bins=100, label=f'Energy', alpha=0.6, color='olive')
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

    data = pd.read_csv('/home/bread/Documents/projects/neutrino/data/csv/built.csv')

    inelasticity = np.array(data['Cascade']/data['Energy'])
    data.insert(0, 'Inelasticity', inelasticity)

    data['AzimuthCos'] = 2 ** np.cos(data['Azimuth'])
    data['AzimuthSin'] = 2 ** np.sin(data['Azimuth'])

    data = data.drop(['Flavor', 'Azimuth', 'IsTrack', 'IsAntineutrino', 'IsCC', 'Charge'], axis=1)

    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes()
    ax.set_facecolor('#faf0e6')
    fig.patch.set_facecolor('#faf0e6')
    cor = data.corr()
    sns.heatmap(cor, annot=True, cmap="Reds")
    plt.savefig('data/fig/correlation_map.png')
    plt.show()
    plt.clf()

    sns.pairplot(data, kind='reg')
    plt.show()

if __name__ == "__main__":

    correlation_mapping()

    # true_energy_distribution()