"""A set of functions for reading density histograms."""

import pathlib

import numpy as np


def read_density_histogram(path):
    """Read a density histogram from a pzhist*.dat file.

    The subensembles and frequencies are read from the pzcnt.dat file
    in the same directory as the pzhist file.

    Args:
        path: The location of the density histogram file.

    Returns:
        A dict with the keys 'subensemble', 'distance', 'density',
        and 'frequency'.
    """
    histogram = np.transpose(np.loadtxt(path))
    parent = pathlib.Path(path).parent
    subensemble, frequency = np.loadtxt(parent / 'pzcnt.dat', unpack=True)

    return {'subensemble': subensemble, 'distance': histogram[1],
            'density': histogram[2:], 'frequency': frequency}


def combine_density_histogram_runs(path, runs, hist_file):
    """Create a density histogram by averaging over several runs.

    Args:
        path: The root directory for the data.
        runs: The list of runs to combine.
        hist_file: The name of the density histogram file to combine.

    Returns:
        A dict with the keys 'subensemble', 'distance', 'density',
        and 'frequency'.
    """
    true_path = pathlib.Path(path)
    histograms = [read_density_histogram(true_path / run / hist_file)
                  for run in runs]
    frequency_sum = sum([h['frequency'] for h in histograms])
    weighted = sum(h['frequency'] * np.transpose(h['density'])
                   for h in histograms)
    nonzero = np.nonzero(frequency_sum)
    density = np.zeros(np.transpose(weighted).shape)
    density[nonzero] = np.transpose(weighted[nonzero] / frequency_sum[nonzero])

    return {'subensemble': histograms[0]['subensemble'],
            'distance': histograms[0]['distance'],
            'density': density, 'frequency': frequency_sum}
