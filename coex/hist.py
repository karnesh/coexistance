"""Objects and functions for working with visited states histograms.

Histograms are used to find the change in free energy via histogram
reweighting.  For example, an energy visited states distribution would
provide a new free energy given a change in the inverse temperature
beta (1/kT).
"""

import pathlib

import numpy as np


# The conversion factor for cubic angstroms -> cubic meters.
CUBIC_METERS = 1.0e-30


class Subhistogram:
    """A frequency distribution for a given property (energy, volume,
    molecule count) in a single subensemble of a simulation.

    Attributes:
        bins: A numpy array with the values of the property.
        counts: An array with the number of times each value was
            visited in the simulation.
    """

    def __init__(self, bins, counts):
        self.bins = bins
        self.counts = counts

    def __len__(self):
        return len(self.bins)

    def __str__(self):
        return "bins: {}, counts: {}\n".format(self.bins, self.counts)

    def __repr__(self):
        return str(self)

    def reweight(self, amount):
        """Get the change in free energy due to histogram reweighting.

        Args:
            amount: The difference in the relevant property.

        Returns:
            The change in free energy as a float.
        """
        shifted = self._shift(amount)

        return np.log(sum(shifted)) - np.log(sum(self.counts))

    def _shift(self, amount):
        """Shift the histogram by a given difference."""
        return self.counts * np.exp(-amount * self.bins)

    def average(self, weight=None):
        """Calculate the weighted average of the histogram.

        Args:
            weight: The optional weight to use.

        Returns:
            The weighted average as a float.
        """
        try:
            if weight is None:
                avg = sum(self.counts*self.bins) / sum(self.counts)
            else:
                shifted = self._shift(weight)
                avg = sum(self.bins*shifted) / sum(shifted)
        except ZeroDivisionError:
            return 0.0

        return avg


class Histogram(object):
    """A visited states histogram for a given property.

    Attributes:
        subhists: A list of Subhistogram objects.
    """

    def __init__(self, subhists):
        self.subhists = subhists

    def __getitem__(self, index):
        return self.subhists[index]

    def __iter__(self):
        for subhist in self.subhists:
            yield subhist

    def __len__(self):
        return len(self.subhists)

    def average(self, weights=None):
        """Calculate the weighted average of the histogram.

        Args:
            weights: A list of weights for each subensemble.

        Returns:
            A numpy array with the weighted average of each
            subensemble-specific histogram.
        """
        if weights is None:
            return np.array([sh.average(weight=None) for sh in self])

        return np.array([sh.average(weights[i]) for i, sh in enumerate(self)])

    def write(self, path):
        """Write a histogram to a pair of hist and lim files.

        Args:
            path: The name of the *hist*.dat file to write.
        """
        is_vhist = 'vhist' in pathlib.Path(path).name
        most_sampled = 0
        step = self[-1].bins[1] - self[-1].bins[0]
        if is_vhist:
            step /= CUBIC_METERS

        with open(_get_limits_path(path), 'w') as lim_file:
            for i, subhist in enumerate(self):
                sampled = len(subhist)
                most_sampled = max(most_sampled, sampled)
                min_bin = np.amin(subhist.bins)
                max_bin = np.amax(subhist.bins)
                if is_vhist:
                    max_bin /= CUBIC_METERS
                    min_bin /= CUBIC_METERS

                format_str = '{:8d} {:7d} {:15.7e} {:15.7e} {:15.7e}'
                print(format_str.format(i, sampled, min_bin, max_bin, step),
                      file=lim_file)

        raw_hist = np.zeros([most_sampled, len(self)+1])
        raw_hist[:, 0] = range(1, most_sampled+1)
        for i, subhist in enumerate(self):
            sampled = len(subhist)
            raw_hist[0:sampled, i+1] = subhist.counts

        np.savetxt(path, raw_hist, fmt='%8d', delimiter='  ')

    def normalize_volume_units(self, exponentiate=False):
        """Adjust the bin units for a volume histogram.

        Args:
            exponentiate: True if the volume is given as log(V).
        """
        for subhist in self.subhists:
            if exponentiate:
                subhist.bins = np.exp(subhist.bins)

            subhist.bins *= CUBIC_METERS


def _get_limits_path(hist_file):
    """Figure out the appropriate file name for the histogram limits
    file.

    Args:
        hist_file: The path to the histogram file.

    Returns:
        A string with the path to the limits file.
    """
    hist_path = pathlib.Path(hist_file)
    return hist_path.parent / hist_path.name.replace('hist', 'lim')


def read_histogram(path):
    """Read a histogram from a pair of files.

    This method accepts the location of the raw histogram file, e.g.,
    ehist.dat and parses the appropriate limits file (here, elim.dat)
    in the same directory.

    Args:
        path: The location of the raw histogram data.

    Returns:
        A Histogram object.
    """
    raw_hist = np.transpose(np.loadtxt(path))[1:]
    limits = np.loadtxt(_get_limits_path(path))

    def create_subhist(subensemble, line):
        """Parse the histogram limits for a given subensemble."""
        _, size, lower, upper, _ = line
        size = int(size)
        bins = np.linspace(lower, upper, size)
        if len(raw_hist.shape) == 1:
            counts = np.array([raw_hist[subensemble]])
        elif size == 0:
            bins = np.zeros(1)
            counts = np.zeros(1).astype('uint64')
        else:
            counts = raw_hist[subensemble][0:size]

        if 'nhist' in pathlib.Path(path).name:
            bins = bins.astype('uint64')

        return Subhistogram(bins=bins, counts=counts.astype('uint64'))

    return Histogram([create_subhist(subensemble, line)
                      for subensemble, line in enumerate(limits)])


def read_volume_histogram(path, exponentiate=False):
    """Read a volume histogram from a vhist.dat file.

    Args:
        path: The location of the histogram data.
        expontentiate: A bool; True if the lim file uses log(V) bins
            instead of volume bins.
    """
    hist = read_histogram(path)
    hist.normalize_volume_units(exponentiate)

    return hist


def combine_histograms(histograms):
    """Combine a series of histograms.

    Args:
        histograms: A list of histograms.

    Returns:
        A Histogram with the combined data.
    """
    subensembles = len(histograms[0])
    dtype = histograms[0][0].bins.dtype
    step = 1
    for sub in histograms[0]:
        if len(sub) > 1:
            step = sub.bins[1] - sub.bins[0]
            break

    def combine_subensemble(i):
        """Combine the subhistograms for the given subensemble."""
        min_bin = min([h[i].bins[0] for h in histograms])
        max_bin = max([h[i].bins[-1] for h in histograms])
        num = int(np.round((max_bin-min_bin) / step + 1))
        bins = np.linspace(min_bin, max_bin, num, dtype=dtype)
        counts = np.zeros(num, dtype=dtype)
        for hist in histograms:
            shift = int(np.round((hist[i].bins[0]-min_bin) / step))
            counts[shift : shift+len(hist[i])] += hist[i].counts

        return Subhistogram(bins=bins, counts=counts)

    return Histogram([combine_subensemble(i) for i in range(subensembles)])


def combine_histogram_runs(path, runs, hist_file):
    """Combine histograms across a series of production runs.

    Args:
        path: The location of the production runs.
        runs: The list of runs to combine.
        hist_file: The name of the histogram to combine.

    Returns:
        A Histogram with the combined data.
    """
    true_path = pathlib.Path(path)
    return combine_histograms([read_histogram(true_path / run / hist_file)
                               for run in runs])


def combine_volume_histogram_runs(path, runs, exponentiate=False):
    """Combine volume histograms across a series of production runs.

    Args:
        path: The location of the production runs.
        runs: The list of runs to combine.
        exponentiate: A bool; True if the lim file uses ln(V) bins
            instead of volume bins.

    Returns:
        A Histogram with the combined data.
    """
    hist = combine_histogram_runs(path, runs, hist_file='vhist.dat')
    hist.normalize_volume_units(exponentiate)

    return hist


def read_all_n_histograms(path):
    """Read all of the molecule number histograms in a directory.

    Args:
        path: The directory containing the the nhist and nlim files to
            read.

    Returns:
        A sorted list of Histogram objects.
    """
    true_path = pathlib.Path(path)
    return [read_histogram(f) for f in sorted(true_path.glob("nhist_??.dat"))]


def combine_all_n_histograms(path, runs):
    """Combine all the molecule number histograms across a set of runs.

    Args:
       path: The base path containing the data to combine.
       runs: The list of runs to combine.

    Returns:
        A list of combined histograms, with one entry for each species.
    """
    hist_paths = (pathlib.Path(path) / runs[0]).glob('nhist_*.dat')
    hist_files = sorted(f.name for f in hist_paths)
    return [combine_histogram_runs(path, runs, f) for f in hist_files]
