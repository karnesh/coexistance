"""Objects and functions for working with probability distributions and
related properties.

Internally, we often deal with the logarithm of the probability
distribution along a path of interest instead of the free energy
differences, which differ only by a minus sign.

In gchybrid, we refer to the logarithm of the order parameter
distribution as lnpi_op.dat, the logarithm of the growth expanded
ensemble distribution as lnpi_tr.dat, the logarithm of the exchange path
distribution as lnpi_ex.dat, and the logarithm of the regrowth path
distribution as lnpi_rg.dat.
"""

import copy
import pathlib

import numpy as np
import scipy.integrate


class TransitionMatrix:
    """A base class for transition matrices.

    Attributes:
        attempts: An array with the number of forward and reverse
            transition attempts for each state.
        probabilities: An array with the acceptance probability for
            forward and reverse transitions from each state.
    """
    def __init__(self, attempts, probabilities):
        self.attempts = attempts
        self.probabilities = probabilities

    def state_sampled(self, fwd, rev, min_attempts):
        """Check if a given state is sampled, i.e., the forward and
        reverse probabilities are greater than 0.0 and the number of
        forward and reverse attempts are greater than some threshold.

        Args:
            fwd: A numpy index describing the forward transition.
            rev: A numpy index describing the reverse transition.
            min_attempts: The minimum number of transition attempts
                required.

        Returns:
            True if the sampling criteria for the state are met.
        """
        return (self.attempts[fwd, 1] > min_attempts
                and self.attempts[rev, 0] > min_attempts
                and self.probabilities[fwd, 1] > 0.0
                and self.probabilities[rev, 0] > 0.0)

    def log_probability(self, fwd, rev):
        """Find the log of the probability of a state.

        Args:
            fwd: A numpy index describing the forward transition.
            rev: A numpy index describing the reverse transition.

        Returns:
            A float.
        """
        return np.log(self.probabilities[fwd, 1] / self.probabilities[rev, 0])


class OrderParameterMatrix(TransitionMatrix):
    """An acceptance probability matrix along the order parameter path.

    Attributes:
        subensembles: A numpy array with the order parameter values.
        attempts: An array with the number of forward and reverse
            transition attempts for each state.
        probabilities: An array with the acceptance probability for
            forward and reverse transitions from each state.
    """
    def __init__(self, subensembles, attempts, probabilities):
        self.subensembles = subensembles
        super().__init__(attempts, probabilities)

    def __len__(self):
        return len(self.subensembles)

    def order_parameter_distribution(self, guess, min_attempts=1):
        """Calculate the free energy of the order parameter path.

        Args:
            guess: A numpy array or OrderParameterDistribution with an
                initial guess for the free energy.
            min_attempts: The minimum number of transitions in each
                direction required to consider the transition matrix
                when updating the free energy estimate.

        Returns:
            An OrderParameterDistribution.
        """
        dist = np.zeros(len(self))
        for i, difference in enumerate(np.diff(guess)):
            dist[i+1] = dist[i] + difference
            if self.state_sampled(i, i+1, min_attempts):
                dist[i+1] += self.log_probability(i, i+1)

        return OrderParameterDistribution(order_parameter=self.subensembles,
                                          log_probability=dist)

    def write(self, path):
        """Write the transition matrix to a file.

        Args:
            path: The location of the file to write.
        """
        fmt = 3*['%8d'] + 2*['%.11e']
        arr = np.column_stack((self.subensembles, self.attempts,
                               self.probabilities))
        np.savetxt(path, arr, fmt=fmt, delimiter=' ')


class GrowthMatrix(TransitionMatrix):
    """An acceptance probability matrix along the molecule growth
    path.

    Attributes:
        attempts: An array with the number of forward and reverse
            transition attempts for each state.
        probabilities: An array with the acceptance probability for
            forward and reverse transitions from each state.
    """
    def __init__(self, index, attempts, probabilities):
        self.subensembles = index['subensembles']
        self.components = index['components']
        self.stages = index['stages']
        super().__init__(attempts, probabilities)

    def __len__(self):
        return len(self.subensembles)

    def _unique_subensembles(self):
        """Generate a list of the unique subensembles."""
        return np.unique(self.subensembles)

    def _unique_components(self):
        """Generate a list of the unique components."""
        return np.unique(self.components)

    def growth_distribution(self, guess, min_attempts=1):
        """Calculate the free energy of the order parameter path.

        Args:
            guess: A numpy array or GrowthDistribution with an
                initial guess for the free energy along the molecule
                transfer path.
            min_attempts: The minimum number of transitions in each
                direction required to consider the transition matrix
                when updating the free energy estimate.

        Returns:
            A GrowthDistribution.
        """
        dist = GrowthDistribution(index=_create_growth_index(self),
                                  log_probability=np.zeros(len(self)))
        for cmp in self._unique_components():
            for sub in self._unique_subensembles():
                sel = (self.components == cmp) & (self.subensembles == sub)
                if len(self.stages[sel]) == 1:
                    continue

                for stg in self.stages[sel][-2::-1]:
                    curr = sel & (self.stages == stg)
                    succ = sel & (self.stages == stg+1)
                    dist[curr] = dist[succ] + guess[curr] - guess[succ]
                    if self.state_sampled(curr, succ, min_attempts):
                        dist[curr] -= self.log_probability(curr, succ)

        return dist

    def order_parameter_distribution(self, tr_guess, op_guess, species=1,
                                     min_attempts=1):
        """Calculate the free energy of the order parameter path using
        the transfer path of the order parameter species.

        This method is only applicable for direct simulations.

        Args:
            tr_guess: A numpy array or GrowthDistribution with an
                initial guess for the free energy along the molecule
                transfer path.
            op_guess: A numpy array or OrderParameterDistribution with
                an initial guess for the free energy along the order
                parameter path.
            species: The order parameter species.
            min_attempts: The minimum number of transitions in each
                direction required to consider the transition matrix
                when updating the free energy estimate.

        Returns:
            An OrderParameterDistribution.
        """
        dist = OrderParameterDistribution(
            order_parameter=self._unique_subensembles(),
            log_probability=np.zeros(len(self._unique_subensembles())))
        growth_dist = self.growth_distribution(tr_guess)
        for i, sub in enumerate(dist.order_parameter[1:]):
            dist[i+1] = dist[i]
            fwd_sub = ((self.components == species)
                       & (self.subensembles == sub-1))
            rev_sub = ((self.components == species)
                       & (self.subensembles == sub))
            fwd_full = (fwd_sub
                        & (self.stages == np.amax(self.stages[fwd_sub])))
            rev_first = rev_sub & (self.stages == 1)
            path_sampled = True
            if self.state_sampled(fwd_full, rev_first, min_attempts):
                dist[i+1] += self.log_probability(fwd_full, rev_first)
            else:
                path_sampled = False

            for stg in self.stages[rev_sub][1:]:
                prev = rev_sub & (self.stages == stg-1)
                curr = rev_sub & (self.stages == stg)
                if (self.state_sampled(prev, curr, min_attempts)
                        and path_sampled):
                    dist[i+1] += growth_dist[curr] - growth_dist[prev]
                else:
                    path_sampled = False

            if path_sampled:
                dist[i+1] += tr_guess[rev_first] - tr_guess[fwd_full]
            else:
                dist[i+1] += op_guess[i+1] - op_guess[i]

        return dist

    def write(self, path):
        """Write the transition matrix to a file.

        Args:
            path: The location of the file to write.
        """
        fmt = 6*['%8d'] + 2*['%.11e']
        arr = np.column_stack((np.arange(len(self))+1,
                               _create_growth_index(self), self.attempts,
                               self.probabilities))
        np.savetxt(path, arr, fmt=fmt, delimiter=' ')


def _read_growth_index(path):
    """Parse the first four columns of an {lnpi,pacc}_tr* file."""
    names = ['subensembles', 'components', 'stages']
    values = np.transpose(np.loadtxt(path, usecols=(1, 2, 3), dtype='int'))

    return {name: value for name, value in zip(names, values)}


def _create_growth_index(obj):
    """Package the state information into a dict."""
    return {'subensembles': obj.subensembles,
            'components': obj.components, 'stages': obj.stages}


def read_matrix(path):
    """Read a pacc_op_*.dat file or a pacc_tr_*.dat file.

    Args:
        path: The location of the file to read.

    Returns:
        A TransitionMatrix object.
    """
    name = pathlib.Path(path).name
    if 'tr' in name:
        index = _read_growth_index(path)
        values = np.loadtxt(path, usecols=(4, 5, 6, 7))
        attempts = values[:, :2].astype('int')
        probabilities = values[:, 2:]
        return GrowthMatrix(index, attempts, probabilities)
    elif 'op' in name:
        values = np.loadtxt(path, usecols=(0, 1, 2, 3, 4))
        index = values[:, 0].astype('int')
        attempts = values[:, 1:3].astype('int')
        probabilities = values[:, 3:]
        return OrderParameterMatrix(index, attempts, probabilities)
    raise NotImplementedError


def combine_matrices(matrices):
    """Combine a set of transition matrices.

    Args:
        matrices: A list of TransitionMatrix-like objects to combine.

    Returns:
        An instance of an appropriate subclass of TransitionMatrix with
        the combined data.
    """
    attempts = np.sum([m.attempts for m in matrices], 0)
    weighted_sum = np.sum([m.attempts * m.probabilities
                           for m in matrices], 0)
    nonzero = np.nonzero(attempts)
    probabilities = np.zeros(attempts.shape)
    probabilities[nonzero] = weighted_sum[nonzero] / attempts[nonzero]
    if isinstance(matrices[0], GrowthMatrix):
        return GrowthMatrix(_create_growth_index(matrices[0]), attempts,
                            probabilities)
    elif isinstance(matrices[0], OrderParameterMatrix):
        return OrderParameterMatrix(matrices[0].subensembles, attempts,
                                    probabilities)
    raise NotImplementedError


def combine_matrix_runs(path, runs, pacc_file):
    """Combine a set of transition matrix files.

    Args:
        path: The base path containing the data to combine.
        runs: The list of runs to combine.
        pacc_file: The name of the file to combine.

    Returns:
        A TransitionMatrix object with the combined data.
    """
    true_path = pathlib.Path(path)
    return combine_matrices([read_matrix(true_path / run / pacc_file)
                             for run in runs])


class OrderParameterDistribution:
    """The logarithm of the probability distribution along the order
    parameter path.

    Attributes:
        order_parameter: A numpy array with the order parameter values.
        log_probability: An array with the logarithm of the
            probability distribution.
    """

    def __init__(self, order_parameter, log_probability):
        self.order_parameter = order_parameter
        self.log_probability = log_probability

    def __add__(self, other):
        if isinstance(other, OrderParameterDistribution):
            return self.log_probability + other.log_probability
        elif isinstance(other, (np.ndarray, int, float)):
            return self.log_probability + other
        raise NotImplementedError

    def __iadd__(self, other):
        if isinstance(other, OrderParameterDistribution):
            self.log_probability += other.log_probability
        elif isinstance(other, (np.ndarray, int, float)):
            self.log_probability += other
        else:
            raise NotImplementedError

    def __sub__(self, other):
        if isinstance(other, OrderParameterDistribution):
            return self.log_probability - other.log_probability
        elif isinstance(other, (np.ndarray, int, float)):
            return self.log_probability - other
        raise NotImplementedError

    def __isub__(self, other):
        if isinstance(other, OrderParameterDistribution):
            self.log_probability -= other.log_probability
        elif isinstance(other, (np.ndarray, int, float)):
            self.log_probability -= other
        else:
            raise NotImplementedError

    def __mul__(self, other):
        if isinstance(other, OrderParameterDistribution):
            return self.log_probability * other.log_probability
        elif isinstance(other, (np.ndarray, int, float)):
            return self.log_probability * other
        raise NotImplementedError

    def __imul__(self, other):
        if isinstance(other, OrderParameterDistribution):
            self.log_probability *= other.log_probability
        elif isinstance(other, (np.ndarray, int, float)):
            self.log_probability *= other
        else:
            raise NotImplementedError

    def __div__(self, other):
        if isinstance(other, OrderParameterDistribution):
            return self.log_probability / other.log_probability
        elif isinstance(other, (np.ndarray, int, float)):
            return self.log_probability / other
        raise NotImplementedError

    def __idiv__(self, other):
        if isinstance(other, OrderParameterDistribution):
            self.log_probability /= other.log_probability
        elif isinstance(other, (np.ndarray, int, float)):
            self.log_probability /= other
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.log_probability)

    def __getitem__(self, i):
        return self.log_probability[i]

    def __setitem__(self, i, value):
        self.log_probability[i] = value

    def __iter__(self):
        for prob in self.log_probability:
            yield prob

    def smooth(self, order, denominator=None, drop=None):
        """Perform curve fitting on the free energy differences to
        produce a new estimate of the free energy.

        Args:
            order: The order of the polynomial used to fit the free
                energy differences.
            denominator: If present, smooth the differences in free
                energy relative to this array. Useful for, e.g.,
                smoothing relative to beta in TEE simulations.
            drop: A boolean numpy array denoting whether to drop each
                subensemble prior to fitting.

        Returns:
            An OrderParameterDistribution with the new estimate for the
            free energy.
        """
        if drop is None:
            drop = np.tile(False, len(self)-1)
        else:
            drop = drop[1:]

        if denominator is None:
            denominator = self.order_parameter

        differences = (np.diff(self) / np.diff(denominator))[~drop]
        fit = np.poly1d(np.polyfit(denominator[1:][~drop], differences,
                                   order))
        smoothed = scipy.integrate.cumtrapz(y=fit(denominator), x=denominator,
                                            initial=0.0)

        return OrderParameterDistribution(order_parameter=self.order_parameter,
                                          log_probability=smoothed)

    def split(self, split=0.5):
        """Split the distribution into two parts.

        Args:
            split: The fraction of the length to use as the boundary for
                the two parts.

        Returns:
            A tuple of OrderParameterDistribution objects.
        """
        bound = int(split * len(self))
        param, logp = self.order_parameter, self.log_probability
        fst = OrderParameterDistribution(order_parameter=param[:bound],
                                         log_probability=logp[:bound])
        snd = OrderParameterDistribution(order_parameter=param[bound:],
                                         log_probability=logp[bound:])

        return fst, snd

    def write(self, path):
        """Write the distribution to a file.

        Args:
            path: The name of the file to write.
        """
        np.savetxt(path, np.column_stack((self.order_parameter,
                                          self.log_probability)),
                   fmt=['%8d', '%.11e'])


class GrowthDistribution:
    """The logarithm of the probability distribution along the
    molecule growth path.

    Attributes:
        index: A dict with the overall number, molecule, subensemble,
            and growth stage of each state in the path.
        log_probability: An array with the logarithm of the
            probability distribution.
    """

    def __init__(self, index, log_probability):
        self.subensembles = index['subensembles']
        self.components = index['components']
        self.stages = index['stages']
        self.log_probability = log_probability

    def __len__(self):
        return len(self.log_probability)

    def __getitem__(self, i):
        return self.log_probability[i]

    def __setitem__(self, i, value):
        self.log_probability[i] = value

    def shift_by_order_parameter(self, op_dist):
        """Add the order parameter free energies to transfer path
        free energies.

        This is the form that gchybrid normally outputs: each
        subensemble's transfer path free energies are relative to that
        subensemble's order parameter free energy.

        Args:
            op_dist: An OrderParameterDistribution object with the free
                energies to shift by.

        Returns:
            A new GrowthDistribution with shifted free energies.
        """
        shifted = copy.deepcopy(self)
        for i, log_prob in enumerate(op_dist):
            shifted[shifted.subensembles == i] += log_prob

        return shifted

    def smooth(self, order, drop=None):
        """Perform curve fitting on the free energy differences to
        produce a new estimate of the free energy.

        Args:
            order: The order of the polynomial used to fit the free
                energy differences.
            drop: A boolean numpy array denoting whether to drop each
                subensemble prior to fitting.

        Returns:
            A GrowthDistribution with the new estimate for the free
            energy.
        """
        size = len(self)
        diff, fit = np.zeros(size), np.zeros(size)
        dist = np.zeros(size)
        if drop is None:
            drop = np.tile(False, size)

        def smooth_component(cmp):
            """Smooth the free energy differences for a given
            component.
            """
            curr_cmp = (self.components == cmp)
            subs = np.unique(self.subensembles[curr_cmp])
            stages = np.unique(self.stages[curr_cmp])[:-1]
            states = np.arange(len(self))
            for sub in subs:
                curr_sub = curr_cmp & (self.subensembles == sub)
                not_max = self.stages < np.amax(self.stages[curr_sub])
                diff[curr_sub & not_max] = np.diff(self[curr_sub])

            for stg in stages:
                curr_stage = curr_cmp & (self.stages == stg)
                poly = np.polyfit(states[curr_stage & ~drop],
                                  diff[curr_stage & ~drop], order)
                fit[curr_stage] = np.polyval(poly, states[curr_stage])

            for sub in subs:
                curr_sub = curr_cmp & (self.subensembles == sub)
                for stg in reversed(stages):
                    curr_stage = curr_sub & (self.stages == stg)
                    next_stage = curr_sub & (self.stages == stg+1)
                    dist[curr_stage] = dist[next_stage] - fit[curr_stage]

        for cmp in np.unique(self.components):
            smooth_component(cmp)

        smoothed = copy.deepcopy(self)
        smoothed.log_probability = dist

        return smoothed

    def split(self, split=0.5):
        """Split the distribution into two parts.

        Args:
            split: The fraction of the length to use as the boundary for
                the two parts.

        Returns:
            A tuple of GrowthDistribution objects.
        """
        bound = int(split * len(self))
        ind, logp = _create_growth_index(self), self.log_probability
        fst = GrowthDistribution(index={k: v[:bound] for k, v in ind.items()},
                                 log_probability=logp[:bound])
        snd = GrowthDistribution(index={k: v[bound:] for k, v in ind.items()},
                                 log_probability=logp[bound:])

        return fst, snd

    def write(self, path):
        """Write the distribution to a file.

        Args:
            path: The name of the file to write.
        """
        np.savetxt(path, np.column_stack((np.arange(len(self))+1,
                                          self.subensembles,
                                          self.components, self.stages,
                                          self.log_probability)),
                   fmt=4*['%8d'] + ['%.11e'])


def read_distribution(path):
    """Read an lnpi_op.dat file or an lnpi_tr.dat file.

    Args:
        path: The location of the file to read.

    Returns:
        A Distribution object.
    """
    name = pathlib.Path(path).name
    if 'tr' in name:
        logp = np.loadtxt(path, usecols=(4, ))
        return GrowthDistribution(index=_read_growth_index(path),
                                  log_probability=logp)
    elif 'op' in name or 'gn' in name:
        param, logp = np.loadtxt(path, usecols=(0, 1), unpack=True)

        return OrderParameterDistribution(order_parameter=param.astype('int'),
                                          log_probability=logp)
    raise NotImplementedError
