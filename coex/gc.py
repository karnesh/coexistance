"""Find the coexistence properties of grand canonical (direct and
expanded ensemble) simulations.
"""

import copy
import pathlib

import numpy as np
from scipy.optimize import fsolve

from coex.dist import read_distribution
from coex.hist import read_all_n_histograms, read_histogram


def activities_to_fractions(activities, one_subensemble=False):
    """Convert a list of activities to activity fractions.

    Args:
        activities: A numpy array with the activities of the system.
        one_subensemble: A bool that describes the shape of the
            input/output.

    Returns:
        A numpy array with the logarithm of the sum of the activities
        and the activity fractions of each species after the first.
        If the array is multidimensional, each column corresponds to
        a subensemble from an expanded ensemble simulation.
    """
    activities = np.array(activities)
    if ((not one_subensemble and len(activities.shape) == 1)
            or (one_subensemble and len(activities) == 1)):
        return np.log(activities)

    fractions = np.copy(activities)
    fractions[0] = np.log(sum(activities))
    fractions[1:] /= np.exp(fractions[0])

    return fractions


def fractions_to_activities(fractions, one_subensemble=False):
    """Convert a list of activity fractions to activities.

    Args:
        fractions: A numpy array with the activity fractions.
        one_subensemble: A bool that describes the shape of the
            input/output.

    Returns:
        A numpy array with the activities. If the array is
        multidimensional, each column corresponds to a subensemble
        from an expanded ensemble simulation.
    """
    fractions = np.array(fractions)
    if ((not one_subensemble and len(fractions.shape) == 1)
            or (one_subensemble and len(fractions) == 1)):
        return np.exp(fractions)

    activities = np.copy(fractions)
    activity_sum = np.exp(fractions[0])
    activities[1:] *= activity_sum
    activities[0] = activity_sum - sum(activities[1:])

    return activities


def read_fractions(path):
    """Read the activity fractions of an expanded ensemble simulation.

    Args:
        path: The location of the 'bz.dat' or 'zz.dat' file to read.

    Returns:
        A numpy array or dict with keys 'beta' and 'fractions'. The
        first row of the fractions contains the logarithm of
        the sum of the activities for each subensemble, and each
        subsequent row contains the activity fractions of each species
        after the first.
    """
    if 'bz' in pathlib.Path(path).name:
        beta = np.loadtxt(path, usecols=(1, ))
        fractions = np.transpose(np.loadtxt(path))[2:]

        return {'beta': beta, 'fractions': fractions}

    return np.transpose(np.loadtxt(path))[1:]


def write_fractions(path, fractions, beta=None):
    """Write the activity fractions to file.

    Args:
        path: The 'bz.dat' or 'zz.dat' file to write.
        fractions: A numpy array with activity fractions.
        beta: An optional array with the inverse temperature (1/kT)
            values.
    """
    if beta is not None:
        arr = np.column_stack((range(len(beta)), beta, *fractions))
    else:
        cols = fractions.shape[1]
        arr = np.column_stack((range(cols), *fractions))

    np.savetxt(path, arr, fmt='%.15g')


class DirectGCSimulation:
    """Calculate the coexistence properties of the output from a direct
    grand canonical simulation.

    Attributes:
        dist: An OrderParameterDistribution object.
        nhists: A list of molecule number VisitedStatesHistogram
            objects.
        species_activities: A numpy array with the activities of each
            species, adjusted by the number of ions.
        ions_per_species: The number of ions each species has.
        weights: The logarithm of the initial activities minus the
            logarithm of the coexistence activities, used to calculate
            the average number of molecules at the coexistence point
            via histogram reweighting.
    """

    def __init__(self, dist, nhists, species_activities, ions_per_species=1):
        self.dist = dist
        self.nhists = nhists
        self.species_activities = np.array(species_activities)
        self.ions_per_species = ions_per_species
        self.weights = np.tile(None, len(nhists)-1)

    @property
    def activities(self):
        """Calculate the species-based activities.

        Returns:
            An array.
        """
        return np.power(self.species_activities, 1.0/self.ions_per_species)

    @activities.setter
    def activities(self, acts):
        self.species_activities = np.power(acts, self.ions_per_species)

    def average_n(self, boundary=0.5):
        """Find the average number of molecules in each phase.

        Args:
            boundary: A float: where (as a fraction of the order
                parameter range) the liquid/vapor phase boundary lies.

        Returns:
            A dict with the keys 'liquid' and 'vapor', each referencing
            a numpy array with the average number of molecules of each
            species.
        """
        bound = int(boundary * len(self.dist))

        def average_phase(hist, weight, is_vapor=True):
            """Calculate the average number of molecules for a given
            phase.
            """
            split_hist = hist[:bound] if is_vapor else hist[bound:]
            split_dist = self.dist[:bound] if is_vapor else self.dist[bound:]
            split_probs = np.exp(split_dist) / sum(np.exp(split_dist))

            return sum([split_probs[i] * d.average(weight)
                        for i, d in enumerate(split_hist)])

        hists = self.nhists[1:]
        if len(hists) > 1:
            vapor = np.array([average_phase(nh, self.weights[i])
                              for i, nh in enumerate(hists)])
            liquid = np.array(
                [average_phase(nh, self.weights[i], is_vapor=False)
                 for i, nh in enumerate(hists)])
        else:
            vapor = average_phase(hists[0], self.weights[0])
            liquid = average_phase(hists[0], self.weights[0], is_vapor=False)

        return {'vapor': vapor, 'liquid': liquid}

    def coexistence(self, species=1, boundary=0.5, initial=-0.001):
        """Find the coexistence point of the simulation.

        Args:
            species: The simulation's order parmeter species.
            boundary: A float: where (as a fraction of the order
                parameter range) the liquid/vapor phase boundary lies.
            initial: The initial guess for the optimization solver.

        Returns:
            A new DirectGCSimulation object at the coexistence point.
        """
        def objective(ratio):
            """Find the absolute value of the difference between the
            area under the vapor and liquid peaks of the probability
            distribution.
            """
            vapor, liquid = self.dist.split(boundary)
            vapor_prob = np.exp(vapor + ratio*vapor.order_parameter)
            liquid_prob = np.exp(liquid + ratio*liquid.order_parameter)

            return np.abs(sum(vapor_prob) - sum(liquid_prob))

        solution = fsolve(objective, x0=initial, maxfev=10000)
        coex = copy.deepcopy(self)
        coex.dist += solution * coex.dist.order_parameter
        if species == 0:
            frac = activities_to_fractions(self.species_activities,
                                           one_subensemble=True)
            frac[0] += solution
            coex.species_activities = fractions_to_activities(
                frac, one_subensemble=True)
        else:
            coex.species_activities[species-1] *= np.exp(solution)

        nonzero = np.nonzero(coex.species_activities)
        log_old = np.log(self.species_activities[nonzero])
        log_new = np.log(coex.species_activities[nonzero])
        coex.weights = np.zeros(self.species_activities.shape)
        coex.weights[nonzero] = log_old - log_new

        return coex

    def composition(self, boundary=0.5):
        """Calculate the average composition of each phase in the
        simulation.

        Args:
            boundary: A float: where (as a fraction of the order
                parameter range) the liquid/vapor phase boundary lies.

        Returns:
            A (vapor, liquid) tuple of numpy arrays, each containing the
            mole fraction of each species.
        """
        size = len(self.dist)
        if len(self.nhists) < 3:
            return {'vapor': np.tile(1.0, size), 'liquid': np.tile(1.0, size)}

        avg_n = self.average_n(boundary)

        return {'vapor': avg_n['vapor'] / sum(avg_n['vapor']),
                'liquid': avg_n['liquid'] / sum(avg_n['liquid'])}

    @property
    def fractions(self):
        """Calculate the activity fractions.

        Returns:
            An array with the log of the sum of the activities and
            the activity fractions of each species beyond the first.
        """
        return np.reshape(activities_to_fractions(self.activities,
                                                  one_subensemble=True),
                          len(self.activities))

    @fractions.setter
    def fractions(self, frac):
        self.activities = fractions_to_activities(frac)

    def grand_potential(self):
        """Calculate the grand potential.

        If the order parameter is the total molecule number N, then
        this is the absolute grand potential of the system.
        Otherwise, it is a relative value for the analyzed species.

        Returns:
            A float.
        """
        prob = np.exp(self.dist)
        prob /= sum(prob)

        return np.log(prob[0] * 2.0)


def direct_coexistence(sim, species, boundary=0.5, initial=-0.001):
    """Find the coexistence point of a direct simulation.

    Args:
        species: The simulation's order parmeter species.
        boundary: A float: where (as a fraction of the order
            parameter range) the liquid/vapor phase boundary lies.
        initial: The initial guess for the optimization solver.

    Returns:
        A new DirectGCSimulation object at the coexistence point.
    """
    return sim.coexistence(species, boundary, initial)


def read_direct_simulation(path, fractions, ions_per_species=1):
    """Read the relevant data from a simulation directory.

    Args:
        path: The directory containing the data.
        fractions: The activity fractions (chi, eta_j) of the
            simulation.
        ions_per_species: The number of ions each species has.

    Returns:
        A dict with the order parameter, logarithm of the
        probability distribution, and molecule number visited
        states histograms.
    """
    dist = read_distribution(pathlib.Path(path) / 'lnpi_op.dat')
    nhists = read_all_n_histograms(path)
    frac = np.transpose(np.array(fractions))
    act = fractions_to_activities(frac)
    act = np.power(act, ions_per_species)

    return DirectGCSimulation(dist=dist, nhists=nhists,
                              species_activities=act,
                              ions_per_species=ions_per_species)


class SamplingError(Exception):
    """An Exception for cases in which a histogram is not adequately
    sampled to calculate a property, such as the grand potential.
    """
    pass


class ExpandedGCSimulation:
    """Calculate the coexistence properties of the output of a grand
    canonical expanded ensemble simulation.

    Attributes:
        dist: A Distribution object.
        index: The reference subensemble index.
        nhists: A list of molecule number VisitedStatesHistogram
            objects.
        species_activities: A numpy array of the activities of each
            species (adjusted by the number of ions) for each
            subensemble in the simulation.
        beta: An optional list of thermodynamic beta (1/kT) values,
            for temperature expanded ensemble simulations.
        ions_per_species: A scalar or list with the number of ions each
            species has.
        weights: The logarithm of the initial activities minus the
            logarithm of the coexistence activities, used to
            calculate the average number of molecules at the
            coexistence point via histogram reweighting.
    """

    def __init__(self, dist, index, nhists=None, species_activities=None,
                 beta=None, ions_per_species=1):
        self.dist = dist
        self.index = index
        self.nhists = nhists
        self.species_activities = np.array(species_activities)
        self.beta = np.array(beta)
        self.ions_per_species = ions_per_species
        self.weights = np.tile(None, species_activities.shape)

    @property
    def activities(self):
        """Calculate the species-based activities.

        Returns:
            An array.
        """
        return np.power(self.species_activities, 1.0/self.ions_per_species)

    @activities.setter
    def activities(self, acts):
        self.species_activities = np.power(acts, self.ions_per_species)

    @property
    def fractions(self):
        """Calculate the activity fractions.

        Returns:
            An array with the log of the sum of the activities and
            the activity fractions of each species beyond the first.
        """
        return activities_to_fractions(self.activities)

    @fractions.setter
    def fractions(self, fracs):
        self.activities = fractions_to_activities(fracs)

    def shift_to_coexistence(self, solutions, species):
        """Shift the activities and order parameter probability
        distribution to the coexistence point.

        Args:
            solutions: A list of log(activitiy) differences.
            species: The species used in histogram reweighting.
        """
        nonzero = np.nonzero(self.species_activities)
        log_old_act = np.zeros(self.species_activities.shape)
        log_old_act[nonzero] = np.log(self.species_activities[nonzero])
        if species == 0:
            frac = activities_to_fractions(self.species_activities)
            frac[0] -= solutions
            self.species_activities = fractions_to_activities(frac)

        for i, sol in enumerate(solutions):
            if i == self.index:
                continue

            self.dist.log_probability[i] += (
                self.nhists[species][i].reweight(sol))
            if species != 0:
                new_act = np.exp(log_old_act[species-1, i] - sol)
                self.species_activities[species-1, i] = new_act

        new_nonzero = np.nonzero(self.species_activities)
        log_new_act = np.log(self.species_activities[new_nonzero])
        self.weights = np.zeros(log_old_act.shape)
        self.weights[new_nonzero] = log_old_act[new_nonzero] - log_new_act

    def composition(self):
        """Calculate the weighted average composition of the phase.

        Returns:
            A numpy array with the mole fraction of each species in each
            subensemble.
        """
        if len(self.nhists) < 3:
            return np.tile(1.0, len(self.dist))

        avg_n = self.average_n()

        return avg_n / sum(avg_n)

    def grand_potential(self):
        """Calculate the grand potential of each subensemble.

        This function walks the length of the expanded ensemble path
        (forwards or backwards) and uses the N=0 visited state
        distribution to calculate the grand potential of each
        subensemble if applicable.  If the N=0 state is not sampled
        sufficiently, the free energy difference between subensembles
        is used.

        Note that this function will not work for liquid phases, which
        do not usually have the N=0 state sampled.

        Returns:
            A numpy array with the grand potential of each subensemble.
        """
        result = np.zeros(len(self.dist))
        nhist = self.nhists[0]

        def sampled_vapor_subensembles():
            """Find the subensembles of the simulation in which the
            total molecule number histogram has the N=0 state sampled
            adequately.
            """
            return np.array([True if (d.bins[0] < 1.0e-8
                                      and d.counts[0] > 1000) else False
                             for d in nhist])

        def calculate_range(iter_range, is_reversed=False):
            """Calculate the grand potential for a given range of
            subensembles.
            """
            in_sampled_block = True
            for i in iter_range:
                if sampled[i] and in_sampled_block:
                    result[i] = np.log(nhist[i].counts[0]
                                       / sum(nhist[i].counts))
                else:
                    in_sampled_block = False
                    if is_reversed:
                        result[i] = (result[i+1] + self.dist[i+1]
                                     - self.dist[i])
                    else:
                        result[i] = (result[i-1] + self.dist[i-1]
                                     - self.dist[i])

        sampled = sampled_vapor_subensembles()
        length = len(result)
        if sampled[0]:
            calculate_range(range(length))
        elif sampled[-1]:
            calculate_range(reversed(range(length)), is_reversed=True)
        else:
            if np.count_nonzero(sampled) == 0:
                raise SamplingError("{}\n{}".format(
                    "Can't find a sampled subensemble for the grand",
                    'potential calculation. Is this phase a liquid?'))

            first_sampled = np.nonzero(sampled)[0][0]
            calculate_range(range(first_sampled, 0), is_reversed=True)
            calculate_range(range(first_sampled, length))

        return result

    def average_n(self):
        """Calculate the weighted average number of molecules.

        Returns:
            A numpy array with the number of molecules of each species
            in each subensemble.
        """
        hists = self.nhists[1:]
        if len(hists) > 1:
            return np.array(
                [h.average(w) for h, w in zip(hists, self.weights)])

        return hists[0].average(self.weights[0])


def read_expanded_simulation(path, index, fractions=None, beta=None,
                             ions_per_species=1):
    """Read the relevant data from an exapnded ensemble simulation
    directory.

    Args:
        path: The directory containing the data.
        index: The reference subensemble index.
        fractions: The reference activity fractions.
        beta: The reference inverse temperature (1/kT).
        ions_per_species: The number of ions each species has.

    Returns:
        An ExpandedGCSimulation with the data contained in the given
        directory.
    """
    true_path = pathlib.Path(path)
    dist = read_distribution(true_path / 'lnpi_op.dat')
    nhists = read_all_n_histograms(path)
    tee_beta = None
    try:
        tee_frac = read_fractions(true_path / 'bz.dat')
        tee_beta = tee_frac['beta']
        act = fractions_to_activities(tee_frac['fractions'])
    except FileNotFoundError:
        try:
            act = fractions_to_activities(read_fractions(true_path / 'zz.dat'))
        except FileNotFoundError:
            act = None

    logp_shift = -dist[index]
    if beta is not None:
        energy = read_histogram(true_path / 'ehist.dat')[index]
        logp_shift += energy.reweight(beta - tee_beta[index])

    if fractions is not None:
        act = np.power(act, ions_per_species)
        ref_act = np.power(fractions_to_activities(fractions,
                                                   one_subensemble=True),
                           ions_per_species)
        nonzero = np.nonzero(ref_act)[0]
        ratios = np.zeros(ref_act.shape)
        ratios[nonzero] = (np.log(act[nonzero, index])
                           - np.log(ref_act[nonzero]))
        for hist, ratio in zip(nhists[1:], ratios):
            logp_shift += hist[index].reweight(ratio)

    dist.log_probability += logp_shift

    return ExpandedGCSimulation(dist=dist, index=index, nhists=nhists,
                                species_activities=act, beta=tee_beta,
                                ions_per_species=ions_per_species)


def read_expanded_simulations(paths, index, fractions=None, beta=None,
                              ions_per_species=1):
    """Read the relevant data from a list of exapnded ensemble
    simulations with the same reference point.

    Args:
        paths: A list of directories containing the data to read.
        index: The reference subensemble index.
        fractions: The reference activity fractions.
        beta: The reference inverse temperature (1/kT).
        ions_per_species: The number of ions each species has.

    Returns:
        A generator expression for reading the data in each directory.
    """
    return (read_expanded_simulation(path=path, index=index,
                                     fractions=fractions, beta=beta,
                                     ions_per_species=ions_per_species)
            for path in paths)


def liquid_liquid_coexistence(first, second, species, grand_potential,
                              initial=0.01):
    """Find the coexistence point of two liquid phases.

    Note that the two phases must already be shifted to their
    appropriate reference points.

    Args:
        first: An ExpandedGCSimulation with data for the first phase.
        second: An ExpandedGCSimulation with data for the second phase.
        species: The species to use for histogram reweighting.
        grand_potential: The reference grand potential.
        initial: The initial guess to use for the solver.

    Returns:
        A tuple with the two ExpandedGCSimulation objects at
        coexistence.
    """
    fst = copy.deepcopy(first)
    snd = copy.deepcopy(second)
    for phase in fst, snd:
        phase.dist.log_probability -= phase.dist[phase.index]+grand_potential

    return _two_phase_coexistence(fst, snd, species, initial)


def liquid_vapor_coexistence(liquid, vapor, species, initial=0.01):
    """Find the coexistence point of a liquid phase and a vapor phase.

    Args:
        liquid: An ExpandedGCSimulation with the liquid data.
        vapor: An ExpandedGCSimulation with the vapor data.
        species: The species to use for histogram reweighting.
        initial: The initial guess to use for the solver.

    Returns:
        A tuple with the two ExpandedGCSimulation objects at
        coexistence.

    Notes:
        The liquid and vapor phases must already be shifted to their
        appropriate reference points.
    """
    liq = copy.deepcopy(liquid)
    vap = copy.deepcopy(vapor)
    try:
        potential = vap.grand_potential()
    except SamplingError as error:
        raise RuntimeError('{}\n{}'.format(
            'Consider using liquid_liquid_coexistence() with a ',
            'reference grand potential.')) from error

    vap.dist.log_probability = -potential
    liq.dist.log_probability += vap.dist[vap.index] - liq.dist[liq.index]

    return _two_phase_coexistence(liq, vap, species, initial)


def _two_phase_coexistence(first, second, species, initial):
    """Find the coexistence point of two grand canonical expanded
    ensemble simulations.

    Args:
        first: An ExpandedGCSimulation with data for the first phase.
        second: An ExpandedGCSimulation with data for the second phase.
        species: The integer representing which species to use for the
            reweighting.
        initial: The initial guess to use for the solver.

    Returns:
        A tuple with the two ExpandedGCSimulation objects at
        coexistence.

    Notes:
        The first and second phases must already be shifted to their
        appropriate reference points.
    """
    def solve(i):
        """Find the coexistence point for a given subensemble."""
        def objective(ratio):
            """Calculate the absolute value of the difference between
            the reweighted probability distribution of each phase.
            """
            fst = first.dist[i] + first.nhists[species][i].reweight(ratio)
            snd = second.dist[i] + second.nhists[species][i].reweight(ratio)
            return np.abs(fst - snd)

        if i == first.index or i == second.index:
            return 0.0

        return fsolve(objective, x0=initial)[0]

    solutions = [solve(i) for i in range(len(first.dist))]
    for phase in first, second:
        phase.shift_to_coexistence(solutions, species)

    return first, second


def liquid_liquid_liquid_coexistence(first, second, third, species,
                                     grand_potential, initial=0.01):
    """Find the coexistence point of two liquid phases.

    Note that the two phases must already be shifted to their
    appropriate reference points.

    Args:
        first: An ExpandedGCSimulation with data for the first phase.
        second: An ExpandedGCSimulation with data for the second phase.
        third: An ExpandedGCSimulation with data for the third phase.
        species: A tuple of two species to use for histogram
            reweighting.
        grand_potential: The reference grand potential.
        initial: The initial guess to use for the solver.

    Returns:
        A tuple with the three ExpandedGCSimulation objects at
        coexistence.
    """
    fst = copy.deepcopy(first)
    snd = copy.deepcopy(second)
    thd = copy.deepcopy(third)
    for phase in fst, snd, thd:
        phase.dist -= phase.dist[phase.index] + grand_potential

    return _three_phase_coexistence(fst, snd, thd, species, initial)


def liquid_liquid_vapor_coexistence(liquid1, liquid2, vapor, species,
                                    initial=0.01):
    """Find the coexistence point of a liquid phase and a vapor phase.

    Args:
        liquid1: An ExpandedGCSimulation with data for the first
            liquid.
        liquid2: An ExpandedGCSimulation with data for the second
            liquid.
        vapor: An ExpandedGCSimulation with the vapor data.
        species: A tuple of two species to use for histogram
            reweighting.
        initial: The initial guess to use for the solver.

    Returns:
        A tuple with the three ExpandedGCSimulation objects at
        coexistence.

    Notes:
        The liquid and vapor phases must already be shifted to their
        appropriate reference points.
    """
    liq1 = copy.deepcopy(liquid1)
    liq2 = copy.deepcopy(liquid2)
    vap = copy.deepcopy(vapor)
    try:
        potential = vap.grand_potential()
    except SamplingError as error:
        raise RuntimeError('{}\n{}'.format(
            'Consider using liquid_liquid_liquid_coexistence() with a ',
            'reference grand potential.')) from error

    vap.dist.log_probability = -potential
    liq1.dist.log_probability += vap.dist[vap.index] - liq1.dist[liq1.index]
    liq2.dist.log_probability += vap.dist[vap.index] - liq2.dist[liq2.index]

    return _three_phase_coexistence(liq1, liq2, vap, species, initial)


def _three_phase_coexistence(first, second, third, species, initial):
    """Find the coexistence point of three grand canonical expanded
    ensemble simulations.

    Args:
        first: An ExpandedGCSimulation with data for the first phase.
        second: An ExpandedGCSimulation with data for the second phase.
        third: An ExpandedGCSimulation with data for the third phase.
        species: A tuple of two species to use for reweighting.
        initial: The initial guess to use for the solver.

    Returns:
        A tuple with the two ExpandedGCSimulation objects at
        coexistence.

    Notes:
        The phases must already be shifted to their appropriate
        reference points.
    """
    phases = (first, second, third)
    assert len(species) == 2, 'Need two species to find triple point.'

    def solve(i):
        """Find the coexistence point for a given subensemble."""
        def objective(ratios):
            """Calculate the absolute value of the difference between
            the reweighted probability distribution of each phase.
            """
            ans = np.zeros(3)
            for j, phase in enumerate(phases):
                ans[j] = phase.dist[i]

            for k, ratio in enumerate(ratios):
                for j, phase in enumerate(phases):
                    if np.amax(phase.nhists[species[k]][i].bins) > 0:
                        ans[j] += phase.nhists[species[k]][i].reweight(ratio)

            return [np.abs(ans[1] - ans[0]), np.abs(ans[2] - ans[0])]

        if i == first.index or i == second.index or i == third.index:
            return np.tile(0.0, 2)

        return fsolve(objective, x0=np.tile(initial, 2))

    solutions = np.reshape([solve(i) for i in range(len(first.dist))],
                           [len(first.dist), 2])
    for phase in first, second, third:
        for j, spec in enumerate(species):
            phase.shift_to_coexistence(solutions[:, j], spec)

    return first, second, third
