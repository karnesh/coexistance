"""Determine the wetting properties of a direct or expanded ensemble
Monte Carlo simulation.
"""

import numpy as np


def contact_angle_cosine(spreading, drying):
    """Calculate the cosine of the contact angle.

    Args:
        spreading: A float (or numpy array): the spreading coefficient.
        drying: A float (or numpy array): the drying coefficient.

    Returns:
        The cosine of the contact angle as a float or numpy array.
    """
    return (drying-spreading) / (drying+spreading)


def drying_coefficient(distribution, plateau_fraction=0.5):
    """Calculate the drying coefficient.

    Args:
        distribution: The logarithm of the probability distribution
            with respect to the order parameter (N_i).
        plateau_fraction: The fraction of the order parameter space
            to use for the plateau region.

    Returns:
        The dimensionless drying coefficient (beta*d*A).
    """
    potential = -distribution[:]
    valley = np.amin(potential)
    split = int(plateau_fraction * len(potential))
    plateau = np.mean(potential[:split])

    return valley - plateau


def expanded_coefficients(valley, plateau, index, reference):
    """Calculate the change in spreading/drying coefficient for a pair
    of simulations.

    Args:
        valley: The log of the order parameter probability distribution
            for the valley region.
        plateau: The log of the order parameter distribution for the
            plateau.
        index: The reference subensemble index.
        reference: The reference spreading/drying coefficient.

    Returns:
        A numpy array with the spreading/drying coefficient of each
        subensemble.
    """
    return plateau - valley - plateau[index] + valley[index] + reference


def spreading_coefficient(distribution, plateau_fraction=0.5):
    """Calculate the spreading coefficient.

    Args:
        distribution: The log of the order parameter probability
            distribution.
        plateau_fraction: The fraction of the order parameter space
            to use for the plateau region.

    Returns:
        The dimensionless spreading coefficient (beta*s*A).
    """
    potential = -distribution[:]
    valley = np.amin(potential)
    split = int((1.0 - plateau_fraction) * len(potential))
    plateau = np.mean(potential[split:])

    return valley - plateau


def tension(spreading, drying):
    """Calculate the interfacial tension.

    Args:
        spreading: A float (or numpy array): the spreading coefficient.
        drying: A float (or numpy array): the drying coefficient.

    Returns:
        The interfacial tension in the appropriate units.
    """
    return -0.5 * (spreading+drying)
