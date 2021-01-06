import numpy as np
from scipy.optimize import fmin


def coexistence(lnpi, N):
        """Locate the coexistence acticity near the critical point by
        maximizing compressibility.

        Args:
            lnpi: The original log of probability distribution.
            N: particle number distribution.

        Returns:
            A newlog of probability distribution at the coexistence
            point.
        """
        def compress(ratio):
            """Return the compressibility kai"""

            lnpi_new = lnpi + ratio*N
            prob = np.exp(lnpi_new)/np.sum(np.exp(lnpi_new))
            kai = np.dot(N*N,prob)-np.dot(N,prob)**2
            return kai

        solution = fmin(lambda x: -compress(x), x0=0)
        lnpi += solution[0] * N

        return lnpi


def finite_scaling(frac,path,T,T0,H,d):
    """Calculate the first order cumulant M = <m^2>/<|m|>^2, m = eta_c -
    <eta_c>
    Args:
        frac : The activity at simulation condition.
        path: The path where simulation data is stored.
        T: The list of temperatures to reweight to.
        T0: The simulation temperature.
        H: Cubic Box length(3D).
        d : Dimension

    Retruns:
        M: The second order cumulant ratio, <m^2>/<|m|>^2
        U: The fourth order cumulant ratio, <m^4>/<m^2>^2
    """
    N,nlnpi = np.loadtxt(path  + '/lnpi_op.dat', usecols=(0,1), unpack=True)
    nlnpi = np.log(np.exp(nlnpi)/np.sum(np.exp(nlnpi)))
    elim = np.loadtxt(path  + '/elim.dat')[:,1:4]
    ehist = np.loadtxt(path  + '/ehist.dat')[:,1:]

    """Histogram Reweighting and M calculation"""
    # Set constants and parameters
    sigma = 4.0
    kb = 1.38e-23
    m = np.zeros(len(T))
    m_2=np.zeros(len(T))
    m_4=np.zeros(len(T))
    if d == 3:
        rho = np.pi/6*sigma**3*N/H**3
    elif d == 2:
        rho = np.pi*sigma**2*N/H**2

    for i in range(len(T)):
        nlnpi_new = np.zeros(len(N))
        #Reweight and calculate the new pi(N)[j] at each N[j]
        for j in range(len(N)):
            num,e_st,e_en = elim[j,:]
            emicro = np.linspace(e_st,e_en,num)
            eh = ehist[:num,j]
            elnpi = np.log(eh/np.sum(eh))
            elnpi_new = elnpi + emicro*(1.0/kb/T0-1.0/kb/T[i])
            eprob_new = np.exp(elnpi_new)/np.sum(np.exp(elnpi_new))
            lnpi_new = (elnpi_new + nlnpi[j]
                        + (1.0/kb/T[i]-1.0/kb/T0)*frac[0]/(1.0/kb/T0)*N[j])
            nlnpi_new[j] = np.log(np.sum(np.exp(lnpi_new)))

        #Reweight new lnpi(N) to saturated acticity
        nlnpi_new = coexistence(nlnpi_new, N)
        prob = np.exp(nlnpi_new)/np.sum(np.exp(nlnpi_new))
        rho_av = np.dot(rho,prob)
        m[i] = np.dot(np.abs(rho-rho_av),prob)
        m_2[i] = np.dot((rho-rho_av)**2,prob)
        m_4[i] = np.dot((rho-rho_av)**4,prob)
    M = m_2/m**2
    U = m_4/m_2**2
    return M, U
