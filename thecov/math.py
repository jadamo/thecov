"""This module contains math functions for the covariance calculation
"""
import numpy as np

def r2c_to_c2c_3d(fourier):
    """Completes a 3D Fourier array generated using PFFT's r2c method with the elements
    that are omitted due to the Hermitian symmetry of the Fourier transform.

    Parameters
    ----------
    fourier : array_like
        3D Fourier array generated using PFFT's r2c method.

    Returns
    -------
    array_like
        Completed 3D Fourier array.
    """

    fourier_c = np.conj(fourier[:, :, (-2 if fourier.shape[0]%2 == 0 else -1):0:-1])

    fourier_c[1:, :, :] = fourier_c[:0:-1, :, :]
    fourier_c[:, 1:, :] = fourier_c[:, :0:-1, :]

    return np.concatenate((fourier, fourier_c), axis=2)

def sample_from_shell(rmin, rmax, discrete=True):
    """Sample a point uniformly from a spherical shell.

    Parameters
    ----------
    rmin : float
        Minimum radius of the shell.
    rmax : float
        Maximum radius of the shell.
    discrete : bool, optional
        If True, the sampled point will be rounded to the nearest integer.
        Default is True.

    Returns
    -------
    x,y,z,r : float
        Coordinates of the sampled point.
    """

    r = rmin + (rmax - rmin) * np.random.rand()
    theta = 2 * np.pi * np.random.rand()
    phi = np.arccos(1 - 2 * np.random.rand())

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    if(discrete):
        x,y,z = int(np.round(x)), int(np.round(y)), int(np.round(z))
        r = np.sqrt(x**2 + y**2 + z**2)
        if(r < rmin or r > rmax):
            return sample_from_shell(rmin, rmax, discrete)

    return x,y,z,r

def sample_from_cube(rmin, rmax, dr, max_modes=np.inf):

    iL = int(np.ceil(rmax))

    ix, iy, iz = np.mgrid[-iL:iL+1,-iL:iL+1,-iL:iL+1]
    ir = np.sqrt(ix**2 + iy**2 + iz**2)

    sort = ((ir - rmin)/dr).astype(int)

    modes = []
    Nmodes = []

    imax = (rmax - rmin)/dr
    imax = int(np.round(imax)) if np.allclose(np.round(imax), imax) else int(imax)
    
    for i in range(imax):
        mask = sort == i
        N = np.sum(mask)
        if N > max_modes:
            rand_mask = np.random.randint(N, size=max_modes)
            modes.append(np.array([ix[mask][rand_mask],
                                   iy[mask][rand_mask],
                                   iz[mask][rand_mask],
                                   ir[mask][rand_mask]]).T)
        else:
            modes.append(np.array([ix[mask],
                                   iy[mask],
                                   iz[mask],
                                   ir[mask]]).T)
        Nmodes.append(N)

    return modes, Nmodes

def sample_kmodes(kmin, kmax, dk=None, boxsize=None, max_modes=1000,
                  k_shell_approx=0.05, binning='linear', nbins=None):
    """
    Sample discrete Fourier-space modes per k-bin.

    Parameters
    ----------
    kmin, kmax : float
        Min/max k (same units).
    dk : float, optional
        Linear bin width (required if binning == 'linear').
    boxsize : float
        Box size for fundamental mode (required).
    max_modes : int
        Maximum number of discrete modes to return per bin.
    k_shell_approx : float
        Threshold k below which we use exact discrete-cube sampling.
    binning : {'linear', 'log'}
        Binning mode.
    nbins : int, optional
        Number of bins (required if binning == 'log').

    Returns
    -------
    modes : list of ndarray
        List of length `nbins` (or deduced linear bins) where each element is an
        (N, 4) array of sampled modes [ix, iy, iz, r].
    nmodes : ndarray
        Array with number of discrete modes per bin (integer counts or shell approximations).
    """    
    import logging
    logger = logging.getLogger('SampleModes')

    # Build edges according to binning
    if binning == 'linear':
        if dk is None:
            raise ValueError("dk must be provided for linear binning")
        kedges = np.arange(kmin, kmax + dk/2, dk)
        kmid = np.arange(kmin + dk/2, kmax + dk/2, dk)
        k_shell = max((k_shell_approx - kmin)//dk * dk + kmin, kmin)
    elif binning == 'log':
        if nbins is None:
            raise ValueError("nbins must be provided for log binning")
        kedges = np.logspace(np.log10(kmin), np.log10(kmax), nbins + 1)
        kmid = np.sqrt(kedges[:-1] * kedges[1:])  # geometric mean as kmid
        k_shell = max(k_shell_approx, kmin)
    else:
        raise ValueError(f"Unknown binning mode: {binning}")

    kbins = len(kmid)
    kfun = 2 * np.pi / boxsize
    r_edges = kedges / kfun

    cube_bin_indices = np.where(kmid < k_shell)[0]
    shell_bin_indices = np.where(kmid >= k_shell)[0]

    kfun = 2 * np.pi / boxsize

    # Uses full cube from k = 0 to k_shell
    if binning == 'linear':
        cube_modes, cube_nmodes = sample_from_cube(kmin / kfun, k_shell / kfun, dk / kfun, max_modes=max_modes)

        # Uses spherical shell approximation from k = k_shell to kmax
        kedges_shell = np.arange(k_shell, kmax + dk/2, dk)
        shell_modes = [np.array([sample_from_shell(kmin / kfun, kmax / kfun) for _ in range(
                        max_modes)]) for kmin, kmax in zip(kedges_shell[:-1], kedges_shell[1:])]
        shell_nmodes = nmodes(boxsize**3, kedges_shell[:-1], kedges_shell[1:])
        logger.info(f'Sampled {len(cube_modes)} bins from cube and {len(shell_modes)} bins from shell approximation.')
        return cube_modes + shell_modes, np.array(cube_nmodes + list(shell_nmodes))
    else:
        # Prepare integer grid once up to required radius (in units of r = k/kfun)
        iL = int(np.ceil(r_edges.max()))
        ix, iy, iz = np.mgrid[-iL:iL+1, -iL:iL+1, -iL:iL+1]
        ir = np.sqrt(ix**2 + iy**2 + iz**2)

        modes = []
        Nmodes = []
        # Loop over ALL bins in order; for cube bins use exact discrete selection,
        # for shell bins use the spherical-shell stochastic sampling (as before).
        for i in range(kbins):
            rmin = r_edges[i]
            rmax = r_edges[i+1]

            if i in cube_bin_indices:
                mask = (ir >= rmin) & (ir < rmax)
                N = int(np.sum(mask))
                if N == 0:
                    modes.append(np.empty((0, 4)))
                else:
                    coords = np.vstack((ix[mask], iy[mask], iz[mask], ir[mask])).T
                    if N > max_modes:
                        # sample without replacement
                        idx = np.random.choice(N, size=max_modes, replace=False)
                        coords = coords[idx]
                    modes.append(coords)
                Nmodes.append(N)
            else:
                # spherical-shell approximation sampling
                # make samples in r-units (r = k/kfun) and round to integers inside sample_from_shell
                samples = np.array([sample_from_shell(rmin, rmax) for _ in range(max_modes)])
                # sample_from_shell returns (x,y,z,r) already
                modes.append(samples)
                # approximate number of modes in shell from continuous formula
                Nmodes.append(None)  # fill later after computing shell_nmodes

        # For shell bins compute continuous nmodes using existing nmodes()
        if len(shell_bin_indices) > 0:
            shell_kmin = kedges[shell_bin_indices]
            shell_kmax = kedges[shell_bin_indices + 1]
            shell_nmodes = nmodes(boxsize**3, shell_kmin, shell_kmax)
            # put shell counts into Nmodes array
            shell_iter = iter(shell_nmodes)
            for idx in range(len(Nmodes)):
                if Nmodes[idx] is None:
                    Nmodes[idx] = int(next(shell_iter))
                    
        logger.info(f'Sampled {len(cube_bin_indices)} bins from cube and {len(shell_bin_indices)} bins from shell approximation.')
        return modes, np.array(Nmodes)

def nmodes(volume, kmin, kmax):
    '''Compute the number of modes in a given shell.

    Parameters
    ----------
    volume : float
        Volume of the survey.
    kmin : float
        Minimum k of the shell.
    kmax : float
        Maximum k of the shell.

    Returns
    -------
    float
        Number of modes.
    '''
    return volume / 3. / (2*np.pi**2) * (kmax**3 - kmin**3)

def cov2cor(covariance):
    '''Compute the correlation matrix from the covariance matrix.

    Parameters
    ----------
    covariance : array_like
        Covariance matrix.

    Returns
    -------
    array_like
        Correlation matrix.'''
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

def fgrowth(Omega_m, z):
    '''Estimates the growth rate at redshift z.

    Parameters
    ----------
    Omega_m : float
        Matter density parameter.
    z : float
        Redshift.

    Returns
    -------
    float
        Growth rate.
    '''
    from scipy.special import hyp2f1

    return (1. + 6*(Omega_m-1)*hyp2f1(4/3., 2, 17/6., (1-1/Omega_m)/(1+z)**3) / \
          (11*Omega_m*(1+z)**3*hyp2f1(1/3., 1, 11/6., (1-1/Omega_m)/(1+z)**3) ))

def wedges_to_multipoles(corr, muedges, ells=(0,2,4)):
    from scipy import special
    corr_ell = []
    for ell in ells:
        poly = special.legendre(ell).integ()(muedges)
        legendre = (2 * ell + 1) * (poly[1:] - poly[:-1])
        corr_ell.append(np.sum(corr * legendre, axis=-1) / (muedges[-1] - muedges[0]))
    return np.array(corr_ell)

def legendre(ell):
    if ell == 0:
        legendre = lambda mu: 1
    elif ell == 2:
        legendre = lambda mu: (3*mu**2 - 1)/2
    elif ell == 4:
        legendre = lambda mu: (35*mu**4 - 30*mu**2 + 3)/8
        
    return np.vectorize(legendre)

def get_real_Ylm(ell, m, modules=None):
    """
    Return a function that computes the real spherical harmonic of order (ell, m).
    Copied from pypower.
    Adapted from https://github.com/bccp/nbodykit/blob/master/nbodykit/algorithms/convpower/fkp.py.

    Note
    ----
    Faster evaluation will be achieved if sympy and numexpr are available.
    Else, fallback to numpy and scipy's functions.

    Parameters
    ----------
    ell : int
        The degree of the harmonic.

    m : int
        The order of the harmonic; abs(m) <= ell.

    modules : str, default=None
        If 'sympy', use sympy + numexpr to speed up calculation.
        If 'scipy', use scipy.
        If ``None``, defaults to sympy if installed, else scipy.

    Returns
    -------
    Ylm : callable
        A function that takes 3 arguments: (xhat, yhat, zhat)
        unit-normalized Cartesian coordinates and returns the
        specified Ylm.

    References
    ----------
    https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
    """
    # Make sure ell, m are integers
    ell = int(ell)
    m = int(m)

    # Normalization of Ylms
    amp = np.sqrt((2 * ell + 1) / (4 * np.pi))
    if m != 0:
        fac = 1
        for n in range(ell - abs(m) + 1, ell + abs(m) + 1): fac *= n  # (ell + |m|)!/(ell - |m|)!
        amp *= np.sqrt(2. / fac)

    sp = None

    if modules is None:
        try: import sympy as sp
        except ImportError: pass

    elif 'sympy' in modules:
        import sympy as sp

    elif 'scipy' not in modules:
        raise ValueError('modules must be either ["sympy", "scipy", None]')

    # sympy is not installed, fallback to scipy
    if sp is None:
        import scipy.special

        def Ylm(xhat, yhat, zhat):
            # The cos(theta) dependence encoded by the associated Legendre polynomial
            toret = amp * (-1)**m * scipy.special.lpmv(abs(m), ell, zhat)
            # The phi dependence
            phi = np.arctan2(yhat, xhat)
            if m < 0:
                toret *= np.sin(abs(m) * phi)
            else:
                toret *= np.cos(abs(m) * phi)
            return toret

        # Attach some meta-data
        Ylm.l = ell
        Ylm.m = m
        return Ylm

    # The relevant cartesian and spherical symbols
    # Using intermediate variable r helps sympy simplify expressions
    x, y, z, r = sp.symbols('x y z r', real=True, positive=True)
    xhat, yhat, zhat = sp.symbols('xhat yhat zhat', real=True, positive=True)
    phi, theta = sp.symbols('phi theta')
    defs = [(sp.sin(phi), y / sp.sqrt(x**2 + y**2)),
            (sp.cos(phi), x / sp.sqrt(x**2 + y**2)),
            (sp.cos(theta), z / sp.sqrt(x**2 + y**2 + z**2))]

    # The cos(theta) dependence encoded by the associated Legendre polynomial
    expr = (-1)**m * sp.assoc_legendre(ell, abs(m), sp.cos(theta))

    # The phi dependence
    if m < 0:
        expr *= sp.expand_trig(sp.sin(abs(m) * phi))
    elif m > 0:
        expr *= sp.expand_trig(sp.cos(m * phi))

    # Simplify
    expr = sp.together(expr.subs(defs)).subs(x**2 + y**2 + z**2, r**2)
    expr = amp * expr.expand().subs([(x / r, xhat), (y / r, yhat), (z / r, zhat)])

    try: import numexpr
    except ImportError: numexpr = None
    Ylm = sp.lambdify((xhat, yhat, zhat), expr, modules='numexpr' if numexpr is not None else ['scipy', 'numpy'])

    # Attach some meta-data
    Ylm.expr = expr
    Ylm.l = ell
    Ylm.m = m
    return Ylm
