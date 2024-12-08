import numpy as np
import pyccl as ccl
from scipy.linalg import block_diag
import emcee
from scipy.optimize import minimize

cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.04, h=0.67, sigma8=0.82, n_s=0.96)
k_arr = np.geomspace(1e-4, 1e1, 64)
a_arr = np.linspace(0.1, 1, 32)
hmd_200m = ccl.halos.MassDef200m
cM = ccl.halos.ConcentrationDuffy08(mass_def=hmd_200m)
nM = ccl.halos.MassFuncTinker08(mass_def=hmd_200m)
bM = ccl.halos.HaloBiasTinker10(mass_def=hmd_200m)
pM = ccl.halos.HaloProfileNFW(
    mass_def=hmd_200m, concentration=cM, fourier_analytic=True
)
hmc = ccl.halos.HMCalculator(mass_function=nM, halo_bias=bM, mass_def=hmd_200m)
HOD2pt = ccl.halos.Profile2ptHOD()
l_arr = np.unique(np.geomspace(1, 20000, 256).astype(np.int32))


class Bin:
    """Generic class for storing datavectors, N(z), covariances, etc. inside of a single bin"""

    def __init__(self, bin_number):
        self.bin_number = bin_number
        self.theta = None  # Theta should be in units of arcmin.
        self.w = None
        self.w_unc = None
        self.cov = None
        self.z = None
        self.nz = None

    def _compute_clustering_model(self, pars):
        log10Mmin, log10M1, alpha, sigma = pars
        pg = ccl.halos.HaloProfileHOD(
            mass_def=hmd_200m,
            concentration=cM,
            log10Mmin_0=log10Mmin,
            log10M1_0=log10M1,
            alpha_0=alpha,
            siglnM_0=sigma,
        )
        # Bias??
        t_g = ccl.NumberCountsTracer(
            cosmo,
            dndz=(self.z, self.nz),
            bias=(self.z, np.ones_like(self.z)),
            has_rsd=False,
        )
        pk_ggf = ccl.halos.halomod_Pk2D(
            cosmo, hmc, pg, prof_2pt=HOD2pt, lk_arr=np.log(k_arr), a_arr=a_arr
        )
        cl_gg = ccl.angular_cl(cosmo, t_g, t_g, l_arr, p_of_k_a=pk_ggf)
        return ccl.correlation(
            cosmo,
            ell=l_arr,
            C_ell=cl_gg,
            theta=self.theta / 60,
            type="NN",
            method="FFTLog",
        )

    def _compute_chi2(self, pars):
        # Very inefficient implementation, this should only be run for testing.
        modelvector = self._compute_clustering_model(pars)
        return np.einsum(
            "i, ij, j",
            self.w - modelvector,
            np.linalg.inv(self.cov),
            self.w - modelvector,
        )


class DataVector:
    """Generic class for storing datavectors, N(z), covariances, etc."""

    def __init__(self, name):
        self.name = name
        self.bins = []

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self.bins):
            self.n += 1
            return self.bins[self.n - 1]
        else:
            raise StopIteration

    def import_datavector(self, thetas, ws, covs):
        # thetas, ws, covs = arrays of shape [bin_0, bin_1, ...]
        assert len(thetas) == len(ws) == len(covs)
        self.nbins = len(thetas)

        for bin in range(self.nbins):
            bin_obj = Bin(bin_number=bin)
            bin_obj.theta = thetas[bin]
            bin_obj.w = ws[bin]
            bin_obj.w_unc = np.sqrt(np.diag(covs[bin]))
            bin_obj.cov = covs[bin]
            self.bins.append(bin_obj)

        self.block_theta = np.array(thetas).flatten()
        self.block_w = np.array(ws).flatten()
        self.block_w_unc = np.sqrt(np.diag(block_diag(*covs)))
        self.block_invcov = np.linalg.inv(block_diag(*covs))

    def import_nz(self, zs, nzs):
        # zs, nzs = arrays of shape [bin_0, bin_1, ...]
        assert self.nbins == len(zs) == len(nzs)

        for bin in range(self.nbins):
            self.bins[bin].z = zs[bin]
            self.bins[bin].nz = nzs[bin]

    def compute_clustering_model(self, pars):
        return np.array(
            [self.bins[bin]._compute_clustering_model(pars) for bin in range(5)]
        ).flatten()

    def compute_chi2(self, pars):
        modelvector = self.compute_clustering_model(pars)
        return np.einsum(
            "i, ij, j",
            self.block_w - modelvector,
            self.block_invcov,
            self.block_w - modelvector,
        )


def log_like(pars, datavector, prior):
    pri = prior(pars)
    if np.isfinite(pri):
        try:
            resid = -0.5 * datavector.compute_chi2(pars)
            return resid + pri
        except Exception as e:
            print(e)
            return -np.inf
    else:
        return pri


def run_emcee(
    datavector,
    filename,
    initial,
    prior,
    samples_dir,
    nsteps=700,
    scatter=0.05,
    posteriors=False,
):
    # SET UP VARIABLES AND DATA
    pos = initial + scatter * np.random.randn(32, len(initial))
    nwalkers, ndim = pos.shape
    backend_name = f"{samples_dir}/clustering_chain.h5"
    backend = emcee.backends.HDFBackend(backend_name)
    backend.reset(nwalkers, ndim)

    # RUN EMCEE model(pars, data, prior):
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_like,
        args=(datavector, prior),
        backend=backend,
    )
    sampler.run_mcmc(pos, nsteps, progress=True)
    flat_samples = sampler.get_chain(flat=True)
    if posteriors:
        np.save(filename, [flat_samples, sampler.get_log_prob(flat=True)])
    else:
        np.save(filename, flat_samples)


def min_func(*args):
    return -1 * log_like(*args)


def gradient_minimize(pars, datavector, prior):
    return minimize(
        min_func,
        pars,
        args=(datavector, prior),
        method="Nelder-Mead",
    )
