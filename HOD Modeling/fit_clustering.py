import pickle
import pyarrow.dataset as ds
from emcee_helpers import *
from query_parquet_helpers import check

# CONFIG
NSTEPS = 1000
CLUSTERING_DATAVECTOR = (
    "/gpfs/home/bclevine/redmagic/data/clustering_results_Nov16_weighted.pkl"
)
LENS_CATALOG = "/gpfs/projects/VonDerLindenGroup/bclevine/redmagic_data/redmagic_parquet/redmagic_full_processed.parquet"
SAMPLES_DIR = "/gpfs/scratch/bclevine/samples"

check(files=[CLUSTERING_DATAVECTOR, LENS_CATALOG], dirs=[SAMPLES_DIR])

with open(CLUSTERING_DATAVECTOR, "rb") as f:
    clustering_results = pickle.load(f)
lens_dataset = (
    ds.dataset(
        LENS_CATALOG,
        format="parquet",
    )
    .to_table()
    .combine_chunks()
)

datavector_list = []

# Let's come up with a quick way to extract the relevant data for a given subsample
print("Loading in data.", flush=True)
labels = ["round", "ellip", "all"]
lens_bin_edges = [0.15, 0.35, 0.50, 0.65, 0.80, 0.90]
for i in range(3):
    # Data vectors:
    datavector_list.append(DataVector(name=labels[i]))
    theta_list = [clustering_results[f"bin_{bin}"][i][0] for bin in range(5)]
    w_list = [clustering_results[f"bin_{bin}"][i][1] for bin in range(5)]
    cov_list = [clustering_results[f"bin_{bin}"][i][2] for bin in range(5)]
    datavector_list[i].import_datavector(theta_list, w_list, cov_list)

    # N(z)s:
    z_temp = []
    nz_temp = []
    for j in range(5):
        if labels[i] == "round":
            zmask_lens = (
                (np.array(lens_dataset["zredmagic"]) > lens_bin_edges[j])
                & (np.array(lens_dataset["zredmagic"]) < lens_bin_edges[j + 1])
                & (np.array(lens_dataset["weight"]) > 0)
                & (np.array(lens_dataset["ellip_bin"]) == -1)
            )
        elif labels[i] == "ellip":
            zmask_lens = (
                (np.array(lens_dataset["zredmagic"]) > lens_bin_edges[j])
                & (np.array(lens_dataset["zredmagic"]) < lens_bin_edges[j + 1])
                & (np.array(lens_dataset["weight"]) > 0)
                & (np.array(lens_dataset["ellip_bin"]) == 1)
            )
        elif labels[i] == "all":
            zmask_lens = (
                (np.array(lens_dataset["zredmagic"]) > lens_bin_edges[j])
                & (np.array(lens_dataset["zredmagic"]) < lens_bin_edges[j + 1])
                & (np.array(lens_dataset["weight"]) > 0)
            )

        nz_arr_lens, z_edge_arr_lens = np.histogram(
            np.array(lens_dataset["zredmagic"])[zmask_lens], bins=100, range=[0, 1]
        )
        z_arr_lens = np.array(
            [
                (z_edge_arr_lens[j] + z_edge_arr_lens[j + 1]) / 2
                for j in range(len(z_edge_arr_lens) - 1)
            ]
        )
        z_temp.append(z_arr_lens)
        nz_temp.append(nz_arr_lens)
    datavector_list[i].import_nz(z_temp, nz_temp)


def prior(pars):
    log10Mmin, log10M1, alpha, sigma = pars
    if (9 < log10Mmin < 17) & (9 < log10M1 < 17) & (-1 < alpha < 3) & (-1 < sigma < 3):
        return 1
    else:
        return -np.inf


init_pars = [12.5, 13.5, 1.3, 0.554]
for subsample_idx in range(3):
    print(f"Minimizing for {labels[subsample_idx]}.", flush=True)
    init = gradient_minimize(init_pars, datavector_list[subsample_idx], prior)
    print(f"Running emcee for {labels[subsample_idx]}.", flush=True)
    run_emcee(
        datavector_list[subsample_idx],
        f"{SAMPLES_DIR}/chain_clustering_{labels[subsample_idx]}",
        init.x,
        prior,
        samples_dir=SAMPLES_DIR,
        nsteps=NSTEPS,
    )
