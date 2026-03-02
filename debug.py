import muon as mu

mdata = mu.read_h5mu("testing.h5mu")

mu.tl.mofa(
    data=mdata,
    use_obs="intersection",
    convergence_mode="medium",
    n_factors=12,
    seed=1114,
    outfile=f"models/test_model.hdf5",
    use_var="highly_variable",
    scale_views=True,
    scale_groups=True,
    center_groups=True,
    ard_weights=True,
    ard_factors=True,
    spikeslab_weights=True,
    spikeslab_factors=True,
    verbose=False,
    gpu_mode=True,
    gpu_device=0
)