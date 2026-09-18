list(
  tar_stan_mcmc(
    model2023,
    "stan/model2023.stan",
    dir = ".stan",
    data = d1,
    chains = 4,
    parallel_chains = 4,
    iter_sampling = 2000,
    max_treedepth = 20,
    deployment = "worker"
  ),
    tar_stan_mcmc(
    model2020,
    "stan/model2020.stan",
    dir = ".stan",
    data = d12020,
    chains = 4,
    parallel_chains = 4,
    iter_sampling = 2000,
    max_treedepth = 20,
    deployment = "worker"
  )

)
