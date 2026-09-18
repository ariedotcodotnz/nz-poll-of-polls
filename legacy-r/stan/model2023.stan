// model2023.stan
// Statespace model to predict voting intention using poll data.
// Based on Peter Ellis' model https://github.com/ellisp/nz-election-forecast.
// Removed 'political science prior' from Ellis' model as it wasn't clear how to model
// the impact of Hipkins becoming Prime Minister.
//
// varies from model2020.stan only in the number of pollsters

data {
  array[3] int n_weeks; // number of weeks between each election
  int n_parties; // number of parties
  array[n_parties] real mu_elect1; // value at first election
  array[n_parties] real mu_elect2; // value at second election
  array[n_parties] real mu_elect3; // value at third election
  real inflator; // amount by which to multiply the standard error of polls

  // note - pollsters are individually hard coded in to avoid having to use some kind of ragged array:
  int n_pollsters;

  int y1_n; // number of polls conducted by pollster 1
  array[y1_n, n_parties] real y1_values; // actual values in polls for pollster 1
  array[y1_n] int y1_weeks; // the number of weeks since first election each poll was taken
  array[n_parties] real y1_se; // the standard error for each party from pollster 1 (note this stays the same throughout history)

  int y2_n;
  array[y2_n, n_parties] real y2_values;
  array[y2_n] int y2_weeks;
  array[n_parties] real y2_se;

  int y3_n;
  array[y3_n, n_parties] real y3_values;
  array[y3_n] int y3_weeks;
  array[n_parties] real y3_se;

  int y4_n;
  array[y4_n, n_parties] real y4_values;
  array[y4_n] int y4_weeks;
  array[n_parties] real y4_se;
  vector[y4_n] reid_method;
}
parameters {
  array[sum(n_weeks)] vector[n_parties] epsilon; // innovations in underlying state of vote intention
  corr_matrix[n_parties] omega;
  array[n_parties] real<lower=0> sigma; // standard deviations for daily innovations for each party
  array[n_pollsters, n_parties] real d; // house effects for n_pollsters and n_parties combinations
  array[n_parties] real reid_impact; // impact on party 5 (Reid Research) of the change in Reid's method in 2017
}
transformed parameters {
  array[sum(n_weeks), n_parties] real mu; // underlying state of vote intention, as a proportion (not percentage)

  mu[1,  : ] = mu_elect1;
  for (i in 2 : sum(n_weeks)) {
    for (j in 1 : n_parties) {
      mu[i, j] = mu[i - 1, j] + epsilon[i][j] * sigma[j];
    }
  }
}
model {
  vector[n_parties] zeroes;
  zeroes = rep_vector(0, n_parties);

  // prior for scaling of innovations
  sigma ~ normal(0.002, 0.001);

  // prior for the effect of Reid Research's changed method in 2017
  reid_impact ~ normal(zeroes, 0.02); // fairly tight prior because it's not plausibly  more than 10% change for a party

  // prior for correlation matrix of innovations, on standardised scale (so SD = 1)
  omega ~ lkj_corr(1); // LKJ prior on the correlation matrix

  // innovations in the state space, on standardised scale
  // Note - when this is done as iid normal rather than multi_normal it makes things
  // dramatically faster (20 minutes versus 80 minutes), but at cost to modelling coherence.
  // We need this multivariate approach to adequately capture the correlation
  epsilon ~ multi_normal(zeroes, omega);

  // measurement model
  // 1. Election result for second and third elections - treat as observations with a very big sample size
  mu_elect2 ~ normal(mu[n_weeks[1],  : ], sqrt(.3 * .7 / 10 ^ 5));
  mu_elect3 ~ normal(mu[n_weeks[1] + n_weeks[2],  : ], sqrt(.3 * .7 / 10 ^ 5));

  // 2. Polls

  // For each pollster, and each party, we model the observations of the vector of values each day
  for (p in 1 : n_pollsters)
    d[p,  : ] ~ normal(0.0, 0.03); // ie a fairly loose prior for the house effects for each pollster.
  // Brought down from 0.075 to 0.025 on 18 August 2017 because it was converging to non-sensible results.

  for (j in 1 : n_parties) {
    y1_values[ : , j] ~ normal(to_vector(mu[y1_weeks, j]) + d[1, j],
                               y1_se[j] * inflator);

    y2_values[ : , j] ~ normal(to_vector(mu[y2_weeks, j]) + d[2, j],
                               y2_se[j] * inflator);

    y3_values[ : , j] ~ normal(to_vector(mu[y3_weeks, j]) + d[3, j],
                               y3_se[j] * inflator);

    y4_values[ : , j] ~ normal(to_vector(mu[y4_weeks, j]) + d[4, j]
                              + reid_impact[j] * reid_method,
                               y4_se[j] * inflator);

  }
}
