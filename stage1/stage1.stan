/*
  Command+ V6 — Stage 1: Population-level aim model
  Estimates structural aim given pitch type, batter side, and count.
  No pitcher random effects — those are isolated in Stage 3.

  Formula (equivalent brms notation):
    y ~ 1 + batter_left + (1|cluster_x_batter) + (1|count) + (1|cluster_x_count)
    sigma ~ [design matrix: intercept + cluster dummies + count dummies]
    family = student_t

  Non-centered parameterization for all random effects.
*/
data {
  int<lower=0> N;              // observations
  int<lower=1> J_cxb;         // cluster x batter levels
  int<lower=1> J_count;       // count levels
  int<lower=1> J_cxcount;     // cluster x count levels
  int<lower=1> P_sigma;       // columns in sigma design matrix (incl intercept)

  vector[N] y;                // PlateLocHeight_z OR PlateLocSide_z
  array[N] int<lower=1, upper=J_cxb>     cxb_id;
  array[N] int<lower=1, upper=J_count>   count_id;
  array[N] int<lower=1, upper=J_cxcount> cxcount_id;
  vector[N] batter_left;      // 1 = Left, 0 = Right
  matrix[N, P_sigma] X_sigma; // design matrix for log(sigma) submodel

  // Empirically derived prior rates (exponential distribution)
  real<lower=0> tau_cxb_rate;
  real<lower=0> tau_count_rate;
  real<lower=0> tau_cxcount_rate;
}

parameters {
  real alpha;
  real beta_left;

  vector[J_cxb]     u_cxb_raw;
  vector[J_count]   u_count_raw;
  vector[J_cxcount] u_cxcount_raw;

  real<lower=0> tau_cxb;
  real<lower=0> tau_count;
  real<lower=0> tau_cxcount;

  vector[P_sigma] beta_sigma;  // [intercept, cluster effects, count effects]

  real<lower=1> nu;
}

transformed parameters {
  vector[J_cxb]     u_cxb    = u_cxb_raw    * tau_cxb;
  vector[J_count]   u_count  = u_count_raw  * tau_count;
  vector[J_cxcount] u_cxcount = u_cxcount_raw * tau_cxcount;
}

model {
  // Hyperpriors
  tau_cxb    ~ exponential(tau_cxb_rate);
  tau_count  ~ exponential(tau_count_rate);
  tau_cxcount ~ exponential(tau_cxcount_rate);

  // Non-centered random effects
  u_cxb_raw    ~ std_normal();
  u_count_raw  ~ std_normal();
  u_cxcount_raw ~ std_normal();

  // Fixed effects
  alpha      ~ normal(0, 0.2);
  beta_left  ~ normal(0, 0.5);

  // Sigma submodel — same prior on intercept and effects
  beta_sigma ~ normal(0, 0.3);

  // Nu prior (calibrated to allow heavy tails; gamma(5,0.9) mean ~ 5.56)
  nu ~ gamma(5, 0.9);

  // Likelihood
  {
    vector[N] mu = alpha + beta_left * batter_left
                   + u_cxb[cxb_id]
                   + u_count[count_id]
                   + u_cxcount[cxcount_id];
    vector[N] log_sigma = X_sigma * beta_sigma;
    y ~ student_t(nu, mu, exp(log_sigma));
  }
}
