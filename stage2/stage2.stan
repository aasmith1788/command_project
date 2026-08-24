/*
  Command+ V6 — Stage 2: Batter x cluster population aim adjustment.

  Models the population-level aim shift induced by each specific batter
  for each pitch cluster x batter side combination, across ALL pitchers.

  Input residuals: r1 = y - mu_pop  (Stage 1 residuals, in feet)

  Model:
    r1_i ~ student_t(nu, omega[b_id[i]], sigma[i])
    omega[b] ~ normal(0, tau_batter)
    tau_batter ~ exponential(tau_rate)

  Non-centered parameterization for omega.
  nu and sigma are fixed from Stage 1 posterior means.
*/
data {
  int<lower=0> N;                            // total pitches
  int<lower=1> B;                            // unique batter x cluster groups
  vector[N]    r1;                           // Stage 1 residuals (feet)
  array[N] int<lower=1, upper=B> b_id;       // batter x cluster index
  vector<lower=0>[N] sigma;                  // per-pitch sigma from Stage 1 (feet)
  real<lower=1>      nu;                     // fixed from Stage 1 posterior mean
  real<lower=0>      tau_rate;              // prior rate for tau_batter
}

parameters {
  vector[B]       omega_raw;
  real<lower=0>   tau_batter;
}

transformed parameters {
  vector[B] omega = omega_raw * tau_batter;
}

model {
  tau_batter ~ exponential(tau_rate);
  omega_raw  ~ std_normal();
  r1 ~ student_t(nu, omega[b_id], sigma);
}
