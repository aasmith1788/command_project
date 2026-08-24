/*
  Command+ V6 — Stage 3: Per-pitcher aim estimation.

  Residuals after removing Stage 1 population aim and Stage 2 batter adjustment:
    r2_i = y_i - mu_pop_i - omega_i

  Model:
    mu_i = delta[k_id[i]] + u_game[g_id[i]]
    r2_i ~ student_t(nu[c_id[i]], mu_i + AR1_correction, sigma_i * sqrt(1 - rho^2))

  AR(1) structure within each game:
    - First pitch in game: no correction (prev_id == 0)
    - Subsequent pitches:  mu_i += rho * (r2[prev_id] - mu_prev)

  Parameters per pitcher:
    delta[K]    — aim deviation per cluster x batter side combo
    u_game[G]   — game-level random intercept
    tau_delta   — SD of aim deviations
    tau_game    — SD of game intercepts
    rho         — AR(1) coefficient
    nu[C]       — per pitch-cluster degrees of freedom (Student-t)

  Non-centered parameterization for delta and u_game.
*/
data {
  int<lower=0> N;                              // pitches for this pitcher
  int<lower=1> K;                              // cluster x batter combos
  int<lower=1> G;                              // games
  int<lower=1> C;                              // distinct pitch clusters (for nu)

  vector[N]    r2;                             // Stage 2 residuals (feet)
  array[N] int<lower=1, upper=K> k_id;        // cluster x batter index
  array[N] int<lower=1, upper=G> g_id;        // game index
  array[N] int<lower=1, upper=C> c_id;        // pitch cluster index (for nu)
  vector<lower=0>[N] sigma;                   // per-pitch sigma from Stage 1 (feet)

  array[N] int<lower=0> prev_id;             // 1-indexed previous pitch in same game; 0 = none

  real<lower=0> tau_delta_rate;
  real<lower=0> tau_game_rate;
}

parameters {
  vector[K]       delta_raw;
  vector[G]       u_game_raw;
  real<lower=0>   tau_delta;
  real<lower=0>   tau_game;
  real<lower=-1, upper=1> rho;
  array[C] real<lower=1>  nu;
}

transformed parameters {
  vector[K] delta  = delta_raw  * tau_delta;
  vector[G] u_game = u_game_raw * tau_game;
}

model {
  tau_delta ~ exponential(tau_delta_rate);
  tau_game  ~ exponential(tau_game_rate);
  delta_raw  ~ std_normal();
  u_game_raw ~ std_normal();
  rho ~ uniform(-1, 1);
  nu  ~ gamma(5, 0.9);

  for (n in 1:N) {
    real mu_n = delta[k_id[n]] + u_game[g_id[n]];

    if (prev_id[n] == 0) {
      r2[n] ~ student_t(nu[c_id[n]], mu_n, sigma[n]);
    } else {
      int   prev    = prev_id[n];
      real  mu_prev = delta[k_id[prev]] + u_game[g_id[prev]];
      real  ar_mu   = mu_n + rho * (r2[prev] - mu_prev);
      real  ar_sig  = sigma[n] * sqrt(1.0 - square(rho));
      r2[n] ~ student_t(nu[c_id[n]], ar_mu, ar_sig);
    }
  }
}
