// random design
// N is supplied by the R data list, so the same model works for 52subs, 64subs, or other sample sizes.
data {
  int<lower=1> N; // number of subjects
  int<lower=1> nT[N]; // number of trials per participant
  int<lower=1, upper=6> s1[sum(nT)]; // 1st stage states: 1-2 for low load condition(2 pairs of doors), 3-6 for high load condition (4 pairs of boats)
  int<lower=1, upper=6> s2[sum(nT)]; // 2nd stage states: 1-2 for low load condition(2 bears), 3-6 for high load condition(4 monkeys)
  int<lower=1, upper=2> a[sum(nT)]; // actions: defined based on choice rather than left and right!(not 1 for left and 2 for right!)
  int<lower=0, upper=9> r[sum(nT)]; // rewards
  int<lower=1, upper=5> magnitude[sum(nT)]; // magnitude condition (1 = small reward, 5 = large reward)
  int<lower=1, upper=2> load[sum(nT)]; // load condition (1 = low load, 2 = high load)
  int<lower=0, upper=2> part[sum(nT)]; // part (0 = pair2, 1 = pair4-1, 2 = pair4-2)
  matrix<lower=1, upper=2>[2,sum(nT)] display; // stimulus order
}

transformed data {
  // transform actions sequence from 1's and 2's to 0's and 1's
  // (necessary to model actions with bernoulli distribution)
  int<lower=0, upper=1> a_tr[sum(nT)];
  for (t in 1:sum(nT)) {
    a_tr[t] = a[t] -1;
  }
}

parameters {
  // group-level parameters
  real mu_alpha_hload_logit;  // reward learning rate - high load
  real<lower=0> sd_alpha_hload_logit;
  real mu_alpha_lload_logit;  // reward learning rate - low load
  real<lower=0> sd_alpha_lload_logit;
  
  vector[4] mu_beta_mf_effects; // model-free weights (b0, bmagnitude, bload, bmagnitudeload)
  vector<lower=0>[4] sd_beta_mf_effects;
  
  vector[4] mu_beta_mb_effects; // model-based weights(b0, bmagnitude, bload, bmagnitudeload)
  vector<lower=0>[4] sd_beta_mb_effects;
  
  real mu_st_hload; // choice stickiness - high load
  real<lower=0> sd_st_hload;
  real mu_st_lload; // choice stickiness - low load
  real<lower=0> sd_st_lload;
  
  real mu_respst_hload; // response stickiness - high load
  real<lower=0> sd_respst_hload;
  real mu_respst_lload; // response stickiness - low load
  real<lower=0> sd_respst_lload;
  
  
  // subject-level parameters - N subjects
  vector[N] alpha_hload_logit_z; // reward learning rate - high load
  vector[N] alpha_lload_logit_z; // reward learning rate - low load
  matrix[4, N] beta_mf_effects_raw; // model-free weights (b0, bmagnitude, bload, bmagnitudeload)
  matrix[4, N] beta_mb_effects_raw; // model-based weights (b0, bmagnitude, bload, bmagnitudeload)
  vector[N] st_hload_z; // choice stickiness - high load
  vector[N] st_lload_z; // choice stickiness - low load
  vector[N] respst_hload_z; // response stickiness - high load
  vector[N] respst_lload_z; // response stickiness - low load
  
  // correlation matrix (cholesky factorized)
  cholesky_factor_corr[4] L_mf; // model-free effects
  cholesky_factor_corr[4] L_mb; // model-based effects
}

transformed parameters {
  
  matrix<lower=0, upper=1>[2,sum(nT)] p; // probability of chosing actions - left or right? 2 (L or R) * sum(nT)(all participants' trials)
  vector[2] Q; // integrated Q values (first stage)
  vector[6] Q2; // second-stages Q values
  matrix[6,2] Qmf; // model-free Q values (first stage) (state, action)
  vector[2] Qmb; // model-based Q values (first stage)
  
  real delta1; // first-stage reward prediction error
  real delta2; // second-stage reward prediction error
  
  matrix<lower=0, upper=1>[6,2] M; // last choice structure (state1, action)
  row_vector<lower=0, upper=1>[2] K; // last response structure
  
  // define variables for model parameters on a trial level
  // values will be updated every trial depending on magnitude and load condition
  real beta_mf;
  real beta_mb;
  real alpha;
  real st;
  real respst;
  
  real lambda = 1; // eligibility trace decay (fixed parameter)
  
  matrix[2,2] Tm;
  // if (part==0||part==1){
  //   matrix<lower=0, upper=1>[2,2] Tm = [[1, 0], [0, 1]]; // transitions matrix in low load condition and high load condition - part1: Tm[action, 2nd stage state]
  //   } else if(part==2){
  //     matrix<lower=0, upper=1>[2,2] Tm = [[0, 1], [1, 0]]; // transitions matrix in high load condition - part2: Tm[action, 2nd stage state]
  //   }
  
  vector[2] Q2_subset;
  
  // subject-level parameters are estimated as standardized and uncorrelated (non-centered parameterization)
  // here, we transform them to their non-standardized values
  // also, logit-transformed parameters (alpha) are transformed back to normal space
  vector[N] alpha_hload_logit = mu_alpha_hload_logit + sd_alpha_hload_logit * alpha_hload_logit_z;
  vector[N] alpha_lload_logit = mu_alpha_lload_logit + sd_alpha_lload_logit * alpha_lload_logit_z;
  vector[N] alpha_hload = exp(alpha_hload_logit) ./ (exp(alpha_hload_logit) + 1);
  vector[N] alpha_lload = exp(alpha_lload_logit) ./ (exp(alpha_lload_logit) + 1);
  
  matrix[4,N] beta_mf_effects = mu_beta_mf_effects * rep_row_vector(1, N)+ (diag_pre_multiply(sd_beta_mf_effects, L_mf) * beta_mf_effects_raw);
  matrix[4,N] beta_mb_effects = mu_beta_mb_effects * rep_row_vector(1, N)+ (diag_pre_multiply(sd_beta_mb_effects, L_mb) * beta_mb_effects_raw);
  
  vector[N] st_hload = mu_st_hload + sd_st_hload * st_hload_z;
  vector[N] st_lload = mu_st_lload + sd_st_lload * st_lload_z;
  
  vector[N] respst_hload = mu_respst_hload + sd_respst_hload * respst_hload_z;
  vector[N] respst_lload = mu_respst_lload + sd_respst_lload * respst_lload_z;
  
  // compute model-free and model-based weights based on effects (b0[1,], bmagnitude[2,], bload[3,], bmagnitudeload[4,]) and effects coding (small reward = -0.5, large reward = 0.5, low load = 0.5, high load = -0.5)
  vector[N] beta_mf_small_lload = beta_mf_effects[1,]' - 0.5*beta_mf_effects[2,]' + 0.5*beta_mf_effects[3,]' - 0.25*beta_mf_effects[4,]';
  vector[N] beta_mf_large_lload = beta_mf_effects[1,]' + 0.5*beta_mf_effects[2,]' + 0.5*beta_mf_effects[3,]' + 0.25*beta_mf_effects[4,]';
  vector[N] beta_mf_small_hload = beta_mf_effects[1,]' - 0.5*beta_mf_effects[2,]' - 0.5*beta_mf_effects[3,]' + 0.25*beta_mf_effects[4,]';
  vector[N] beta_mf_large_hload = beta_mf_effects[1,]' + 0.5*beta_mf_effects[2,]' - 0.5*beta_mf_effects[3,]' - 0.25*beta_mf_effects[4,]';
  
  vector[N] beta_mb_small_lload = beta_mb_effects[1,]' - 0.5*beta_mb_effects[2,]' + 0.5*beta_mb_effects[3,]' - 0.25*beta_mb_effects[4,]';
  vector[N] beta_mb_large_lload = beta_mb_effects[1,]' + 0.5*beta_mb_effects[2,]' + 0.5*beta_mb_effects[3,]' + 0.25*beta_mb_effects[4,]';
  vector[N] beta_mb_small_hload = beta_mb_effects[1,]' - 0.5*beta_mb_effects[2,]' - 0.5*beta_mb_effects[3,]' + 0.25*beta_mb_effects[4,]';
  vector[N] beta_mb_large_hload = beta_mb_effects[1,]' + 0.5*beta_mb_effects[2,]' - 0.5*beta_mb_effects[3,]' - 0.25*beta_mb_effects[4,]';
  
  // iterate over participant
  for (s in 1:N) {
    
    // iterate over participant-specific trials
    for (t in (sum(nT[1:s-1])+1):sum(nT[1:s])) {
      // transition matrix of low load condition(part = 0) and high load condition - part1(part = 1)
      if (part[t] == 0 || part[t] == 1) {
        Tm = [[1, 0], [0, 1]];
      } else { // transition matrix of high load condition - part2(part = 2)
        Tm = [[0, 1], [1, 0]];
      }
      // test if trial is start of a new block (either first trial or change in load condition)
      if (t==sum(nT[1:s-1])+1 || load[t] != load[t-1]) {
        // reset values & determine parameters
        M = rep_matrix(0, 6, 2);
        K = [0, 0];
        Q2 = rep_vector(4.5, 6);
        Qmf = rep_matrix(4.5, 6, 2);
        if (load[t]==1) { // for low-load block
          alpha = alpha_lload[s];
          st = st_lload[s];
          respst = respst_lload[s];
        } else { // load[t] == 2, for high-load block
          alpha = alpha_hload[s];
          st = st_hload[s];
          respst = respst_hload[s];
        }
      }
  
      // compute model-based values
      if (s1[t] <= 2){
        Q2_subset = Q2[1:2];
      } else if (s1[t] <= 4){
        Q2_subset = Q2[3:4];
      } else {
        Q2_subset = Q2[5:6];
      }
      Qmb = Tm * Q2_subset;
      
      // get appropriate decision weights according to magnitude and load condition
      if (load[t]==1) { // low load
        if (magnitude[t]==1) {
          // small reward, low load
          beta_mf = beta_mf_small_lload[s];
          beta_mb = beta_mb_small_lload[s];
        } else {
          // large reward, low load
          beta_mf = beta_mf_large_lload[s];
          beta_mb = beta_mb_large_lload[s];
        }
      } else { // load[t] == 2, high load
        if (magnitude[t]==1) {
          // small reward, high load
          beta_mf = beta_mf_small_hload[s];
          beta_mb = beta_mb_small_hload[s];
        } else {
          // large reward, high load
          beta_mf = beta_mf_large_hload[s];
          beta_mb = beta_mb_large_hload[s];
        }
      }
      
      // reverse last response structure, if action 2 is displayed on the left side
      if (display[1,t]==2) {
        K = [K[2], K[1]];
      }
      
      // integrate values
      Q = beta_mf*Qmf[s1[t],]' + beta_mb*Qmb + st*M[s1[t],]' + respst*K';
  
      // compute action probabilities
      p[1,t] = exp(Q[1])/sum(exp(Q));
      p[2,t] = exp(Q[2])/sum(exp(Q));
      
      // update last choice structure
      M[s1[t], 1] = 0; 
      M[s1[t], 2] = 0;
      M[s1[t],a[t]] = 1;
      
      // update last response structure
      K = [0, 0];
      if (a[t]==display[1,t]) {
        K[1] = 1;
      } else {
        K[2] = 1;
      }
      
      // compute reward prediction errors
      delta1 = Q2[s2[t]] - Qmf[s1[t],a[t]];
      delta2 = r[t] - Q2[s2[t]];
  
      // update model-free values
      Qmf[s1[t],a[t]] = Qmf[s1[t],a[t]] + alpha*delta1 + alpha*lambda*delta2;
      Q2[s2[t]] = Q2[s2[t]] + alpha*delta2;
    }
  }
}

model {
  // parameter priors
  // group-level parameters
	mu_alpha_hload_logit ~ normal(0,1);
  sd_alpha_hload_logit ~ normal(0,2);
  mu_alpha_lload_logit ~ normal(0,1);
  sd_alpha_lload_logit ~ normal(0,2);
  
  mu_beta_mf_effects ~ normal(0,1);
  sd_beta_mf_effects ~ normal(0,2);
  
  mu_beta_mb_effects ~ normal(0,1);
  sd_beta_mb_effects ~ normal(0,2);

  mu_st_hload ~ normal(0,1);
  sd_st_hload ~ normal(0,2);
  mu_st_lload ~ normal(0,1);
  sd_st_lload ~ normal(0,2);
  
  mu_respst_hload ~ normal(0,1);
  sd_respst_hload ~ normal(0,2);
  mu_respst_lload ~ normal(0,1);
  sd_respst_lload ~ normal(0,2);
  
  // participant-level parameters
  alpha_hload_logit_z ~ normal(0, 1);
  alpha_lload_logit_z ~ normal(0, 1);
  to_vector(beta_mf_effects_raw) ~ normal(0, 1);
  to_vector(beta_mb_effects_raw) ~ normal(0, 1);
  st_hload_z ~ normal(0, 1);
  st_lload_z ~ normal(0, 1);
  respst_hload_z ~ normal(0, 1);
  respst_lload_z ~ normal(0, 1);
  
  // correlation matrix
  L_mf ~ lkj_corr_cholesky(2);
  L_mb ~ lkj_corr_cholesky(2);
  
  
  // model observed actions with action probabilities
  a_tr ~ bernoulli(p[2,]);
}

generated quantities {
  // transform correlation matrix (Cholesky --> normal)
  matrix[4,4] Omega_mf; // correlation matrix model-free effects
  matrix[4,4] Omega_mb; // correlation matrix model-based effects
  Omega_mf = multiply_lower_tri_self_transpose(L_mf);
  Omega_mb = multiply_lower_tri_self_transpose(L_mb);
}
