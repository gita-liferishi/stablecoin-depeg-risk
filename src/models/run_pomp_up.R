# ============================================================================
# Stablecoin Depeg Panel POMP — Enhanced Bretó, joint fit across 4 assets
# ----------------------------------------------------------------------------
# Training units: USDT, USDC, DAI, FRAX  (all pre-collapse history)
# Held-out test:  UST                    (calibration 2020-11-25 to 2022-01-31)
#
# Shared params:   phi, sigma_eta, sigma_nu, gamma, delta, nu
# Specific params: mu_h, G_0, H_0   (per asset)
# ETH mask:        USDT=0, USDC=0, DAI=1, FRAX=1, UST=0
# ============================================================================

library(pomp)
library(panelPomp)
library(doParallel)
library(doRNG)
library(ggplot2)
library(tidyr)
library(dplyr)
library(patchwork)

set.seed(2050320976)

PANEL_CSV <- "pomp_panel.csv"
run_level <- 3                    # 1=debug, 2=medium, 3=production
OUT_DIR   <- "output/"
dir.create(OUT_DIR, showWarnings = FALSE)

TRAIN_ASSETS   <- c("DAI", "LUSD", "sUSD", "FRAX", "USDD")
ETH_MASK       <- c(DAI = 1, LUSD = 1, sUSD = 0, FRAX = 1,
                    MIM = 1, USDD = 0, UST = 0)

ASSET_END_DATE <- list(
  sUSD = as.Date("2025-03-31")
)

UST_CALIB_END  <- as.Date("2022-01-31")   # UST specific-parameter estimation ends here
UST_EVAL_END   <- as.Date("2022-05-10")   # UST signal evaluation window ends here

MIM_CALIB_END <- as.Date("2022-10-31")
MIM_TEST_END  <- as.Date("2023-05-31")

panel_all <- read.csv(PANEL_CSV, stringsAsFactors = FALSE)
panel_all$timestamp <- as.Date(panel_all$timestamp)

cat("Panel loaded. Counts per asset:\n")
print(table(panel_all$symbol))

build_unit_pomp <- function(asset, panel_all, date_max = NULL) {
  d <- subset(panel_all, symbol == asset)
  if (nrow(d) == 0) {
    stop(sprintf("build_unit_pomp('%s'): zero rows in panel_all. Available: %s",
                 asset, paste(sort(unique(panel_all$symbol)), collapse = ", ")))
  }
  d <- d[order(d$timestamp), ]
  
  asset_end <- ASSET_END_DATE[[asset]]
  if (!is.null(asset_end)) {
    n_before <- nrow(d)
    d <- d[d$timestamp <= asset_end, ]
    cat(sprintf("  %s truncated at <= %s  (%d -> %d rows)\n",
                asset, format(asset_end), n_before, nrow(d)))
  }
  
  if (!is.null(date_max)) d <- subset(d, timestamp <= date_max)
  
  keep <- complete.cases(d$y_bp, d$delta_fgi_scaled, d$eth_drawdown, d$eth_drawdown_3d)
  d <- d[keep, ]
  
  y_obs     <- d$y_bp
  delta_fgi <- d$delta_fgi_scaled
  eth_dd    <- d$eth_drawdown_3d
  N <- length(y_obs)
  
  if (N == 0) {
    stop(sprintf("build_unit_pomp('%s'): zero rows after filtering. ",
                 "Check date_max, ASSET_END_DATE, and NA patterns.", asset))
  }
  
  if (!(asset %in% names(ETH_MASK))) {
    stop(sprintf("build_unit_pomp('%s'): no entry in ETH_MASK. ",
                 "Add '%s' to the ETH_MASK vector.", asset, asset))
  }
  
  m_eth <- unname(ETH_MASK[asset])
  
  cat(sprintf("  %-5s  N=%4d  sd(y_bp)=%6.2f  m_eth=%d  (%s to %s)\n",
              asset, N, sd(y_obs), m_eth, min(d$timestamp), max(d$timestamp)))
  
  sc_statenames <- c("H", "G", "Y_state")
  sc_paramnames <- c("sigma_nu", "mu_h", "phi", "sigma_eta",
                     "gamma", "delta", "nu", "G_0", "H_0")
  
  rproc1 <- "
    double beta, omega, nu_shock;
    omega    = rnorm(0, sigma_eta * sqrt(1 - phi*phi) * sqrt(1 - tanh(G)*tanh(G)));
    nu_shock = rnorm(0, sigma_nu);
    G       += nu_shock;
    beta     = Y_state * sigma_eta * sqrt(1 - phi*phi);
    H        = mu_h * (1 - phi) + phi * H
               + beta * tanh(G) * exp(-H/2)
               + gamma * delta_fgi_cov
               + m_eth * delta * eth_dd_cov
               + omega;
    if (H > 30.0)  H = 30.0;
    if (H < -30.0) H = -30.0;
  "
  rproc2_filt <- "Y_state = covaryt;"
  rproc2_sim  <- "Y_state = exp(H/2) * rt(nu);"
  
  sc_rinit <- "
    G = G_0;
    H = H_0;
    Y_state = exp(H/2) * rt(nu);
  "
  sc_rmeasure <- "y = exp(H/2) * rt(nu);"
  sc_dmeasure <- "
    double s = exp(H/2);
    if (!R_FINITE(s) || s <= 0.0) {
      lik = -1000.0;
    } else {
      lik = dt(y / s, nu, 1) - log(s);
    }
    if (!give_log) lik = exp(lik);
  "
  
  covar_tbl <- covariate_table(
    time          = 0:N,
    covaryt       = c(0, y_obs),
    delta_fgi_cov = c(0, delta_fgi),
    eth_dd_cov    = c(0, eth_dd),
    m_eth         = rep(m_eth, N + 1),
    times         = "time"
  )
  
  po <- pomp(
    data = data.frame(y = y_obs, time = 1:N),
    statenames = sc_statenames,
    paramnames = sc_paramnames,
    times = "time",
    t0 = 0,
    covar = covar_tbl,
    rmeasure = Csnippet(sc_rmeasure),
    dmeasure = Csnippet(sc_dmeasure),
    rprocess = discrete_time(step.fun = Csnippet(paste(rproc1, rproc2_filt)),
                             delta.t = 1),
    rinit = Csnippet(sc_rinit),
    partrans = parameter_trans(
      log   = c("sigma_eta", "sigma_nu"),
      logit = "phi"
    )
  )
  
  # Attach dates for downstream extraction
  attr(po, "dates") <- d$timestamp
  attr(po, "depeg_severity") <- d$depeg_severity
  attr(po, "in_depeg") <- d$in_depeg
  po
}



cat("\nBuilding unit pomps for training assets:\n")
train_units <- setNames(
  lapply(TRAIN_ASSETS, function(a) build_unit_pomp(a, panel_all)),
  TRAIN_ASSETS
)


# ============================================================================
# 4. Assemble panelPomp object
# ============================================================================

# Starting values: mu_h scaled to each asset's y_bp; shared params start neutral.
mu_h_inits <- sapply(train_units, function(po) 2 * log(sd(obs(po)[1, ])))
cat("\nAsset-specific mu_h starting values (2*log(sd(y_bp))):\n")
print(round(mu_h_inits, 3))

shared_start <- c(
  sigma_nu  = exp(-4.5),
  phi       = plogis(3.0),
  sigma_eta = exp(-0.5),
  gamma     = 0.0,
  delta     = 0.5,
  nu        = 8.0
)

# Specific parameters: matrix with rows = param, cols = unit
specific_start <- rbind(
  mu_h = mu_h_inits,
  G_0  = rep(0.0, length(TRAIN_ASSETS)),
  H_0  = mu_h_inits
)
colnames(specific_start) <- TRAIN_ASSETS

pp <- panelPomp(
  train_units,
  shared   = shared_start,
  specific = specific_start
)

cat("\nPanelPomp assembled.\n")
print(pp)


# ============================================================================
# 5. Parallel setup and run-level controls
# ============================================================================

sc_Np            <- switch(run_level,   50, 1000, 2000)
sc_Nmif          <- switch(run_level,    5,  100,  200)
sc_Nreps_eval    <- switch(run_level,    4,   10,   10)
sc_Nreps_local   <- switch(run_level,    3,   15,   20)
sc_Nreps_global  <- switch(run_level,    4,   20,   20)
sc_cooling       <- 0.5

cores <- as.numeric(Sys.getenv("CORES", unset = NA))
if (is.na(cores)) cores <- max(1, parallel::detectCores() - 1)
cat(sprintf("\nUsing %d cores for parallel computation\n", cores))
registerDoParallel(cores)
registerDoRNG(34118892)


# ============================================================================
# 6. Initial panel particle filter
# ============================================================================

stew(file = file.path(OUT_DIR, sprintf("pf_init_panel_%d.rda", run_level)), {
  pf_init <- foreach(i = 1:sc_Nreps_eval, .packages = "panelPomp") %dopar%
    pfilter(pp, Np = sc_Np)
})
ll_init <- logmeanexp(sapply(pf_init, logLik), se = TRUE)
cat(sprintf("\nInitial panel log-likelihood: %.2f (se %.3f)\n",
            ll_init[1], ll_init[2]))


# ============================================================================
# 7. Local panel IF2
# ============================================================================

# rw.sd applied to shared AND specific params
# sigma_eta, delta, gamma get tighter perturbations to prevent IF2 drift into
# numerical blowup regions (H spiraling to +/- infinity).
sc_rw_sd <- rw_sd(
  sigma_nu  = 0.02,
  phi       = 0.02,
  sigma_eta = 0.01,
  gamma     = 0.01,
  delta     = 0.01,
  nu        = 0.02,
  mu_h      = 0.02,
  G_0       = ivp(0.1),
  H_0       = ivp(0.1)
)

stew(file = file.path(OUT_DIR, sprintf("mif_local_panel_%d.rda", run_level)), {
  mif_local <- foreach(i = 1:sc_Nreps_local,
                       .packages = "panelPomp",
                       .combine = c) %dopar%
    mif2(pp,
         Np = sc_Np,
         Nmif = sc_Nmif,
         cooling.fraction.50 = sc_cooling,
         rw.sd = sc_rw_sd)
  
  L_local <- foreach(i = 1:sc_Nreps_local,
                     .packages = "panelPomp",
                     .combine = rbind) %dopar% {
                       logmeanexp(
                         replicate(sc_Nreps_eval,
                                   logLik(pfilter(mif_local[[i]], Np = sc_Np))),
                         se = TRUE)
                     }
})

r_local <- data.frame(
  logLik = L_local[, 1], logLik_se = L_local[, 2],
  t(sapply(mif_local, function(m) {
    sp <- specific(m)
    sp_vec <- as.vector(sp)
    names(sp_vec) <- paste(rep(rownames(sp), ncol(sp)),
                           rep(colnames(sp), each = nrow(sp)),
                           sep = "_")
    c(shared(m), sp_vec)
  }))
)
write.csv(r_local, file.path(OUT_DIR, "panel_local_search.csv"), row.names = FALSE)
cat("\nLocal panel search (top 5):\n")
print(head(r_local[order(-r_local$logLik), ], 5))


# ============================================================================
# 8. Global panel IF2 — random starts across parameter box
# ============================================================================

# Wider box for shared dynamics; per-unit intercepts centered on mu_h_inits.
mu_h_lo <- mu_h_inits - 2
mu_h_hi <- mu_h_inits + 2

sample_start <- function() {
  shared_rand <- c(
    sigma_nu  = runif(1, 0.005, 0.05),
    phi       = runif(1, 0.88, 0.98),
    sigma_eta = runif(1, 0.5, 3.0),
    gamma     = runif(1, -0.3, 0.3),
    delta     = runif(1, 0, 1.0),
    nu        = runif(1, 5, 15)
  )
  specific_rand <- rbind(
    mu_h = mapply(runif, n = 1, min = mu_h_lo, max = mu_h_hi),
    G_0  = runif(length(TRAIN_ASSETS), -2, 2),
    H_0  = mapply(runif, n = 1, min = mu_h_lo, max = mu_h_hi)
  )
  colnames(specific_rand) <- TRAIN_ASSETS
  list(shared = shared_rand, specific = specific_rand)
}

stew(file = file.path(OUT_DIR, sprintf("mif_global_panel_%d.rda", run_level)), {
  mif_global <- foreach(i = 1:sc_Nreps_global,
                        .packages = "panelPomp",
                        .combine = c) %dopar% {
                          start_i <- sample_start()
                          pp_i <- panelPomp(
  train_units,
  shared   = start_i$shared,  
  specific = start_i$specific 
)                         
                        mif2(pp_i,
                               Np = sc_Np, Nmif = sc_Nmif,
                               cooling.fraction.50 = sc_cooling,
                               rw.sd = sc_rw_sd)
                        }
  
  L_global <- foreach(i = 1:sc_Nreps_global,
                      .packages = "panelPomp",
                      .combine = rbind) %dopar% {
                        logmeanexp(
                          replicate(sc_Nreps_eval,
                                    logLik(pfilter(mif_global[[i]], Np = sc_Np))),
                          se = TRUE)
                      }
})

r_global <- data.frame(
  logLik = L_global[, 1], logLik_se = L_global[, 2],
  t(sapply(mif_global, function(m) {
    sp <- specific(m)
    sp_vec <- as.vector(sp)
    names(sp_vec) <- paste(rep(rownames(sp), ncol(sp)),
                           rep(colnames(sp), each = nrow(sp)),
                           sep = "_")
    c(shared(m), sp_vec)
  }))
)
write.csv(r_global, file.path(OUT_DIR, "panel_global_search.csv"), row.names = FALSE)
cat("\nGlobal panel search logLik summary:\n")
print(summary(r_global$logLik, digits = 5))

trace_list <- lapply(seq_along(mif_global), function(i) {
  tr <- as.data.frame(traces(mif_global[[i]]))
  tr$iteration <- 0:(nrow(tr) - 1)
  tr$replicate <- i
  tr
})
traces_df <- bind_rows(trace_list)
shared_par_names <- c("sigma_nu", "phi", "sigma_eta", "gamma", "delta", "nu", "loglik")
plot_df <- traces_df %>%
  select(iteration, replicate, any_of(shared_par_names)) %>%
  pivot_longer(cols = -c(iteration, replicate),
               names_to = "parameter", values_to = "value")

p_traces <- ggplot(plot_df,
                   aes(x = iteration, y = value, group = replicate)) +
  geom_line(alpha = 0.35, colour = "steelblue") +
  facet_wrap(~parameter, scales = "free_y", ncol = 2) +
  labs(title = "IF2 convergence traces — shared parameters",
       subtitle = sprintf("%d global replicates, %d IF2 iterations",
                          length(mif_global), sc_Nmif),
       x = "IF2 iteration", y = NULL) +
  theme_bw(base_size = 11)

ggsave(file.path(OUT_DIR, "convergence_traces_shared.png"),
       p_traces, width = 10, height = 8, dpi = 150)

mu_h_cols <- grep("^mu_h", names(traces_df), value = TRUE)
if (length(mu_h_cols) > 0) {
  mu_h_df <- traces_df %>%
    select(iteration, replicate, all_of(mu_h_cols)) %>%
    pivot_longer(cols = all_of(mu_h_cols),
                 names_to = "asset", values_to = "mu_h") %>%
    mutate(asset = gsub("mu_h\\.|mu_h\\[|\\]", "", asset))
  
  p_muh <- ggplot(mu_h_df, aes(iteration, mu_h, group = replicate)) +
    geom_line(alpha = 0.35, colour = "darkgreen") +
    facet_wrap(~asset, scales = "free_y") +
    labs(title = "IF2 traces — asset-specific mu_h",
         x = "IF2 iteration", y = "mu_h") +
    theme_bw(base_size = 11)
  ggsave(file.path(OUT_DIR, "08b_convergence_traces_muh.png"),
         p_muh, width = 10, height = 6, dpi = 150)
}

# ============================================================================
# 9. Best panel fit and diagnostics
# ============================================================================

best_idx <- which.max(r_global$logLik)
best_mif <- mif_global[[best_idx]]
best_shared <- shared(best_mif)
best_specific <- specific(best_mif)

cat("\n=== BEST PANEL FIT ===\n")
cat(sprintf("logLik = %.2f (se %.3f)\n",
            r_global$logLik[best_idx], r_global$logLik_se[best_idx]))
cat("\nShared parameters:\n")
print(round(best_shared, 4))
cat("\nAsset-specific parameters:\n")
print(round(best_specific, 4))

ll_vec <- sapply(mif_global, function(m) tryCatch(logLik(m), error = function(e) -Inf))
best_panel <- mif_global[[which.max(ll_vec)]]
cat("best_panel resolved: logLik =", round(logLik(best_panel), 2), "\n")

# Pairs plot of shared params across top replicates
top_cutoff <- max(r_global$logLik) - 10
r_top <- subset(r_global, logLik > top_cutoff)
png(file.path(OUT_DIR, "panel_pairs_shared.png"), width = 2400, height = 2400, res = 200)
pairs(~logLik + phi + sigma_eta + sigma_nu + gamma + delta + nu,
      data = r_top, main = "Shared parameters, top replicates")
dev.off()


# ============================================================================
# 10. Extract filtered states for each training asset
# ============================================================================

cat("\nExtracting filtered states for training assets...\n")
for (asset in TRAIN_ASSETS) {
  u_pomp <- unit_objects(best_mif)[[asset]]
  sp_col <- best_specific[, asset]
  names(sp_col) <- rownames(best_specific)
  u_params <- c(best_shared, sp_col)
  pf_u <- pfilter(u_pomp, params = u_params, Np = sc_Np * 2, filter.mean = TRUE)
  
  ess <- eff_sample_size(pf_u)
  cond_loglik <- cond_logLik(pf_u)
  
  diag_df <- data.frame(
    timestamp = attr(train_units[[asset]], "dates"),
    ess = ess,
    cond_loglik = cond_loglik,
    in_depeg = attr(train_units[[asset]], "in_depeg")
  )
  
  p_ess <- ggplot(diag_df, aes(timestamp, ess)) +
    geom_line(alpha = 0.7) +
    geom_point(data = subset(diag_df, in_depeg == 1),
               aes(y = ess), colour = "red", size = 0.8, alpha = 0.6) +
    geom_hline(yintercept = sc_Np / 2, linetype = "dashed", colour = "grey50") +
    labs(title = sprintf("%s: particle filter ESS over time", asset),
         subtitle = "Red points = known depeg days; dashed line = Np/2",
         x = NULL, y = "Effective sample size") +
    theme_bw(base_size = 11)
  
  p_cll <- ggplot(diag_df, aes(timestamp, cond_loglik)) +
    geom_line(alpha = 0.7, colour = "darkred") +
    labs(title = sprintf("%s: conditional log-likelihood", asset),
         x = NULL, y = "log p(y_n | y_{1:n-1})") +
    theme_bw(base_size = 11)
  
  ggsave(file.path(OUT_DIR, sprintf("_filter_diagnostics_%s.png", asset)),
         p_ess / p_cll, width = 10, height = 6, dpi = 150)
  
  fm <- filter_mean(pf_u)
  
  out_df <- data.frame(
    timestamp       = attr(train_units[[asset]], "dates"),
    y_bp            = obs(u_pomp)[1, ],
    filtered_H      = as.numeric(fm["H", ]),
    filtered_G      = as.numeric(fm["G", ]),
    filtered_vol_bp = as.numeric(exp(fm["H", ] / 2)),
    in_depeg        = attr(train_units[[asset]], "in_depeg"),
    depeg_severity  = attr(train_units[[asset]], "depeg_severity")
  )
  write.csv(out_df,
            file.path(OUT_DIR, sprintf("filtered_states_%s.csv", asset)),
            row.names = FALSE)
  cat(sprintf("  %s: %d rows written\n", asset, nrow(out_df)))
  
  p_filt <- ggplot(out_df, aes(timestamp)) +
    geom_line(aes(y = y_bp), colour = "grey40", alpha = 0.6) +
    geom_line(aes(y = filtered_vol_bp), colour = "firebrick", linewidth = 0.5) +
    geom_line(aes(y = -filtered_vol_bp), colour = "firebrick",
              linewidth = 0.5, alpha = 0.6) +
    geom_point(data = subset(out_df, in_depeg == 1),
               aes(y = y_bp), colour = "red", size = 0.8) +
    labs(title = sprintf("%s: observed y_bp with filtered ±1-sigma envelope", asset),
         subtitle = "Red points = labelled depeg days; red lines = ±exp(H/2)",
         x = NULL, y = "basis points") +
    theme_bw(base_size = 11)
  
  ggsave(file.path(OUT_DIR, sprintf("10_filtered_overlay_%s.png", asset)),
         p_filt, width = 10, height = 4, dpi = 150)
}


# ============================================================================
# 11. UST HELD-OUT TEST — two-step procedure
# ============================================================================
# Step 1: calibrate UST-specific (mu_h, G_0, H_0) on Nov 2020 - Jan 2022 only,
#         with shared params fixed at MLE.
# Step 2: extend particle filter through May 10, 2022 to evaluate signal.

cat("\n=== UST HELD-OUT TEST ===\n")

# --- Build UST pomp for calibration window only ---
ust_calib <- build_unit_pomp("UST", panel_all, date_max = UST_CALIB_END)

# --- Build UST pomp for full eval window (calib + stress) ---
ust_eval <- build_unit_pomp("UST", panel_all, date_max = UST_EVAL_END)

# --- Step 1: IF2 on UST calibration data with shared params frozen ---
# rw.sd perturbs only specific params; shared params get sd=0
ust_calib_rw_sd <- rw_sd(
  sigma_nu  = 0,
  phi       = 0,
  sigma_eta = 0,
  gamma     = 0,
  delta     = 0,
  nu        = 0,
  mu_h      = 0.02,
  G_0       = ivp(0.1),
  H_0       = ivp(0.1)
)

ust_start <- c(
  best_shared,
  mu_h = 2 * log(sd(obs(ust_calib)[1, ])),
  G_0 = 0,
  H_0 = 2 * log(sd(obs(ust_calib)[1, ]))
)

stew(file = file.path(OUT_DIR, sprintf("mif_ust_calib_%d.rda", run_level)), {
  ust_mif <- foreach(i = 1:10, .packages = "pomp", .combine = c) %dopar%
    mif2(ust_calib, params = ust_start, Np = sc_Np,
         Nmif = sc_Nmif, cooling.fraction.50 = sc_cooling,
         rw.sd = ust_calib_rw_sd)
  
  ust_ll <- foreach(i = 1:10, .packages = "pomp", .combine = c) %dopar%
    logmeanexp(
      replicate(sc_Nreps_eval,
                logLik(pfilter(ust_calib, params = coef(ust_mif[[i]]),
                               Np = sc_Np))))
})

ust_best_idx <- which.max(ust_ll)
ust_best_params <- coef(ust_mif[[ust_best_idx]])
cat("\nUST calibrated specific params (shared frozen at panel MLE):\n")
print(round(ust_best_params[c("mu_h", "G_0", "H_0")], 4))

# --- Step 2: Filter through full eval window at the calibrated params ---
ust_pf <- pfilter(ust_eval, params = ust_best_params,
                  Np = sc_Np * 2, filter.mean = TRUE)
ust_fm <- filter_mean(ust_pf)

ust_out <- data.frame(
  timestamp       = attr(ust_eval, "dates"),
  y_bp            = obs(ust_eval)[1, ],
  filtered_H      = as.numeric(ust_fm["H", ]),
  filtered_G      = as.numeric(ust_fm["G", ]),
  filtered_vol_bp = as.numeric(exp(ust_fm["H", ] / 2)),
  in_depeg        = attr(ust_eval, "in_depeg"),
  depeg_severity  = attr(ust_eval, "depeg_severity")
)
ust_out$window <- ifelse(ust_out$timestamp <= UST_CALIB_END, "calibration", "held-out")

write.csv(ust_out, file.path(OUT_DIR, "filtered_states_UST.csv"), row.names = FALSE)
cat(sprintf("UST filtered states: %d rows (%d calibration + %d held-out)\n",
            nrow(ust_out),
            sum(ust_out$window == "calibration"),
            sum(ust_out$window == "held-out")))

# --- UST diagnostic plot ---
png(file.path(OUT_DIR, "UST_held_out_test.png"), width = 2400, height = 1800, res = 200)
par(mfrow = c(3, 1), mai = c(0.5, 0.9, 0.3, 0.3))
plot(ust_out$timestamp, ust_out$y_bp, type = "l",
     xlab = "", ylab = "UST y_bp",
     main = "UST held-out test (calibration ends 2022-01-31)")
abline(v = UST_CALIB_END, col = "red", lty = 2)
plot(ust_out$timestamp, ust_out$filtered_H, type = "l", col = "darkgreen",
     xlab = "", ylab = "filtered H (log-var)",
     main = "Filtered arbitrage-inefficiency state")
abline(v = UST_CALIB_END, col = "red", lty = 2)
plot(ust_out$timestamp, ust_out$filtered_vol_bp, type = "l", col = "firebrick",
     xlab = "Date", ylab = "filtered vol (bp)",
     main = "Filtered volatility — does signal fire before May 2022 collapse?")
abline(v = UST_CALIB_END, col = "red", lty = 2)
abline(v = as.Date("2022-05-09"), col = "blue", lty = 3)
dev.off()


cat("\n=== MIM HELD-OUT TEST ===\n")

# Calibration-only unit for mu_h / G_0 / H_0 fit
mim_calib <- build_unit_pomp("MIM", panel_all, date_max = MIM_CALIB_END)

if (!exists("best_panel")) {
  ll_vec <- sapply(mif_global, function(m) tryCatch(logLik(m), error = function(e) -Inf))
  best_panel <- mif_global[[which.max(ll_vec)]]
  cat("Recovered best_panel from mif_global (logLik =", round(logLik(best_panel), 2), ")\n")
}

# Freeze shared params at panel MLE
shared_mle <- shared(best_panel)

# Fit unit-specific params via IF2 on MIM calibration data
mim_sp_start <- c(mu_h = 5.0, G_0 = 0.0, H_0 = 7.0)
mim_pp <- panelPomp(
  units    = list(MIM = mim_calib),
  shared   = shared_mle,
  specific = matrix(mim_sp_start, ncol=1, dimnames=list(names(mim_sp_start), "MIM"))
)

mim_fit <- mif2(
  mim_pp,
  Nmif           = 200,
  Np             = 2000,
  cooling.fraction.50 = 0.5,
  rw.sd          = rw_sd(mu_h = 0.02, G_0 = 0.02, H_0 = 0.02)  # only unit-specific
)

mim_specific_mle <- specific(mim_fit)[, "MIM"]
cat("MIM calibrated specific params (shared frozen at panel MLE):\n")
print(round(mim_specific_mle, 4))

# Build full-window MIM unit for filtering through the depeg
mim_full <- build_unit_pomp("MIM", panel_all, date_max = MIM_TEST_END)

# Run pfilter with frozen params through the full window
mim_full_params <- c(shared_mle, mim_specific_mle)
mim_pfilt <- pfilter(
  mim_full,
  params = mim_full_params,
  Np     = 4000,
  save.states = "filter"
)

# Extract filtered H and observations
dates_mim <- index(as.data.frame(mim_full))
y_bp_mim  <- obs(mim_full)["y",]
H_filt_mim <- sapply(saved.states(mim_pfilt), function(s) weighted.mean(s["H",], s[".weight",]))
vol_bp_mim <- exp(H_filt_mim / 2)

# Save states
write.csv(
  data.frame(date = dates_mim, y_bp = y_bp_mim, H_filt = H_filt_mim, vol_bp = vol_bp_mim),
  "output/filtered_states_MIM_heldout.csv",
  row.names = FALSE
)

cat(sprintf("MIM filtered states: %d rows (%d calib + %d held-out)\n",
            length(H_filt_mim),
            sum(dates_mim <= MIM_CALIB_END),
            sum(dates_mim >  MIM_CALIB_END)))

# Plot (similar to UST)
png("output/MIM_held_out_test.png", width=1200, height=800, res=100)
par(mfrow = c(3,1), mar=c(3,4,2,1))
plot(dates_mim, y_bp_mim, type="l", main="MIM held-out test (calib ends 2022-10-31)",
     ylab="MIM y_bp")
abline(v = MIM_CALIB_END, col="red", lty=2)

plot(dates_mim, H_filt_mim, type="l", col="darkgreen",
     main="Filtered H (log-variance)", ylab="filtered H")
abline(v = MIM_CALIB_END, col="red", lty=2)

plot(dates_mim, vol_bp_mim, type="l", col="darkred",
     main="Filtered vol - does signal fire in Nov 2022 FTX fallout?",
     ylab="filtered vol (bp)")
abline(v = MIM_CALIB_END, col="red", lty=2)
abline(v = as.Date("2022-11-08"), col="blue", lty=3)  # FTX collapse day
dev.off()

# ============================================================================
# 12. Write summary
# ============================================================================

summary_df <- data.frame(
  panel_logLik    = r_global$logLik[best_idx],
  panel_logLik_se = r_global$logLik_se[best_idx],
  t(best_shared),
  n_train_assets  = length(TRAIN_ASSETS),
  n_ust_calib     = sum(ust_out$window == "calibration"),
  n_ust_heldout   = sum(ust_out$window == "held-out")
)
write.csv(summary_df, file.path(OUT_DIR, "panel_fit_summary.csv"), row.names = FALSE)

cat("\n=== DONE ===\n")
cat(sprintf("Outputs in %s/\n", OUT_DIR))
cat("  panel_global_search.csv     : all IF2 replicates\n")
cat("  panel_fit_summary.csv       : shared MLE + metadata\n")
cat("  filtered_states_<ASSET>.csv : filtered H/G/vol per asset (4 train + UST)\n")
cat("  panel_pairs_shared.png      : shared-parameter posterior-like scatter\n")
cat("  UST_held_out_test.png       : UST signal with calibration/eval split\n")
