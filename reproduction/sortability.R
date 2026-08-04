# Analytic varsortability and R^2-sortability for the four bnlearn Gaussian networks.
#
# For a linear Gaussian bn.fit object with X_v = sum_u A[v,u] X_u + eps_v,
#   X = (I - A)^{-1} eps   =>   Sigma = (I - A)^{-1} diag(sigma^2) (I - A)^{-T}
# so both quantities follow in closed form from the published coefficients --
# no simulation, no sampling error.
#
# varsortability (Reisach, Seiler & Weichwald 2021): over all directed paths,
# the fraction along which the marginal variance increases. 1.0 means sorting the
# nodes by variance recovers a valid causal order; 0.5 is chance.
# R^2-sortability (Reisach et al. 2023): same, with R^2 of each node regressed on
# all others in place of the variance. Standardising the data kills varsortability
# but leaves R^2-sortability untouched.

.libPaths("../rlib")
suppressMessages(library(bnlearn))

sortability <- function(crit, A_adj) {
  # crit: per-node criterion; A_adj[u, v] = 1 iff u -> v
  d <- nrow(A_adj)
  num <- 0; den <- 0
  P <- A_adj
  for (k in 1:(d - 1)) {
    idx <- which(P > 0, arr.ind = TRUE)
    if (nrow(idx) == 0) break
    for (r in seq_len(nrow(idx))) {
      u <- idx[r, 1]; v <- idx[r, 2]; w <- P[u, v]
      den <- den + w
      num <- num + w * (if (crit[u] < crit[v]) 1 else if (crit[u] == crit[v]) 0.5 else 0)
    }
    P <- P %*% A_adj
  }
  num / den
}

nets <- c("ecoli70", "magic-niab", "magic-irri", "arth150")
res <- NULL

for (net in nets) {
  load(sprintf("../bnrepo/%s.rda", net))          # -> bn (a bn.fit object)
  nm <- names(bn); d <- length(nm)
  idx <- setNames(seq_along(nm), nm)

  A <- matrix(0, d, d, dimnames = list(nm, nm))   # A[v, u] = coef of parent u in node v
  s2 <- numeric(d)
  for (v in nm) {
    node <- bn[[v]]
    s2[idx[v]] <- node$sd^2
    co <- node$coefficients
    pa <- names(co)[names(co) != "(Intercept)"]
    if (length(pa)) A[idx[v], idx[pa]] <- as.numeric(co[pa])
  }

  IA    <- solve(diag(d) - A)
  Sigma <- IA %*% diag(s2) %*% t(IA)
  Theta <- solve(Sigma)
  vars  <- diag(Sigma)
  R2    <- 1 - 1 / (diag(Sigma) * diag(Theta))    # R^2 of X_v on all others

  A_adj <- t(ifelse(A != 0, 1, 0))                # A_adj[u, v] = 1 iff u -> v
  dimnames(A_adj) <- list(nm, nm)

  res <- rbind(res, data.frame(
    network = net, nodes = d, arcs = sum(A_adj),
    varsortability = round(sortability(vars, A_adj), 3),
    R2sortability  = round(sortability(R2,   A_adj), 3),
    var_ratio_max_min = round(max(vars) / min(vars), 1)))
}

print(res, row.names = FALSE)
cat("\n0.5 = chance. Reisach et al. report ~0.94+ for the standard",
    "iid-random-weight simulations that the critique targets.\n")
write.csv(res, "sortability.csv", row.names = FALSE)
