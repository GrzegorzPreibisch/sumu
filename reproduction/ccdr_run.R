# CCDr (Aragam & Zhou, JMLR 2015) on the same rbn replications as everything else.
# usage: Rscript ccdr_run.R NET NREPS "300,1000"
#
# Model selection uses sparsebnUtils::select.parameter -- the package's own
# recommended selector. We deliberately do NOT pick the point on the solution path
# with the smallest SHD: that would be oracle tuning and would flatter CCDr.

.libPaths("../rlib")
suppressMessages({library(bnlearn); library(ccdrAlgorithm); library(sparsebnUtils)})

args  <- commandArgs(trailingOnly = TRUE)
net   <- args[1]; nreps <- as.integer(args[2])
sizes <- as.integer(strsplit(args[3], ",")[[1]])

load(sprintf("../bnrepo/%s.rda", net))
true_g <- model2network(modelstring(bn)); nm <- nodes(true_g)
true   <- amat(true_g)                                   # [from, to]
cat(net, ": ", length(nm), " nodes, ", sum(true), " arcs\n", sep = "")

mk <- function(M) { g <- empty.graph(nm); dimnames(M) <- list(nm, nm); amat(g) <- M; g }
out <- NULL

for (n in sizes) for (k in 1:nreps) {
  d <- read.csv(sprintf("rbn/%s_%d_rep%d.csv", net, n, k - 1), sep = ";")
  d <- d[, nm]
  sbd <- sparsebnData(d, type = "continuous")

  fit <- tryCatch(ccdr.run(data = sbd, lambdas.length = 20, alpha = 10,
                           verbose = FALSE),
                  error = function(e) { cat("ERR", conditionMessage(e), "\n"); NULL })
  if (is.null(fit)) next

  sel <- tryCatch(select.parameter(fit, sbd), error = function(e) NA)
  if (is.na(sel[1])) sel <- length(fit)                  # fallback: densest
  est <- as.matrix(get.adjacency.matrix(fit[[sel]]))
  est <- ifelse(est != 0, 1, 0)
  storage.mode(est) <- "numeric"
  diag(est) <- 0

  tp <- sum(true * est)
  out <- rbind(out, data.frame(
    net = net, n = n, rep = k - 1, method = "CCDr",
    power = tp / sum(true),
    fdr   = (sum(est) - tp) / max(sum(est), 1),
    shd   = shd(mk(est), true_g),
    edges = sum(est), lambda_idx = sel))
  cat(".")
}
cat("\n")
write.csv(out, sprintf("comp/%s_ccdr.csv", net), row.names = FALSE)
print(aggregate(cbind(power, fdr, shd, edges) ~ n, out, mean), digits = 3)
