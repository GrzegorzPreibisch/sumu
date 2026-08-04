# Generate replications from the TRUE bnlearn networks with rbn(), run the four
# competitors, and write both the data (for the Python side) and the scores.
# Metric definitions taken verbatim from Wojtek's SS_do_nizszych.R.

.libPaths("../rlib")
suppressMessages(library(bnlearn))

args  <- commandArgs(trailingOnly = TRUE)
net   <- args[1]
nreps <- as.integer(args[2])
sizes <- as.integer(strsplit(args[3], ",")[[1]])

load(sprintf("../bnrepo/%s.rda", net))          # provides `bn`
true_g <- model2network(modelstring(bn))
nm     <- nodes(true_g)
true   <- amat(true_g)                          # [from, to]
cat(net, ": ", length(nm), " nodes, ", sum(true), " arcs\n", sep = "")

dir.create("rbn", showWarnings = FALSE)
dir.create("comp", showWarnings = FALSE)

methods <- list(hc = hc, tabu = tabu, mmhc = mmhc, h2pc = h2pc)
out <- NULL

for (n in sizes) {
  for (k in 1:nreps) {
    set.seed(1000 * n + k)
    d <- rbn(bn, n)
    d <- d[, nm]
    write.table(d, sprintf("rbn/%s_%d_rep%d.csv", net, n, k - 1),
                sep = ";", row.names = FALSE, col.names = TRUE)

    for (mn in names(methods)) {
      est_g <- tryCatch(methods[[mn]](d), error = function(e) NULL)
      if (is.null(est_g)) next
      est <- amat(est_g)
      tp  <- sum(true * est)
      power <- tp / sum(true)
      fdr   <- (sum(est) - tp) / max(sum(est), 1)
      out <- rbind(out, data.frame(net = net, n = n, rep = k - 1, method = mn,
                                   power = power, fdr = fdr,
                                   shd = shd(est_g, true_g)))
    }
    cat(".", if (k %% 10 == 0) sprintf(" n=%d rep=%d\n", n, k) else "")
  }
}

write.csv(out, sprintf("comp/%s.csv", net), row.names = FALSE)
cat("\n=== ", net, " means ===\n", sep = "")
print(aggregate(cbind(power, fdr, shd) ~ method + n, out, mean), digits = 3)
