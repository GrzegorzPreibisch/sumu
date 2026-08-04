.libPaths("../rlib")
source_sortability <- function() NULL
sortability <- function(crit, A_adj) {
  d <- nrow(A_adj); num <- 0; den <- 0; P <- A_adj
  for (k in 1:(d-1)) {
    idx <- which(P > 0, arr.ind = TRUE); if (nrow(idx)==0) break
    for (r in seq_len(nrow(idx))) {
      u<-idx[r,1]; v<-idx[r,2]; w<-P[u,v]; den<-den+w
      num <- num + w*(if (crit[u]<crit[v]) 1 else if (crit[u]==crit[v]) 0.5 else 0)
    }
    P <- P %*% A_adj
  }
  num/den
}
# Standardowa symulacja, ktorej dotyczy krytyka: ER-2, wagi iid U(0.5,2) z losowym znakiem,
# szum jednostkowy. Reisach i in. raportuja tu varsortability ~0.94.
set.seed(1); d <- 50; vs <- numeric(20)
for (t in 1:20) {
  ord <- sample(d); A <- matrix(0,d,d)
  for (a in 1:(d-1)) for (b in (a+1):d)
    if (runif(1) < 4/(d-1)/2) {           # ER-2: srednio 2 krawedzie na wezel
      w <- runif(1,0.5,2)*sample(c(-1,1),1)
      A[ord[b], ord[a]] <- w              # A[dziecko, rodzic]
    }
  IA <- solve(diag(d)-A); Sig <- IA %*% diag(rep(1,d)) %*% t(IA)
  vs[t] <- sortability(diag(Sig), t(ifelse(A!=0,1,0)))
}
cat(sprintf("ER-2, d=50, wagi iid, szum jednostkowy: varsortability = %.3f (sd %.3f)\n",
            mean(vs), sd(vs)))
cat("Reisach i in. (2021) raportuja ~0.94 dla tego ustawienia.\n")
