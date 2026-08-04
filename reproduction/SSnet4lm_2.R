SSnet4lm_2 <- function(X, y, nla = 100, nmin = .0001, iter, pen){
  y=y-mean(y)
  Xg <- apply(X, 2, function(x) sqrt(length(y)/sum(x^2))*x)
  lmax <- max(abs(2*t(Xg)%*%y))/length(y); lmin <- lmax*nmin
  La0 <- exp(seq(log(lmax), log(lmin), length.out = nla)) 
  betaL <- glmnet(Xg, y, alpha = 1, intercept = FALSE, lambda=La0)$beta
  dfy <- apply(betaL, 2, function(x) sum(x!=0))
  kt <- 1:nla; kt <- which( dfy < nrow(X) & dfy > 0) 
  BB <- as.matrix(abs(betaL[, kt]))
  II <- duplicated( t(ifelse(BB > 0, 1, 0)) )
  BB <- BB[,II == FALSE, drop = FALSE]
  mm <- lapply(1:ncol(BB), function(i) {
       b <- BB[,i]; screenPred <- which(b != 0); b <- b[b != 0]
       ind <- sort(b, decreasing = TRUE, index.return = TRUE)$ix
       QR <- qr(X[, screenPred[ind], drop = FALSE])
       q2 <- (t(qr.Q(QR))%*%y)^2
       l <- length(b); rss <- rep(NA, l); rss[1] <- sum(y^2) - q2[1]
       if (l > 1){ for (k in 2:l) rss[k] <- rss[k-1] - q2[k] }
       R2 <- 1 - rss/sum(y^2)
       return(list(R2 = R2, ind = screenPred[ind]))
     })
  maxl <- max(sapply(1:length(mm), function(i) length(mm[[i]]$R2)))
  r2 <- sapply(1:length(mm), function(i) c( mm[[i]]$R2, rep(0, maxl - length(mm[[i]]$R2)) ))
  ind0 <- apply(r2, 1, which.max); R2 <-  apply(r2, 1, max)
  
  #p <- min( ncol(X), ceiling(nrow(X)/2), which(R2/R2[length(R2)] > .985)[1] )
  
  p <- min( ncol(X), ceiling(nrow(X)/2),min(length(mm),length(R2)) )

  R2 <-R2[1:p]
  nosnik <- NULL
  Nosniki <- sapply(1:p, function(i) { 
            nos0 <- sort(mm[[ind0[i]]]$ind[1:i])
            nosnik <<- union(nosnik, nos0)
            return(nos0)
          })

nosnik=nosnik[nosnik %in% Nosniki[[length(Nosniki)]]]

s=length(R2)
sS=s
p=ncol(Xg)
n=nrow(Xg)

# GIC z sigma nieznane - nie trzeba iterowac

#n*log(1-R2)+(1:s)* 2.5*log(p)

#dodany model pusty
k2=which.min(c(n*log(1-R2)+(1:s)* 2.5*log(p*pen),0))
#######################################
k1=which.min( c(1-R2 + (1:s)* 2.5*log(p*pen)/(n-s)*(1-R2[s]),1) )


iter1=0

ss=as.list(iter)

while (iter1<iter){
iter1=iter1+1

kk=which.min( c(1-R2 + (1:s)* 2.5*log(p*pen)/(n-s)*(1-R2[s]),1) )

if (kk==sS+1) break

QR <- qr(Xg[,  nosnik[1:kk], drop = FALSE])
       q2 <- (t(qr.Q(QR))%*%y)^2
       l <- length(Nosniki[[kk]]); rss <- rep(NA, l); rss[1] <- sum(y^2) - q2[1]
       if (l > 1){ for (k in 2:l) rss[k] <- rss[k-1] - q2[k] }
       R2 <- 1 - rss/sum(y^2)
s=kk
ss[[iter1]]=kk
}

if (k2==(sS+1)){support_bez=0} else {support_bez=Nosniki[[k2]]}
if (k1==(sS+1)){support_1=0} else {support_1=Nosniki[[k1]]}
if (kk==(sS+1)){support_iter=0} else {support_iter=Nosniki[[kk]]}

   return( list(support_iter = support_iter, support_1 = support_1,
support_bez=support_bez))
}