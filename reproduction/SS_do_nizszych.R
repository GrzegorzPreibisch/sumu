library(bnlearn)
library(glmnet)
library(rjson)

a=file.choose()

#true[i,j] != 0 <=> strzalka z i do j

true=t(read.csv(a,header=FALSE, sep=";"))

true_g = empty.graph(sprintf("%s", 1:dim(true)[1]))
w=matrix(as.matrix(true),nrow=dim(true)[1])
amat(true_g)=w

t_index=which(true!=0, arr.ind = T)

t_index
t_index[order(t_index[,1]),]

library(igraph)

true_mat=graph_from_adjacency_matrix(as.matrix(true), mode = "directed", 
  weighted = NULL, diag = FALSE, add.colnames = NULL, add.rownames = NA)

test.layout <- layout_(true_mat,with_dh(weight.edge.lengths = edge_density(true_mat)/1000))
plot(true_mat, layout = test.layout,vertex.size=4,
     vertex.label.dist=1, vertex.color="red", edge.arrow.size=0.5)
plot(true_mat)
plot(true_mat, layout=layout_with_fr, vertex.size=4,
     vertex.label.dist=1, vertex.color="red", edge.arrow.size=0.5)

###############################################

files <- list.files(pattern="sample.*csv")
files2 <- list.files(pattern="arth150_layers.*json")

power_1=0
fdr_1=0
power_iter=0
fdr_iter=0
power_bez=0
fdr_bez=0
sham_1=0
sham_iter=0
sham_bez=0
ham_1=0
ham_iter=0
ham_bez=0


estim_1_g = empty.graph(sprintf("%s", 1:dim(true)[1]))
estim_iter_g = empty.graph(sprintf("%s", 1:dim(true)[1]))
estim_bez_g = empty.graph(sprintf("%s", 1:dim(true)[1]))

for (k in 1:100){ 

sample=read.csv(files[k],header=FALSE, sep=";")

q=fromJSON(file=files2[k])

qq=matrix(0,nrow=length(q)+1,ncol=dim(true)[1])

for (j in 1:length(q)){
qq[j,]=c(q[[j]], rep(0,dim(true)[1]-length(q[[j]])))
}


#strzalki do nizszych

estim_1=matrix(0, nrow=dim(true)[1], ncol=dim(true)[1])
estim_iter=matrix(0, nrow=dim(true)[1], ncol=dim(true)[1])
estim_bez=matrix(0, nrow=dim(true)[1], ncol=dim(true)[1])

w1=qq[1,]
w1=w1[w1!=0]


r=(1:dim(true)[1])[-w1]

for (i in r){

w=as.vector(qq[1:(which(qq==i,arr.ind = T)[1,1]-1),])
w=w[w!=0]
y=sample[,i]
X=sample[,w]

if (length(w)==1){
	val=summary(lm(y~X))$coefficients[2,4]<0.01
	estim_1[w,i]=val
	estim_iter[w,i]=val
	estim_bez[w,i]=val} else {

XX=scale(as.matrix(X))

#kara: lambda^2 ~ log(pen*liczba regresorow)

#pen=1
pen=min(length(q)*length(which(qq[(which(qq==i,arr.ind = T)[1,1]),] !=0)),dim(true)[1])

SS=SSnet4lm_2(XX, y, iter=10,pen=pen)


w1=w[SS$support_1]
w2=w[SS$support_iter]

estim_1[w1,i]=1
estim_iter[w2,i]=1

if (SS$support_bez[1] !=0){
w3=w[SS$support_bez]
estim_bez[w3,i]=1}

}}


power_1[k]=sum(true*estim_1)/sum(true)
fdr_1[k]=(sum(estim_1)-sum(true*estim_1))/max(sum(estim_1),1)

power_iter[k]=sum(true*estim_iter)/sum(true)
fdr_iter[k]=(sum(estim_iter)-sum(true*estim_iter))/max(sum(estim_iter),1)

power_bez[k]=sum(true*estim_bez)/sum(true)
fdr_bez[k]=(sum(estim_bez)-sum(true*estim_bez))/max(sum(estim_bez),1)

amat(estim_1_g)=estim_1
sham_1[k]=shd(estim_1_g,true_g)
ham_1[k]=hamming(estim_1_g,true_g)

amat(estim_iter_g)=estim_iter
sham_iter[k]=shd(estim_iter_g,true_g)
ham_iter[k]=hamming(estim_iter_g,true_g)

amat(estim_bez_g)=estim_bez
sham_bez[k]=shd(estim_bez_g,true_g)
ham_bez[k]=hamming(estim_bez_g,true_g)

print(k)
}


power_1
power_iter
power_bez

fdr_1
fdr_iter
fdr_bez

sham_1
sham_iter
sham_bez

ham_1
ham_iter
ham_bez

mean(power_1)
mean(power_iter)
mean(power_bez)

mean(fdr_1)
mean(fdr_iter)
mean(fdr_bez)

mean(sham_1)
mean(sham_iter)
mean(sham_bez)

mean(ham_1)
mean(ham_iter)
mean(ham_bez)

sd(power_1)
sd(power_iter)
sd(power_bez)

sd(fdr_1)
sd(fdr_iter)
sd(fdr_bez)

sd(sham_1)
sd(sham_iter)
sd(sham_bez)

sd(ham_1)
sd(ham_iter)
sd(ham_bez)

