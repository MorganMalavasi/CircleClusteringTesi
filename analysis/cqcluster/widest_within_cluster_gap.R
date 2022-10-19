if(!require('fpc')) {
    install.packages('fpc')
    library('fpc')
}
###########################################################################################

path_to_dir = getwd()
path_to_file_samples = paste(path_to_dir , "/analysis/cqcluster/k_means_input.csv", sep = "")
samples <- read.csv(path_to_file_samples)

path_to_file_labels = paste(path_to_dir , "/analysis/cqcluster/labels_input.csv", sep = "")
labels <- read.csv(path_to_file_labels)

d <- dist(samples) 
clustering <- labels$X0
alt.clustering = NULL
noisecluster = FALSE
silhouette = TRUE 
G2 = FALSE
G3 = FALSE 
wgap = TRUE 
sepindex = TRUE 
sepprob = 0.1 
sepwithnoise = TRUE 
compareonly = FALSE 
aggregateonly = FALSE
              
averagegap=FALSE 
pamcrit=TRUE

dquantile=0.1
nndist=TRUE
nnk=2 
standardisation="max" 
sepall=TRUE 
maxk=10
cvstan=sqrt(length(clustering))

lweight <- function(x,md)
    (x<md)*(-x/md+1)

## preparations    
if (!is.null(d)) 
    d <- as.dist(d)
cn <- max(clustering)
clusteringf <- as.factor(clustering)
clusteringl <- levels(clusteringf)
cnn <- length(clusteringl)
if (cn != cnn) {
    warning("clustering renumbered because maximum != number of clusters")
    for (i in 1:cnn) clustering[clusteringf == clusteringl[i]] <- i
    cn <- cnn
}
n <- length(clustering)
noisen <- 0
cwn <- cn
if (noisecluster) {
    noisen <- sum(clustering == cn)
    cwn <- cn - 1
}
parsimony <- cn/maxk
# cn: number of clusters including noise; cwn: number of clusters w/o noise
diameter <- average.distance <- median.distance <- separation <- average.toother <- cluster.size <- within.dist <- between.dist <- numeric(0)
for (i in 1:cn) cluster.size[i] <- sum(clustering == i)
## standardisation    
if (is.numeric(standardisation)) {
    stan <- standardisation
} else { 
    stan <- switch(standardisation, max=max(d), ave=mean(d), q90=quantile(d,0.9), 1)
}



dmat <- as.matrix(d)
within.cluster.ss <- 0
overall.ss <- nonnoise.ss <- sum(d^2)/n
if (noisecluster) 
    nonnoise.ss <- sum(as.dist(dmat[clustering <= cwn, clustering <= cwn])^2)/sum(clustering <= cwn)
ave.between.matrix <- separation.matrix <- matrix(0, ncol = cn, nrow = cn)
nnd <- numeric(0)
cvnndc <- rep(NA,cn)
mnnd <- cvnnd <- NULL
di <- list()
for (i in 1:cn) { 
    cluster.size[i] <- sum(clustering == i)
    di <- as.dist(dmat[clustering == i, clustering == i])
    if (i <= cwn) {
        within.cluster.ss <- within.cluster.ss + sum(di^2)/cluster.size[i]
        within.dist <- c(within.dist, di)
    }
    if (length(di) > 0){ 
        diameter[i] <- max(di)                
        average.distance[i] <- mean(di)
        median.distance[i] <- median(di)
    } 
    else {
        diameter[i] <- average.distance[i] <- median.distance[i] <- NA
    }
    bv <- numeric(0)
    for (j in 1:cn) {
        if (j != i) {
            sij <- dmat[clustering == i, clustering == j]
            bv <- c(bv, sij)
            if (i < j) {
                separation.matrix[i, j] <- separation.matrix[j, i] <- min(sij)
                ave.between.matrix[i, j] <- ave.between.matrix[j, i] <- mean(sij)
                if (i <= cwn & j <= cwn) 
                    between.dist <- c(between.dist, sij)
            } # if i<j
        } # if j!=i
    } # for j
    separation[i] <- min(bv)
    average.toother[i] <- mean(bv)
} # for i



# wide gap
cwidegap <- rep(0, cwn)
for (i in 1:cwn) if (sum(clustering == i) > 1) 
    cwidegap[i] <- max(hclust(as.dist(dmat[clustering == 
        i, clustering == i]), method = "single")$height)
if (averagegap){
    widestgap <- mean(cwidegap)/stan
} else {
    widestgap <- max(cwidegap)/stan 
}
print(widestgap)