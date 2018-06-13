# Jason Rich


set.seed(1234)
setwd('./')
getwd()

source(file = './code/conjugate-gradient-method.R')


# Task 2:
# Refer to the sample program, design your own regression program using
# any computer language you like. Train a linear model using the training data
# set, regression.tra. This data set contains 8 inputs and 7 outputs. Report
# the training error. Test your trained model on the testing data set,
# regression.tst. This data set contains 8 inputs and 7 outputs as well. Report
# the testing error.


################################################################################
 #install packages (if missing)
################################################################################
pkgs <- installed.packages()[,1]
pkgs.need <- c('caTools','tidyverse','MASS', 'Matrix', 'matrixcalc')
pkgs.missing <- pkgs.need[!pkgs.need %in% pkgs]
if (length(pkgs.missing) > 0){
    install.packages(pkgs.missing, dep = TRUE)
}

library(caTools) # Toolkit for data preprocessing in R
library(tidyverse) # Loads tidyverse
library(MASS) # Modern Applied Statistics with S
library(Matrix)
library(matrixcalc)

################################################################################
# data pre-processing
################################################################################


# check for data directory, then check for data files. If the data files exit
# remove the files in the data directory.
if(!file.exists('./data/x.*') || !file.exists('./data/y.*')){
        system('rm ./data/x.* ./data/y.*')
}

# create y training dataframe
system('awk "NR%2==0" ./data/regression.tra > ./data/y.trn', intern = TRUE)

y.trn <- read.table('./data/y.trn', header = FALSE, sep = '')
View(y.trn)
str(y.trn)
length(y.trn) #7
summary(y.trn)




# create y test dataframe
system('awk "NR%2==0" ./data/regression.tst > ./data/y.tst', intern = TRUE)
y.tst <- read.table('./data/y.tst', header = FALSE, sep = '')
View(y.tst)
str(y.tst)
length(y.tst) #7
summary(y.tst)



# convert factor to numeric if required
# y.trn$V1 <- as.numeric(levels(y.trn$V1))[y.trn$V1]
# y.trn$V2 <- as.numeric(levels(y.trn$V2))[y.trn$V2]
# y.trn$V3 <- as.numeric(levels(y.trn$V3))[y.trn$V3]
# y.trn$V4 <- as.numeric(levels(y.trn$V4))[y.trn$V4]
# y.trn$V5 <- as.numeric(levels(y.trn$V5))[y.trn$V5]
# y.trn$V6 <- as.numeric(levels(y.trn$V6))[y.trn$V6]
# y.trn$V7 <- as.numeric(levels(y.trn$V7))[y.trn$V7]



################################################################################
# create the feature matrix
################################################################################

# read the file line by line to create features
df <- readLines('./data/regression.tra')
# will receive an error about incomplete final line
# View(df)

# read the file line by line to create test features
df.tst <- readLines('./data/regression.tst')


# parse out the odd rows
parser <- function(x){
    first <- unlist(strsplit(x[1], '\\s{2,}'))
return(first)
}


# apply the parse function to the dataset
x.trn <- sapply(split(df, ceiling(seq_along(df)/2)),parser)
# View(x.trn)

x.trn.df <- data.frame(x.trn, stringsAsFactors = FALSE)
# View(x.trn.df)

# write dataframe to disk
write.table(x.trn.df, './data/x.trn', quote = FALSE, sep = ' ', row.names =  FALSE, col.names = FALSE)

# read in new x dataframe
x.trn.new  <- read.table('./data/x.trn', header = FALSE, sep = ' ')
View(x.trn.new)
str(x.trn.new)

# test x dataframe
x.tst <- sapply(split(df.tst, ceiling(seq_along(df.tst)/2)),parser)
x.tst.df <- data.frame(x.tst, stringsAsFactors = FALSE)
write.table(x.tst.df, './data/x.tst', quote = FALSE, sep = ' ', row.names =  FALSE, col.names = FALSE)

# read in new x dataframe
x.tst.new  <- read.table('./data/x.tst', header = FALSE, sep = ' ')
View(x.tst.new)
str(x.tst.new)




################################################################################
# Linear Regression
################################################################################

# training feature matrix
x.mat <- as.matrix(x.trn.new)
# View(x.mat)

# test feature matrix
x.mat <- as.matrix(x.tst.new)


# training response matrix
y.mat <- as.matrix(y.trn)
# View(y.mat)

# test response matrix
y.mat <- as.matrix(y.tst)

# intercept vector to add to feature matrix
# int <- rep(1, length(y.mat))
# x.mat <- cbind(int, x.mat)



# Least Square Regression (general Model)

# linear regression hand calculation
fit.cmp <- solve(t(x.mat) %*% x.mat) %*% t(x.mat) %*% y.mat

# RSS hand calculation
resd <- y.mat - (x.mat %*% fit.cmp)
rss <- sum(resd)
# -0.7364869

fit <- lm(y.mat ~ x.mat)
summary(fit)
fit.err <- sum(fit$residuals)

# Calculate the MSE
mean((fit.err - predict(fit))^2)

# The Training error:
# 2.864511

# The Testing error:
# 2.899701


################################################################################
# Two Class Data
################################################################################
# Task 3:
# Modify your linear regression model such that it can do linear classification.
# Run the modified codes on the generated data sets you have in project 1, which
# has two classes. Report the training classification accuracy and testing
# classification accuracy.


# read in the generated training data
trn <-  read.table('./data/train.txt', sep = ' ')
# View(trn)


colnames(trn)<- c('ftr1', 'ftr2', 'rsp')
# View(trn)

# create the feature dataframe
x.trn <- trn[,-3]
# View(x.trn)
# str(x.trn)

# q: are the features the same size?
# a: yes; 400
# length(x.trn[,1]) # 400
# length(x.trn[,2]) # 400

# create the response dataframe
y.trn <- trn[3]
# View(y.trn)
# str(y.trn)

# q: is the response vector the same size as the feature df size?
# a: yes; 400
# length(y.trn[,1])



# read in the generated training data
tst <-  read.table('./data/test.txt', sep = ' ')
# View(tst)


colnames(tst)<- c('ftr1', 'ftr2', 'rsp')
# View(tst)

# create the feature dataframe
x.tst <- tst[,-3]
# View(x.tst)
# str(x.tst)

# q: are the features the same size?
# a: yes; 400
# length(x.tst[,1]) # 400
# length(x.tst[,2]) # 400

# create the response dataframe
y.tst <- tst[3]
# View(y.tst)
# str(y.tst)

# q: is the response vector the same size as the feature df size?
# a: yes; 400
# length(y.tst[,1])



################################################################################
# Linear Classification
################################################################################
# training dataset
# convert x and y to matrix
x.class.mat <- as.matrix(x.trn)
y.class.mat <- as.matrix(y.trn)

fit.2 <- glm(y.class.mat ~ x.class.mat, family = 'binomial')
summary(fit.2)

# Call:  glm(formula = y.class.mat ~ x.class.mat, family = "binomial")

# Coefficients:
#     (Intercept)  x.class.matftr1  x.class.matftr2
#       -2.7714           2.2159           0.2117

# Degrees of Freedom: 399 Total (i.e. Null);  397 Residual
# Null Deviance:	 554.5
# Residual Deviance: 260.3 	AIC: 266.3

# create the parameters for the CGM

A <- (t(x.class.mat) %*% x.class.mat) + diag(x=1, nrow=ncol(x.class.mat))
b <- t(x.class.mat) %*% y.class.mat
x0 <- rep(0, ncol(x.class.mat))

ConjugateGradient(A, b, x0, iter_num = 100, sig_thres = 1e-5)
# The conjugate gradient converged with 2 iterations
# The training accuracy error for frt1 and frt2
# rsp
# ftr1 0.3201239
# ftr2 0.0284924


# testing data
# convert the x and y matrix
x.class.mat <- as.matrix(x.tst)
y.class.mat <- as.matrix(y.tst)


fit.2 <- glm(y.class.mat ~ x.class.mat, family = 'binomial')
summary(fit.2)

# Call:  glm(formula = y.class.mat ~ x.class.mat, family = "binomial")

#Coefficients:
#    (Intercept)  x.class.matftr1  x.class.matftr2
#       -2.7345           2.2531           0.2927

# Degrees of Freedom: 399 Total (i.e. Null);  397 Residual
# Null Deviance:	 554.5
# Residual Deviance: 245.2 	AIC: 251.2

# create the parameters for the CGM

A <- (t(x.class.mat) %*% x.class.mat) + diag(x=1, nrow=ncol(x.class.mat))
b <- t(x.class.mat) %*% y.class.mat
x0 <- rep(0, ncol(x.class.mat))

ConjugateGradient(A, b, x0, iter_num = 100, sig_thres = 1e-5)
#
# The conjugate gradient converged with 2 iterations
# The training accuracy error for frt1 and frt2
# rsp
# ftr1 0.32231548
# ftr2 0.03589415



################################################################################
# Zip Code Data
################################################################################
# Task 4: Run the classification codes on the zip code data set you have in
# project 1. If you have difficulty to inverse the autocorrelation matrix, you
# can address this issue by utilizing the regularization technique. Repeat this
# task six times by using six different regularization coefficients of 0.01, 0.1,
# 0.5, 1, 5, and 10. Report the testing classification accuracies.




################################################################################
# Zip code training data
################################################################################
# ?read.csv
zip.trn <- read.csv('./data/Valid_ZC_train_Data.csv', header = FALSE, sep = ',')
# View(zip.trn)

colnames(zip.trn) <- c('ftr1','ftr2','ftr3','ftr4','ftr5','ftr6','ftr7','ftr8','ftr9'
                       ,'ftr10','ftr11','ftr12','ftr13','ftr14','ftr15','ftr16','rsp')


zip.x.trn <- zip.trn[, -17]
zip.y.trn <- zip.trn[17]

zip.x.mat <- as.matrix(zip.x.trn)
zip.y.mat <- as.matrix(zip.y.trn)


fit.3 <- glm(zip.y.mat ~ zip.x.mat, family = 'gaussian')
fit.3

# Call:  glm(formula = zip.y.mat ~ zip.x.mat, family = "gaussian")

# Coefficients:
#     (Intercept)   zip.x.matftr1   zip.x.matftr2   zip.x.matftr3   zip.x.matftr4   zip.x.matftr5   zip.x.matftr6   zip.x.matftr7   zip.x.matftr8
# 5.600167       -2.809430       -0.427344        0.204414        2.142529       -4.023995       -0.483247        3.196823        1.465467
# zip.x.matftr9  zip.x.matftr10  zip.x.matftr11  zip.x.matftr12  zip.x.matftr13  zip.x.matftr14  zip.x.matftr15  zip.x.matftr16
# -0.039510        0.112512       -0.083371       -0.129410       -0.008066        1.531432        0.047230       -0.042129

# Degrees of Freedom: 2999 Total (i.e. Null);  2983 Residual
# Null Deviance:	    24750
# Residual Deviance: 8908 	AIC: 11810


# regularization coefficients
weights <- c(0.01, 0.1, 0.5, 1, 5, 10)

A.1 <- (t(zip.x.mat) %*% zip.x.mat) + weights[[1]] *diag(x=1, nrow=ncol(zip.x.mat))
A.2 <- (t(zip.x.mat) %*% zip.x.mat) + weights[[2]] *diag(x=1, nrow=ncol(zip.x.mat))
A.3 <- (t(zip.x.mat) %*% zip.x.mat) + weights[[3]] *diag(x=1, nrow=ncol(zip.x.mat))
A.4 <- (t(zip.x.mat) %*% zip.x.mat) + weights[[4]] *diag(x=1, nrow=ncol(zip.x.mat))
A.5 <- (t(zip.x.mat) %*% zip.x.mat) + weights[[5]] *diag(x=1, nrow=ncol(zip.x.mat))
A.6 <- (t(zip.x.mat) %*% zip.x.mat) + weights[[6]] *diag(x=1, nrow=ncol(zip.x.mat))
b <- t(zip.x.mat) %*% zip.y.mat
x0 <- rep(0, ncol(zip.x.mat))

ConjugateGradient(A.1, b, x0, iter_num = 100, sig_thres = 1e-5)
ConjugateGradient(A.2, b, x0, iter_num = 100, sig_thres = 1e-5)
ConjugateGradient(A.3, b, x0, iter_num = 100, sig_thres = 1e-5)
ConjugateGradient(A.4, b, x0, iter_num = 100, sig_thres = 1e-5)
ConjugateGradient(A.5, b, x0, iter_num = 100, sig_thres = 1e-5)
ConjugateGradient(A.6, b, x0, iter_num = 100, sig_thres = 1e-5)



################################################################################
# Zip code test data
################################################################################
zip.tst <- read.csv('./data/Valid_ZC_test_Data.csv', header = FALSE, sep = ',')
# View(zip.tst)

colnames(zip.tst) <- c('ftr1','ftr2','ftr3','ftr4','ftr5','ftr6','ftr7','ftr8','ftr9'
                       ,'ftr10','ftr11','ftr12','ftr13','ftr14','ftr15','ftr16','rsp')


zip.x.tst <- zip.tst[, -17]
zip.y.tst <- zip.tst[17]


zip.x.mat <- as.matrix(zip.x.trn)
zip.y.mat <- as.matrix(zip.y.trn)


fit.3 <- glm(zip.y.mat ~ zip.x.mat, family = 'gaussian')
fit.3
# Call:  glm(formula = zip.y.mat ~ zip.x.mat, family = "gaussian")

# Coefficients:
#     (Intercept)   zip.x.matftr1   zip.x.matftr2   zip.x.matftr3   zip.x.matftr4   zip.x.matftr5   zip.x.matftr6   zip.x.matftr7   zip.x.matftr8
# 5.600167       -2.809430       -0.427344        0.204414        2.142529       -4.023995       -0.483247        3.196823        1.465467
# zip.x.matftr9  zip.x.matftr10  zip.x.matftr11  zip.x.matftr12  zip.x.matftr13  zip.x.matftr14  zip.x.matftr15  zip.x.matftr16
# -0.039510        0.112512       -0.083371       -0.129410       -0.008066        1.531432        0.047230       -0.042129

# Degrees of Freedom: 2999 Total (i.e. Null);  2983 Residual
# Null Deviance:	 24750
# Residual Deviance: 8908 	AIC: 11810

# MSE calculation
mean((fit.3$residuals - predict(fit.3))^2)
# 38.5



# regularization coefficients
weights <- c(0.01, 0.1, 0.5, 1, 5, 10)

A.1 <- (t(zip.x.mat) %*% zip.x.mat) + weights[[1]] *diag(x=1, nrow=ncol(zip.x.mat))
A.2 <- (t(zip.x.mat) %*% zip.x.mat) + weights[[2]] *diag(x=1, nrow=ncol(zip.x.mat))
A.3 <- (t(zip.x.mat) %*% zip.x.mat) + weights[[3]] *diag(x=1, nrow=ncol(zip.x.mat))
A.4 <- (t(zip.x.mat) %*% zip.x.mat) + weights[[4]] *diag(x=1, nrow=ncol(zip.x.mat))
A.5 <- (t(zip.x.mat) %*% zip.x.mat) + weights[[5]] *diag(x=1, nrow=ncol(zip.x.mat))
A.6 <- (t(zip.x.mat) %*% zip.x.mat) + weights[[6]] *diag(x=1, nrow=ncol(zip.x.mat))
b <- t(zip.x.mat) %*% zip.y.mat
x0 <- rep(0, ncol(zip.x.mat))

conj.1 <- ConjugateGradient(A.1, b, x0, iter_num = 100, sig_thres = 1e-5) # the iteration number is 31!
conj.2 <- ConjugateGradient(A.2, b, x0, iter_num = 100, sig_thres = 1e-5) # the iteration number is 35!
conj.3 <- ConjugateGradient(A.3, b, x0, iter_num = 100, sig_thres = 1e-5) # the iteration number is 31!
conj.4 <- ConjugateGradient(A.4, b, x0, iter_num = 100, sig_thres = 1e-5) # the iteration number is 36!
conj.5 <- ConjugateGradient(A.5, b, x0, iter_num = 100, sig_thres = 1e-5) # the iteration number is 30!
conj.6 <- ConjugateGradient(A.6, b, x0, iter_num = 100, sig_thres = 1e-5) # the iteration number is 30!

mean(conj.1) # the average classification accuracy: 0.3286414
mean(conj.2) # the average classification accuracy: 0.328801
mean(conj.3) # the average classification accuracy: 0.3286542
mean(conj.4) # the average classification accuracy: 0.3288661
mean(conj.5) # the average classification accuracy: 0.3307102
mean(conj.6) # the average classification accuracy: 0.3327098





