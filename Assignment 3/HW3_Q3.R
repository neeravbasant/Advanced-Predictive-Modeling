####################################
############   3 - a   #############
####################################

# Created a function called sgd_ridge that will return the beta values based on the ridge regression


## load the data
train = read.csv('C:/Users/Neerav Basant/Desktop/Fall Semester/Advanced Predictive Modeling/Assignments/Assignment 3/hw3files/forestfire-train.csv')
test = read.csv('C:/Users/Neerav Basant/Desktop/Fall Semester/Advanced Predictive Modeling/Assignments/Assignment 3/hw3files/forestfire-test.csv')
library(ggplot2)

getRMSE <- function(pred, actual) {
  ## TODO: given a vector or predicted values and actual values, calculate RMSE
  return (sqrt(mean((actual - pred) ^ 2)))
}

addIntercept <- function(mat) {
  ## add intercept to the matrix
  allones= rep(1, nrow(mat))
  return(cbind(Intercept=allones, mat))
}

predictSamples <- function(beta, mat) {
  ## TODO: compute the predicted value using matrix multiplication
  ## Note that for a single row of mat, pred = sum_i (beta_i * feature_i)
  return (mat %*% beta)
}

MAX_EPOCH = 100

sgd_ridge <- function(learn.rate, lambda, train, test, epoch=MAX_EPOCH) {
  ## convert the train and test to matrix format
  train.mat = as.matrix(train) 
  test.mat = as.matrix(test)
  
  N = nrow(train.mat)
  d = ncol(train.mat)
  
  ## standardize the columns of both matrices
  for (i in 1:(d-1)){
    ## TODO: standardize the train and test matrices
    train.mat[,i] = scale(train.mat[,i])
    test.mat[,i] = scale(test.mat[,i])
  }
  
  ## add a feature to represent the intercept
  tmat <- addIntercept(train.mat[, -d])
  testmat <- addIntercept(test.mat[, -d])
  y = as.matrix(train.mat[,d])
  
  ## initialize all the coefficients to be 0.5
  beta = rep(0.5,d)
  j = 1
  mse.df <- NULL
  # predict training residuals
  pred_train =predictSamples(beta, tmat)
  pred_test = predictSamples(beta, testmat)
  tMse = getRMSE(pred_train, train$area)
  testMSE = getRMSE(pred_test, test$area)
  mse.df <- rbind(mse.df, data.frame(epoch=j, train=tMse, test=testMSE))
  
  while(j < MAX_EPOCH){  
    j=j+1;
    # for each row in the training data
    for (n in seq(1:N)){
      ##TODO: update beta according to slide #6 in APA-reg2
      gradient <- (t(tmat[n,,drop=FALSE]) %*% ((tmat[n,,drop=FALSE] %*% (beta)) - y[n,drop=FALSE])) + lambda*(beta)
      beta <- (beta) - (learn.rate  * (gradient))
    }
    pred_train = predictSamples(beta, tmat)
    pred_test = predictSamples(beta, testmat)
    tmp_test <- data.frame(pred=pred_test, actual=test$area, type="test")
    tmp_train <- data.frame(pred=pred_train, actual=train$area, type="train")
    tmp <- rbind(tmp_train, tmp_test)
    ggplot(tmp, aes(x=pred, y=actual, color=type)) + theme_bw() + geom_point()
    
    tMse = getRMSE(pred_train, train$area)
    testMSE = getRMSE(pred_test, test$area)
    mse.df <- rbind(mse.df, data.frame(epoch=j, train=tMse, test=testMSE))
  } 
  return(beta)
}

# Comparing glmnet and sgd_ridge

sgd_ridge(0.00055, 0.1, train, test)

X = as.matrix(data.frame(FFMC = train$FFMC, DMC = train$DMC, DC = train$DC, ISI = train$ISI, temp = train$temp, RH = train$RH, wind = train$wind, rain = train$rain))
x=scale(X)
ridge.mod=glmnet(x,train$area,alpha=0,lambda=0.1)
coef(ridge.mod)

####################################
############   3 - b   #############
####################################

# Created sgd function to return the required plot between epoch and error
# Function will also return RMSE value for both training and test dataset for each epoch

sgd <- function(learn.rate, lambda, train, test, epoch=MAX_EPOCH) {
  ## convert the train and test to matrix format
  train.mat = as.matrix(train) 
  test.mat = as.matrix(test)
  
  N = nrow(train.mat)
  d = ncol(train.mat)
  
  ## standardize the columns of both matrices
  for (i in 1:(d-1)){
    ## TODO: standardize the train and test matrices
    train.mat[,i] = scale(train.mat[,i])
    test.mat[,i] = scale(test.mat[,i])
  }
  
  ## add a feature to represent the intercept
  tmat <- addIntercept(train.mat[, -d])
  testmat <- addIntercept(test.mat[, -d])
  y = as.matrix(train.mat[,d])
  
  ## initialize all the coefficients to be 0.5
  beta = rep(0.5,d)
  j = 1
  mse.df <- NULL
  # predict training residuals
  pred_train =predictSamples(beta, tmat)
  pred_test = predictSamples(beta, testmat)
  tMse = getRMSE(pred_train, train$area)
  testMSE = getRMSE(pred_test, test$area)
  mse.df <- rbind(mse.df, data.frame(epoch=j, train=tMse, test=testMSE))
  
  while(j < MAX_EPOCH){  
    j=j+1;
    # for each row in the training data
    for (n in seq(1:N)){
      ##TODO: update beta according to slide #6 in APA-reg2
      gradient <- (t(tmat[n,,drop=FALSE]) %*% ((tmat[n,,drop=FALSE] %*% (beta)) - y[n,drop=FALSE])) + lambda*(beta)
      beta <- (beta) - (learn.rate  * (gradient))
    }
    pred_train = predictSamples(beta, tmat)
    pred_test = predictSamples(beta, testmat)
    tmp_test <- data.frame(pred=pred_test, actual=test$area, type="test")
    tmp_train <- data.frame(pred=pred_train, actual=train$area, type="train")
    tmp <- rbind(tmp_train, tmp_test)
    ggplot(tmp, aes(x=pred, y=actual, color=type)) + theme_bw() + geom_point()
    
    tMse = getRMSE(pred_train, train$area)
    testMSE = getRMSE(pred_test, test$area)
    mse.df <- rbind(mse.df, data.frame(epoch=j, train=tMse, test=testMSE))
    plt = ggplot(mse.df, aes(x=epoch, y=test, color='red')) + theme_bw() + geom_point()
  } 
  return(list(plt, mse.df))
}

sgd(0.000025, 0.1, train, test)
sgd(0.00055, 0.1, train, test)
sgd(0.0075, 0.1, train, test)


####################################
############   3 - c   #############
####################################

sgd(0.000025, 0.01, train, test)
sgd(0.000025, 0.1, train, test)
sgd(0.000025, 1, train, test)

sgd(0.000025, 0, train, test)

X = data.frame(FFMC = train$FFMC, DMC = train$DMC, DC = train$DC, ISI = train$ISI, temp = train$temp, RH = train$RH, wind = train$wind, rain = train$rain)
x = scale(X)
new_train = 
X_test = data.frame(FFMC = test$FFMC, DMC = test$DMC, DC = test$DC, ISI = test$ISI, temp = test$temp, RH = test$RH, wind = test$wind, rain = test$rain)
x_test = scale(X_test)
new_train

lm.mod = lm(train$area ~ ., train)
summary(lm.mod)
lm.pred = predict(lm.mod, test)
lm.rmse = getRMSE(lm.pred, test$area)
lm.rmse
