## load the data
train = read.csv('forestfire-train.csv')
test = read.csv('forestfire-test.csv')
library(ggplot2)

getRMSE <- function(pred, actual) {
  ## TODO: given a vector or predicted values and actual values, calculate RMSE
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

sgd <- function(learn.rate, lambda, train, test, epoch=MAX_EPOCH) {
  ## convert the train and test to matrix format
  train.mat = as.matrix(train) 
  test.mat = as.matrix(test)

  N = nrow(train.mat)
  d = ncol(train.mat)

  ## standardize the columns of both matrices
  for (i in 1:(d-1)){
    ## TODO: standardize the train and test matrices
  
  }

  ## add a feature to represent the intercept
  tmat <- addIntercept(train.mat[, -d])
  testmat <- addIntercept(test.mat[, -d])

  ## initialize all the coefficients to be 0.5
  beta = rep(0.75,d)
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
  return(mse.df)
}
