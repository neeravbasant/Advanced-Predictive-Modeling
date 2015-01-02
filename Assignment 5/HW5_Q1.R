setwd("C:/Users/Neerav Basant/Desktop/Fall Semester/Advanced Predictive Modeling/Assignments/Assignment 5/HW5_files")

#install.packages("monmlp")
library("monmlp")

train <- read.csv("pendigits.tra")
test <- read.csv("pendigits.tes")

train.X = as.matrix(train[,-(length(train))])
train.Y = as.matrix(train[,(length(train))])
test.X = as.matrix(test[,-(length(test))])
test.Y = as.matrix(test[,(length(test))])

#function for running model
model <- function(nodes, epochs){
  set.seed(123)
  mlp.fit <- monmlp.fit(x = train.X, y = train.Y, hidden1 = nodes, iter.max = epochs, Th = logistic)
  mlp.pred <- round(monmlp.predict(x = test.X, weights = mlp.fit))
  mlp.pred[mlp.pred < 0] = 0
  mlp.pred[mlp.pred > 9] = 9
  return(mlp.pred)
}

#function for getting accuracy
accuracy <- function(pred, test){
  
  tab = table(pred, test)
  
  acc = 0
    for (i in 1:ncol(tab)){
    acc = acc + tab[i,i]
  }
  
  return(acc/sum(tab))
}

#MAking prediction for all the combination of nodes and epochs
pred_5_500 = model(5, 500)
table(pred_5_500, test.Y)
accuracy(pred_5_500, test.Y)

pred_5_1000 = model(5, 1000)
table(pred_5_1000, test.Y)
accuracy(pred_5_1000, test.Y)

pred_5_2000 = model(5, 2000)
table(pred_5_2000, test.Y)
accuracy(pred_5_2000, test.Y)

pred_10_100 = model(10, 100)
table(pred_10_100, test.Y)
accuracy(pred_10_100, test.Y)

pred_10_1000 = model(10, 1000)
table(pred_10_1000, test.Y)
accuracy(pred_10_1000, test.Y)

pred_10_2000 = model(10, 2000)
table(pred_10_2000, test.Y)
accuracy(pred_10_2000, test.Y)

pred_15_100 = model(15, 100)
table(pred_15_100, test.Y)
accuracy(pred_15_100, test.Y)

pred_15_1000 = model(15, 1000)
table(pred_15_1000, test.Y)
accuracy(pred_15_1000, test.Y)

pred_15_2000 = model(15, 2000)
table(pred_15_2000, test.Y)
accuracy(pred_15_2000, test.Y)

