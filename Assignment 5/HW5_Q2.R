setwd("C:/Users/Neerav Basant/Desktop/Fall Semester/Advanced Predictive Modeling/Assignments/Assignment 5/HW5_files")

load("imbalanced.RData")

library("caret")
library("ROCR")
set.seed(10)


##### 2 (a)  ########
log.fit = glm(income ~ ., family = binomial(), data = trdat)
log.pred = predict(log.fit, newdata = tedat, type = "response")

pred.test=rep(0,4000)
pred.test[log.pred > .5]=1
table(pred.test, tedat$income)

pred <- prediction(log.pred,tedat$income)  
auc.tmp <- performance(pred,"auc")
auc <- as.numeric(auc.tmp@y.values)
auc 

# Sampling
up_trdat = upSample(x = trdat[,-(length(trdat))], y = trdat$income, yname = "income")
down_trdat = downSample(x = trdat[,-(length(trdat))], y = trdat$income, yname = "income")

table(trdat$income)
table(up_trdat$income)
table(down_trdat$income)

up.log.fit = glm(income ~ ., family = binomial(), data = up_trdat)
up.log.pred = predict(up.log.fit, newdata = tedat, type = "response")

up.pred.test=rep(0,4000)
up.pred.test[up.log.pred > .5]=1
table(up.pred.test, tedat$income)

pred <- prediction(up.log.pred,tedat$income)  
auc.tmp <- performance(pred,"auc")
auc <- as.numeric(auc.tmp@y.values)
auc

down.log.fit = glm(income ~ ., family = binomial(), data = down_trdat)
down.log.pred = predict(down.log.fit, newdata = tedat, type = "response")

down.pred.test=rep(0,4000)
down.pred.test[down.log.pred > .5]=1
table(down.pred.test, tedat$income)

pred <- prediction(down.log.pred,tedat$income)  
auc.tmp <- performance(pred,"auc")
auc <- as.numeric(auc.tmp@y.values)
auc





#########     2(b)    #########
library(rpart)
library(rpart.plot)

reg.fit <- rpart(income ~ ., data=trdat, method = "class", control = rpart.control(xval=5,cp=0.0001))
plotcp(reg.fit)
printcp(reg.fit)
reg.prune <- prune(reg.fit,cp=0.00081037)
reg.pred = predict(reg.prune, newdata= tedat)

reg.pred.test=rep(0,4000)
reg.pred.test[reg.pred[,2] > .5]=1
table(reg.pred.test, tedat$income)

pred <- prediction(reg.pred[,2], tedat$income)  
auc.tmp <- performance(pred,"auc")
auc <- as.numeric(auc.tmp@y.values)
auc

reg.fit.loss <- rpart(income ~ ., data=trdat, method = "class", parms=list(split="information", loss=matrix(c(0,1,2,0), byrow=TRUE, nrow=2)), control = rpart.control(xval=5,cp=0.0001))
plotcp(reg.fit.loss)
printcp(reg.fit.loss)
reg.prune.loss <- prune(reg.fit.loss,cp=0.00035454)
reg.pred.loss = predict(reg.prune.loss, newdata= tedat)

reg.pred.loss.test=rep(0,4000)
reg.pred.loss.test[reg.pred.loss[,2] > .5]=1
table(reg.pred.loss.test, tedat$income)

pred <- prediction(reg.pred.loss[,2], tedat$income)  
auc.tmp <- performance(pred,"auc")
auc <- as.numeric(auc.tmp@y.values)
auc


#########     2(c)    #########
#install.packages("kernlab")
library(kernlab)

svm.fit <- ksvm(income ~ ., data = trdat, kernel = "vanilladot", prob.model = TRUE)
svm.pred <- predict(svm.fit, newdata = tedat, type = "r")
 
pred <- prediction(as.numeric(svm.pred),tedat$income)
auc.tmp <- performance(pred,"auc")
auc <- as.numeric(auc.tmp@y.values)
auc

svm.fit.loss <- ksvm(income ~ ., data = trdat, kernel = "vanilladot", class.weights=c("small"=1,"large"=2), prob.model = TRUE)
svm.pred.loss <- predict(svm.fit.loss, newdata = tedat, type = "r")

pred <- prediction(as.numeric(svm.pred.loss),tedat$income)
auc.tmp <- performance(pred,"auc")
auc <- as.numeric(auc.tmp@y.values)
auc
