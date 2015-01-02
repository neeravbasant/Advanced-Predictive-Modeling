library(rpart)
library(rpart.plot)

rm(list = ls())

setwd("C:/Users/Neerav Basant/Desktop/Fall Semester/Advanced Predictive Modeling/Assignments/Assignment 4/hw4files/")

train <- read.table("data.train",sep=',',header=TRUE)
test <- read.table("data.test",sep=',',header=TRUE)

attach(train)

# 3(a)
reg_model <- rpart(mpg ~ ., data=train, method="anova",control = rpart.control(xval=5,cp=0.0001))
rpart.plot(reg_model)
text(reg_model, use.n = TRUE)
title("Decision tree with 5 fold cross validation")


# 3(b)
plotcp(reg_model)
printcp(reg_model)
reg_prune <- prune(reg_model,cp=0.01170368)
rpart.plot(reg_prune)
text(reg_prune, use.n = TRUE)

# 3(c)
prp(reg_prune, main="Regression Tree Plot - Pruned",
    nn=TRUE,             # display the node numbers
    fallen.leaves=TRUE,  # put the leaves on the bottom of the page
    branch=.5,           # change angle of branch lines
    faclen=0,            # do not abbreviate factor levels
    trace=1,             # print the automatically calculated cex
    shadow.col="gray",   # shadows under the leaves
    branch.lty=3,        # draw branches using dotted lines
    split.cex=1.2,       # make the split text larger than the node text
    split.prefix="is ",  # put "is " before split text
    split.suffix="?",    # put "?" after split text
    split.box.col="lightgray",   # lightgray split boxes (default is white)
    split.border.col="darkgray", # darkgray border on split boxes
    split.round=.5,
    under.font=15)

# 3(d)

pred.train=predict(reg_prune,newdata= train)
plot(pred.train,train$mpg)
abline (0,1)
title("Actual MPG vs Predicted MPG - Train")
mse.train = mean((pred.train - train$mpg)^2)
mse.train

pred.test=predict(reg_prune,newdata= test)
plot(pred.test,test$mpg)
abline (0,1)
title("Actual MPG vs Predicted MPG - Test")
mse.test = mean((pred.test - test$mpg)^2)
mse.test


# 3(e)

reg_model.poisson <- rpart(mpg ~ ., data=train, method="poisson",control = rpart.control(xval=5,cp=0.0001))
rpart.plot(reg_model.poisson)
text(reg_model.poisson, use.n = TRUE)
title("Decision tree with 5 fold cross validation - Poisson")

plotcp(reg_model.poisson)
printcp(reg_model.poisson)
reg_prune.poisson <- prune(reg_model.poisson,cp=0.00872413)
rpart.plot(reg_prune.poisson)
text(reg_prune.poisson, use.n = TRUE)

pred.train.p=predict(reg_prune.poisson,newdata= train)
plot(pred.train.p,train$mpg)
abline (0,1)
title("Actual MPG vs Predicted MPG - Train (Poisson)")
mse.train.p = mean((pred.train.p - train$mpg)^2)
mse.train.p

pred.test.p=predict(reg_prune.poisson,newdata= test)
plot(pred.test.p,test$mpg)
abline (0,1)
title("Actual MPG vs Predicted MPG - Test (Poisson)")
mse.test.p = mean((pred.test.p - test$mpg)^2)
mse.test.p
