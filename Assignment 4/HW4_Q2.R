#install.packages('ROCR')
library(glmnet)
library(ROCR)

rm(list = ls())

#Read Table
df <- read.table(
  "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data",
  sep=","
);

attach(df)

# Remove column V2 and scale all the independent variables
drops <- c("V2")
df <- df[,!(names(df) %in% drops)]
df$V35 <- ifelse(df$V35 == 'g',1,0)

df.dat <- scale(df[,!(names(df) %in% c("V35"))])
df.dat<-as.data.frame(df.dat)
df.dat$V35 <- df[,c("V35")]

# Split the given dataset into training and test set
set.seed(123)
df.dat <- df.dat[sample(nrow(df.dat)), ]
bound <- floor(nrow(df.dat)*0.7)
df.dat.train <- df.dat[1:bound, ]
df.dat.test <- df.dat[(bound+1):nrow(df.dat), ]

df.dat.train.x<-df.dat.train[,!(names(df.dat.train) %in% c("V35"))]
df.dat.train.y<-df.dat.train$V35
df.dat.test.x<-df.dat.test[,!(names(df.dat.test) %in% c("V35"))]
df.dat.test.y<-df.dat.test$V35

df.dat.train.x<-as.matrix(df.dat.train.x)
df.dat.test.x<-as.matrix(df.dat.test.x)

# 2(a)
###############################################
##### Running a ridge logistic regression #####
###############################################

cv.out =cv.glmnet(df.dat.train.x,df.dat.train.y,alpha =0,nfolds=10,family="binomial")
par(mfrow=c(1,1))
plot(cv.out)
bestlam =cv.out$lambda.min

#Model probability and prediction
#Train
ridge.prob.train=predict(cv.out,s=bestlam,newx=df.dat.train.x,type="response")
ridge.pred.train=rep(0 ,245)
ridge.pred.train[ridge.prob.train>.5]=1
train.error.rate <- mean((ridge.pred.train==0 & df.dat.train.y==1) | 
                           (ridge.pred.train==1 & df.dat.train.y==0))
train.error.rate
#Test
ridge.prob.test=predict(cv.out,s=bestlam,newx=df.dat.test.x,type="response")
ridge.pred.test=rep(0,106)
ridge.pred.test[ridge.prob.test>.5]=1
test.error.rate <- mean((ridge.pred.test==0 & df.dat.test.y==1) | 
                          (ridge.pred.test==1 & df.dat.test.y==0))
test.error.rate

# 2(b)
##################################################
##### Plotting ROC curve nad calculating AUC #####
##################################################
pred <- prediction(ridge.prob.test,df.dat.test.y)

perf <- performance(pred,"tpr","fpr")
plot(perf, main="ROC curve", colorize=T)

auc.tmp <- performance(pred,"auc")
auc <- as.numeric(auc.tmp@y.values)
auc

# 2(c)
###############################
##### Plotting lift curve #####
###############################
lift<- performance(pred, "lift", "rpp")
plot(lift, main="Lift curve", colorize=T)
