load("C:/Users/Neerav Basant/Desktop/Fall Semester/Advanced Predictive Modeling/Assignments/Assignment 3/hw3files/mlm.Rdata")
install.packages("lme4")

# 2(a)

###############################################
# Linear regression using numerical variables #
###############################################

lm.fit = lm(Y ~ num1 + num2 + num3, data = train.data)
summary(lm.fit)
lm.pred = predict(lm.fit, test.data)
val.errors_a = mean((test.data$Y - lm.pred) ^ 2)
val.errors_a

# 2(b)

#################################################
# Creating dummy variables for training dataset #
#################################################

n.train = dim(train.data)[1]

unique(train.data$cat1)

train.data$d1.cat1 = rep(0,n.train)
train.data$d1.cat1[train.data$cat1=='a']=1

train.data$d2.cat1 = rep(0,n.train)
train.data$d2.cat1[train.data$cat1=='b']=1

train.data$d3.cat1 = rep(0,n.train)
train.data$d3.cat1[train.data$cat1=='c']=1

unique(train.data$cat2)

train.data$d1.cat2 = rep(0,n.train)
train.data$d1.cat2[train.data$cat2=='I']=1

train.data$d2.cat2 = rep(0,n.train)
train.data$d2.cat2[train.data$cat2=='II']=1

train.data$d3.cat2 = rep(0,n.train)
train.data$d3.cat2[train.data$cat2=='III']=1

train.data$d4.cat2 = rep(0,n.train)
train.data$d4.cat2[train.data$cat2=='IV']=1

train.data$d5.cat2 = rep(0,n.train)
train.data$d5.cat2[train.data$cat2=='V']=1

#############################################
# Creating dummy variables for test dataset #
#############################################

n.test = dim(test.data)[1]

unique(test.data$cat1)

test.data$d1.cat1 = rep(0,n.test)
test.data$d1.cat1[test.data$cat1=='a']=1

test.data$d2.cat1 = rep(0,n.test)
test.data$d2.cat1[test.data$cat1=='b']=1

test.data$d3.cat1 = rep(0,n.test)
test.data$d3.cat1[test.data$cat1=='c']=1

unique(test.data$cat2)

test.data$d1.cat2 = rep(0,n.test)
test.data$d1.cat2[test.data$cat2=='I']=1

test.data$d2.cat2 = rep(0,n.test)
test.data$d2.cat2[test.data$cat2=='II']=1

test.data$d3.cat2 = rep(0,n.test)
test.data$d3.cat2[test.data$cat2=='III']=1

test.data$d4.cat2 = rep(0,n.test)
test.data$d4.cat2[test.data$cat2=='IV']=1

test.data$d5.cat2 = rep(0,n.test)
test.data$d5.cat2[test.data$cat2=='V']=1

############################################################################
# Linear regression using numerical variables and dummy variables for cat1 #
############################################################################

lm.fit.c1 = glm(Y ~ num1 + num2 + num3 + d1.cat1 + d2.cat1 + d3.cat1, data = train.data)
summary(lm.fit.c1)
lm.pred.c1 = predict(lm.fit.c1, test.data)
val.errors_b1 = mean((test.data$Y - lm.pred.c1) ^ 2)
val.errors_b1

############################################################################
# Linear regression using numerical variables and dummy variables for cat2 #
############################################################################

lm.fit.c2 = glm(Y ~ num1 + num2 + num3 + d1.cat2 + d2.cat2 + d3.cat2 + d4.cat2 + d5.cat2, data = train.data)
summary(lm.fit.c2)
lm.pred.c2 = predict(lm.fit.c2, test.data)
val.errors_b2 = mean((test.data$Y - lm.pred.c2) ^ 2)
val.errors_b2

########################################################################################
# Linear regression using numerical variables and dummy variables for both cat1 & cat2 #
########################################################################################

lm.fit.c1c2 = glm(Y ~ num1 + num2 + num3 + d1.cat1 + d2.cat1 + d3.cat1 + d1.cat2 + d2.cat2 + d3.cat2 + d4.cat2 + d5.cat2, data = train.data)
summary(lm.fit.c1c2)
lm.pred.c1c2 = predict(lm.fit.c1c2, test.data)
val.errors_b3 = mean((test.data$Y - lm.pred.c1c2) ^ 2)
val.errors_b3

#####################################################################################
# Linear regression using numerical variables and interaction terms for cat1 & cat2 #
#####################################################################################

lm.fit.c1c2.int = glm(Y ~ num1 + num2 + num3 + cat1:cat2, data = train.data)
summary(lm.fit.c1c2.int)
lm.pred.c1c2.int = predict(lm.fit.c1c2.int, test.data)
val.errors_b4 = mean((test.data$Y - lm.pred.c1c2.int) ^ 2)
val.errors_b4

# 2(c)

library(lme4)

###########################
# Varying intercept model #
###########################

fm.fit <- lmer(Y ~ num1 + num2 + num3 + (1|cat1) + (1|cat2), data = train.data)
summary(fm.fit)
fm.pred = predict(fm.fit, test.data)
val.errors_c1 = mean((test.data$Y - fm.pred) ^ 2)
val.errors_c1

#######################
# Hierarchical models #
#######################

hm.fit <- lmer(Y ~ num1 + num2 + num3 + (1|cat1/cat2), data = train.data)
summary(hm.fit)
hm.pred = predict(hm.fit, test.data)
val.errors_c2 = mean((test.data$Y - hm.pred) ^ 2)
val.errors_c2