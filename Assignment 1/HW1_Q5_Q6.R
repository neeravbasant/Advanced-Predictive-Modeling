
######################
###   Question 5   ###
######################

library(MASS)
library(RColorBrewer)

## Do this for consistent results
set.seed(10)

## generate the bivariate normal sample for the three cases
bivn1 <- mvrnorm(500000, mu = c(0, 0), Sigma = matrix(c(4, 0, 0, 9), 2))
bivn2 <- mvrnorm(500000, mu = c(0, 0), Sigma = matrix(c(4, 3, 3, 9), 2))

## get the kernel density estimates for the 3 pairs
bivn1.kde <- kde2d(bivn1[,1], bivn1[,2], n = 50)
bivn2.kde <- kde2d(bivn2[,1], bivn2[,2], n = 50)

## contour plots
pdf("bivariate-contour-1.pdf", width=6, height=6)
image(bivn1.kde, col = brewer.pal(8,"OrRd"), ylim=c(-10,10), xlim=c(-10,10))
contour(bivn1.kde, add = T)
dev.off()
pdf("bivariate-contour-2.pdf", width=6, height=6)
image(bivn2.kde, col = brewer.pal(8,"OrRd"), ylim=c(-10,10), xlim=c(-10,10))
contour(bivn2.kde, add = T)
dev.off()

persp(bivn1.kde, phi = 45, theta = 45, shade = .6, border = NA, ticktype="detailed")
persp(bivn2.kde, phi = 45, theta = 45, shade = .6, border = NA, ticktype="detailed")


######################
###   Question 6   ###
######################

library(LearnBayes)
data('studentdata')

# 6a
hist(studentdata$Shoes, breaks = 20)

# 6b
plot(density(na.omit(log(studentdata$Dvds))))

# Q 6c
summary(na.omit(studentdata$Haircut))
quantile(studentdata$Dvds,probs=c(.025,.975),na.rm=TRUE)

# 6d
barplot(table(studentdata$Gender, studentdata$Drink), xlab = "Drink", ylab = "Count")

# 6e
plot(studentdata$ToSleep,studentdata$WakeUp,xlab="ToSleep",ylab="WakeUp")
fit=lm(studentdata$WakeUp~studentdata$ToSleep)
abline(fit)
