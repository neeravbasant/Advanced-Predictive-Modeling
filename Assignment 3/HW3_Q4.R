library(rpart)

df = read.csv ("C:/Users/Neerav Basant/Desktop/Fall Semester/Advanced Predictive Modeling/Assignments/Assignment 3/hw3files/BreastCancer.csv", header =T)

df = as.data.frame(df)
df$id = NULL

out_gini = rpart(formula = diagnosis ~ ., data = df, maxdepth = 2, parms = list(split = "gini"))
out_entropy = rpart(formula = diagnosis ~ ., data = df, maxdepth = 2, parms = list(split = "information"))

par(mfrow = c(1,2), xpd = NA)
plot(out_gini, main = "Decision Tree - Gini")
text(out_gini, use.n = TRUE)

plot(out_entropy, main = "Decision Tree - Entropy")
text(out_entropy, use.n = TRUE)
