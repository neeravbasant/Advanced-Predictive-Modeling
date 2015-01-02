
##################
### Question 1 ###
##################

#################
# Code for ui.R #
#################

library(shiny)

# Client side code for application that draws a histogram
shinyUI(fluidPage(
  
  # Title
  titlePanel("Data Exploration"),
  
  # Input: Sidebar with a slider input for the number of bins
  sidebarLayout(
    sidebarPanel(
      sliderInput("bins",
                  "Number of bins:",
                  min = 6,
                  max = 18,
                  value = 10),
      
      radioButtons("radio", label = h3("Pick the column"),
                   choices = list("Log Medium Income" = 1, "Flood Depth" = 2), selected = 1)
    ),
    
    # Output: Show a plot of the generated distribution
    mainPanel(
      plotOutput("distPlot")
    )
  )
))

#####################
# Code for server.R #
#####################

library(shiny)

# Define server logic required to draw the histogram
shinyServer(function(input, output) {
  
  output$distPlot <- renderPlot({
    df    <- katrina[, as.numeric(input$radio) + 3]
    bins <- seq(min(df), max(df), length.out = input$bins + 1)
    
    hist(df, breaks = bins, col = 'red', border = 'white')
  })
  
})

  
#######################
# Code to run the app #
#######################

setwd("C:/Users/Neerav Basant/Desktop/Fall Semester/Advanced Predictive Modeling/Assignment 2")
katrina = read.csv("C:/Users/Neerav Basant/Desktop/Fall Semester/Advanced Predictive Modeling/Assignment 2/HW2 - Q1/Data/katrina.csv")
runApp('HW2 - Q1', display.mode = "showcase")


##################
### Question 4 ###
##################

install.packages("glmnet")
library(glmnet)
library(stats)
library(MASS)

data(mtcars)
X = data.frame(disp = mtcars$disp, hp = mtcars$hp, wt = mtcars$wt, drat = mtcars$drat)
X2 = X
X2$alsohp = X$hp
Y = mtcars$mpg

#####################
########(a)##########
#####################

#Scale subtracts the mean and divides by standard deviation

x=scale(X)
y=scale(Y)
x2=scale(X2)

#Running regressions for y,x with lamda=1 

lm.fit=lm(y~x)
summary(lm.fit)
ridge.mod=glmnet(x,y,alpha=0,lambda=1)
coef(ridge.mod)
lasso.mod=glmnet(x,y,alpha=1,lambda=1)
coef(lasso.mod)

#Running regressions for y,x2 with lamda=1 

lm.fit1=lm(y~x2)
summary(lm.fit1)
ridge.mod1=glmnet(x2,y,alpha=0,lambda=1)
coef(ridge.mod1)
lasso.mod1=glmnet(x2,y,alpha=1,lambda=1)
coef(lasso.mod1)

#####################
########(b)##########
#####################

# lambda = 0
ridge.mod_0 = glmnet(x,y,alpha=0,lambda=0)
coef(ridge.mod_0)
l2_norm_0 = sqrt(sum(coef(ridge.mod_0)^2) )
l2_norm_0
ridge.pred_0 = predict (ridge.mod_0, s=0, newx=x)
recon_error_0 = sum((ridge.pred_0 -y)^2)
recon_error_0

# lambda = 1
ridge.mod_1 = glmnet(x,y,alpha=0,lambda=1)
coef(ridge.mod_1)
l2_norm_1 = sqrt(sum(coef(ridge.mod_1)^2) )
l2_norm_1
ridge.pred_1 = predict (ridge.mod_1, s=1, newx=x)
recon_error_1= sum((ridge.pred_1 - y)^2)
recon_error_1

# lambda = 10
ridge.mod_10 = glmnet(x,y,alpha=0,lambda=10)
coef(ridge.mod_10)
l2_norm_10 = sqrt(sum(coef(ridge.mod_10)^2) )
l2_norm_10
ridge.pred_10 = predict (ridge.mod_10, s=10, newx=x)
recon_error_10 = sum((ridge.pred_10 - y)^2)
recon_error_10

# lambda = 100
ridge.mod_100 = glmnet(x,y,alpha=0,lambda=100)
coef(ridge.mod_100)
l2_norm_100 = sqrt(sum(coef(ridge.mod_100)^2) )
l2_norm_100
ridge.pred_100 = predict (ridge.mod_100, s=100, newx=x)
recon_error_100 = sum((ridge.pred_100 - y)^2)
recon_error_100

# lambda = 1000
ridge.mod_1000 = glmnet(x,y,alpha=0,lambda=1000)
coef(ridge.mod_1000)
l2_norm_1000 = sqrt(sum(coef(ridge.mod_1000)^2) )
l2_norm_1000
ridge.pred_1000 = predict (ridge.mod_1000, s=1000, newx=x)
recon_error_1000 = sum((ridge.pred_1000 - y)^2)
recon_error_1000
