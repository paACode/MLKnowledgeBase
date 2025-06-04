
#### Packages ####
library(tidyverse)    # data manipulation and visualization
library(kernlab)      # SVM methodology
library(e1071)        # SVM methodology
library(caret)

#### Helper Functions####

plot_svm_results <- function(model, data, xvar, yvar, labelvar) {
  # Predict on the data
  data$predicted <- predict(model, data)
  
  # Identify misclassified points
  data$misclassified <- data[[labelvar]] != data$predicted
  
  # Get support vectors
  sv <- data[model$index, ]
  
  # Get weights and intercept from model
  w <- t(model$coefs) %*% model$SV
  b <- -model$rho
  
  # Create decision boundary and margins
  slope <- -w[1]/w[2]
  intercept <- -b/w[2]
  margin <- 1 / sqrt(sum(w^2))
  
  # Create plotting grid
  ggplot(data, aes_string(x = xvar, y = yvar)) +
    # Points, color by true label
    geom_point(aes_string(color = labelvar), size = 2) +
    
    # Misclassified points
    geom_point(data = filter(data, misclassified), shape = 4, size = 3, color = "black", stroke = 1.5) +
    
    # Support vectors
    geom_point(data = sv, shape = 1, size = 4, color = "black", stroke = 1.5) +
    
    # Decision boundary
    geom_abline(slope = slope, intercept = intercept, color = "blue", linetype = "solid") +
    
    # Margins
    geom_abline(slope = slope, intercept = intercept +margin, color = "blue", linetype = "dashed") +
    geom_abline(slope = slope, intercept = intercept - margin, color = "blue", linetype = "dashed") +
    
    scale_color_manual(values = c("-1" = "magenta", "1" = "cyan")) +
    
    labs(title = "SVM: Decision Boundary, Margins, Support Vectors & Misclassifications") +
    theme_minimal()
}



#### Maximal Margin Classifier without outlier ####

# Create a Dataset
d.iris.mmc <- iris %>% 
  filter(Species %in%  c("setosa", "versicolor")) %>% 
  mutate(label = factor(ifelse(Species == "setosa", -1, 1))) %>%
  select(Petal.Length, Petal.Width, label)%>%
  mutate(
    Petal.Length = if_else(label == -1, Petal.Length + 1, Petal.Length),
    Petal.Width  = if_else(label == -1, Petal.Width + 0, Petal.Width)
  )


ggplot(data = d.iris.mmc, aes(x = Petal.Length, y = Petal.Width, color = label))+ 
  geom_point()

# Fit Support Vector Machine model to data set
svmfit <- svm(label ~ Petal.Length + Petal.Width, 
              data = d.iris.mmc, kernel = "linear", scale = FALSE, cost=1e5)

plot_svm_results(svmfit, d.iris.mmc, 
                 xvar = "Petal.Length", 
                 yvar = "Petal.Width", 
                 labelvar = "label")
# Plot with e1071 function
# plot(svmfit, d.iris.mmc,  formula = Petal.Width ~ Petal.Length)


#### Maximal Margin Classifier with outlier ####


outlier <- data.frame(Petal.Length = 3, Petal.Width = 1, label = factor(-1, levels = c(1,-1)))
d.iris.mmc <- rbind(d.iris.mmc, outlier)

ggplot(data = d.iris.mmc, aes(x = Petal.Length, y = Petal.Width, color = label))+ 
  geom_point()

# Fit Support Vector Machine model to data set
svmfit <- svm(label ~ Petal.Length + Petal.Width, 
              data = d.iris.mmc, kernel = "linear", scale = FALSE, cost=10)

plot_svm_results(svmfit, d.iris.mmc, 
                 xvar = "Petal.Length", 
                 yvar = "Petal.Width", 
                 labelvar = "label")
# Plot with e1071 function
plot(svmfit, d.iris.mmc,  formula = Petal.Width ~ Petal.Length)




#### Example SVM IRIS ####

#EDA
iris %>% ggplot(aes(x = Sepal.Length, y = Sepal.Width, color = Species)) + geom_point()

iris %>% ggplot(aes(x = Petal.Length, y = Petal.Width, color = Species)) + geom_point()

set.seed(123) 
indices <- createDataPartition(iris$Species, p=.85, list = F)

train <- iris %>%
  slice(indices) # 129 of 150 are train data
test_in <- iris %>%
  slice(-indices) %>% select(-Species) # 21 are test
test_truth <- iris %>%
  slice(-indices) %>%
  pull(Species) # these are the results of test data

# With Linear-Kernel
set.seed(123)
iris_svm_lin <- svm(Species ~ ., train, 
                kernel = "linear", scale = TRUE, cost = 10)

plot(iris_svm, train, Petal.Length ~ Petal.Width)
#Need to check in other 2D Plot where the 2 species are distinguishable --> insert coordinates to slice
plot(iris_svm, train, Petal.Length ~ Petal.Width, slice = list(Sepal.Length = 6, Sepal.Width = 3))


#With Radial-Kernel
set.seed(123)
iris_svm_radial <- svm(Species ~ ., train, 
                kernel = "radial", scale = TRUE, cost = 10)

plot(iris_svm, train, Petal.Length ~ Petal.Width, slice = list(Sepal.Length = 5.5, Sepal.Width = 3.25))

# Test Linear Model

test_pred <- predict(iris_svm_lin, test_in) 
table(test_pred)

conf_matrix <- confusionMatrix(test_pred, test_truth) 
conf_matrix

# Test Radial Model
test_pred <- predict(iris_svm_radial, test_in) 
table(test_pred)

conf_matrix <- confusionMatrix(test_pred, test_truth) 
conf_matrix



####Example Hand Writing####

#Structure : Each row contains 64 Pixels V2 to V65 --> Representing a Grayscale img
# V65 is the label

digits_train <- read.csv("data/optdigits.tra", header = FALSE) 
# We drop V1 because it only contains of 0
X = digits_train[,-1]
# V65 contains the Labels
Y =  as.factor(digits_train[,65])
digits_train <- data.frame(x = X, y = Y)

m = matrix(unlist(digits_train[1,-1]),8,8)
image(m, axes = FALSE, col = grey(seq(0, 1, length = 256)))
      
digits_train[1,65]


digits_svm <- svm(y~., data = digits_train, kernel = "linear", cost = 10, scale = FALSE)
table(digits_svm$fitted, digits_train$y)


#Now we test 

# Load test set
digits_test <- read.csv("data/optdigits.tes", header = FALSE)

# Drop V1  because all
X_test = digits_test[,-1]
Y_test = as.factor(digits_test[,65])

digits_test <- data.frame(x = X_test, y = Y_test)

# Predict on test set
pred <- predict(digits_svm, digits_test)
table(pred, digits_test$y)


#### Example Support Vector Regression ####
library(MASS)
set.seed(123) 
indexes <- createDataPartition(Boston$medv, p = .9, list = F)
train <- Boston[indexes, ]
test <- Boston[-indexes, ]
model_reg <- svm(medv~., data=train, scale = TRUE, epsilon =0.1) 
print(model_reg)

pred <-  predict(model_reg, test)
x <- 1:length(test$medv)
plot(x, test$medv, pch=18, col="red") 
lines(x, pred, lwd="1", col="blue")

sqrt(mean((test$medv - pred)^2))
