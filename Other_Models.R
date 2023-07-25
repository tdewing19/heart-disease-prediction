getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

drop_cols_train = numeric(0)
for(i in 1:ncol(h_train)) {
  if (sum(is.na(h_train[ , i])) > 0) {
    drop_cols_train <- c(drop_cols_train, i)
  }
}
h_train <- h_train[, -drop_cols_train]

drop_cols_test = numeric(0)
for(i in 1:ncol(h_test)) {
  if (sum(is.na(h_test[ , i])) > 0) {
    drop_cols_test <- c(drop_cols_test, i)
  }
}
h_test <- h_test[, -drop_cols_test]

h_train$Sex[h_train$Sex == "M"] <- "Male"
h_train$Sex[h_train$Sex == "F"] <- "Female"
h_test$Sex[h_test$Sex == "M"] <- "Male"
h_test$Sex[h_test$Sex == "F"] <- "Female"

# Impute All NA's with 0
h_train[is.na(h_train)] <- 0
h_test[is.na(h_test)] <- 0

for(i in 1:ncol(h_train)) {
  if (class(h_train[ , i]) == 'character') {
    h_train[ , i][is.na(h_train[ , i])] <- getmode(h_train[ , i])
  } else {
    h_train[ , i][is.na(h_train[ , i])] <- median(h_train[ , i], na.rm = TRUE)
  }
}

for(i in 1:ncol(h_test)) {
  if (class(h_test[ , i]) == 'character') {
    h_test[ , i][is.na(h_test[ , i])] <- getmode(h_test[ , i])
  } else {
    h_test[ , i][is.na(h_test[ , i])] <- median(h_test[ , i], na.rm = TRUE)
  }
}

# SVM

library(e1071)

for (i in seq(-2, 2, 0.25)) {
  gamma = 2^i
  h_svm <- svm(as.factor(HeartDisease) ~., data = h_train,
               kernel = "radial", gamma = gamma) # gamma = 1, degree = 3
  train_pred <- predict(h_svm, type = "class")
  print(mean(train_pred == factor(h_train$HeartDisease)))
}

h_svm <- svm(as.factor(HeartDisease) ~ Oldpeak, data = h_train,
             kernel = "linear") # gamma = 1, degree = 3
train_pred <- predict(h_svm, type = "class")
cat("Confusion Matrix and Accuracy of Training Set:\n")
table(train_pred, factor(h_train$HeartDisease))
mean(train_pred == factor(h_train$HeartDisease))

tune.out <- tune(svm,as.factor(HeartDisease)~.,data=h_train,kernel="radial",
                 ranges=list(cost=c(0.1,1,10,100,1000),gamma=c(0.4,1,2,3,4)))

tune.out <- tune(svm, HeartDisease~.,data=h_train,kernel="radial",
                 ranges=list(cost=c(0.1,1,10),gamma=c(0.4,1,2)))

test_pred <- predict(h_svm, h_test, type = "class")
write.csv(data.frame("Ob" = h_test$Ob, "HeartDisease" = test_pred), "svm.csv",
          row.names = FALSE)

# Decision Tree

library(tree)
h_dt <- tree(as.factor(HeartDisease) ~ ., data = h_train)
h_dt_cv <- cv.tree(h_dt, FUN = prune.misclass)
plot(h_dt_cv$size, h_dt_cv$dev)
prune_dt <- prune.tree(h_dt, best = 4)

library(rpart)
prune_dt <- rpart(as.factor(HeartDisease) ~ ., data = h_train)

train_pred <- predict(prune_dt, type = "class")
cat("Confusion Matrix and Accuracy of Training Set:\n")
table(train_pred, factor(h_train$HeartDisease))
mean(train_pred == factor(h_train$HeartDisease))

test_pred <- predict(prune_dt, h_test, type = "class")
write.csv(data.frame("Ob" = h_test$Ob, "HeartDisease" = test_pred), "submission.csv",
          row.names = FALSE)

# Random Forest
library(randomForest)
h_rf <- randomForest(as.factor(HeartDisease) ~ ., data = h_train, mtry = 10,
                     ntree = 200, importance = TRUE)
train_pred <- predict(h_rf, type = "class")
cat("Confusion Matrix and Accuracy of Training Set:\n")
table(train_pred, factor(h_train$HeartDisease))
mean(train_pred == factor(h_train$HeartDisease))

test_pred <- predict(h_rf, h_test, type = "class")
write.csv(data.frame("Ob" = h_test$Ob, "HeartDisease" = test_pred), "submission.csv",
          row.names = FALSE)

# Gradient Boost
library(gbm)

h_train$HeartDisease <- as.numeric(h_train$HeartDisease == "Yes")

h_train$HeartDisease <- c("No", "Yes")[h_train$HeartDisease + 1]


h_gbm <- gbm(HeartDisease ~ ., data = h_train,
             distribution = "bernoulli", n.trees = 500, 
             interaction.depth = 5, cv.folds = 5)
train_pred <- round(predict(h_gbm, type = "response"))
cat("Confusion Matrix and Accuracy of Training Set:\n")
table(train_pred, factor(h_train$HeartDisease))
mean(train_pred == factor(h_train$HeartDisease))

test_pred_ind <- (round(predict(h_gbm, h_test, type = "response")) + 1)
test_pred <- c("No", "Yes")[test_pred_ind]
write.csv(data.frame("Ob" = h_test$Ob, "HeartDisease" = test_pred), "gb2.csv",
          row.names = FALSE)

# XGBoost

library(tidyverse)
library(caret)
library(xgboost)

h_xgb <- train(HeartDisease ~., data = h_train, method = "xgbTree",
               trControl = trainControl("cv", number = 5))
train_pred <- predict(h_xgb)
cat("Confusion Matrix and Accuracy of Training Set:\n")
table(train_pred, factor(h_train$HeartDisease))
mean(train_pred == factor(h_train$HeartDisease))

