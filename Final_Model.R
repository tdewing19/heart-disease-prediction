# Stats 101C Kaggle Project Final Model

# Group Members: Tristan Dewing, Vivian Luk, Karina Santoso, Brandon Wang

# Load in Datasets
h_train <- read.csv("HDtrainNew.csv", header = TRUE)
h_test <- read.csv("HDtestNoYNew.csv", header = TRUE)

# NA Imputation Function
na_impute <- function(data) {
  na_loc <- which(is.na(data))
  non_na_loc <- which(!is.na(data))
  intervals <- findInterval(na_loc, non_na_loc, all.inside = TRUE)
  l_loc <- non_na_loc[pmax(1, intervals)]
  r_loc <- non_na_loc[pmin(length(data), (intervals + 1))]
  l_diff <- na_loc - l_loc
  r_diff <- r_loc - na_loc
  data[na_loc] <- ifelse(l_diff <= r_diff, data[l_loc], data[r_loc])
  return(data)
}

# Imputing NA's
train_colnames <- colnames(h_train)
test_colnames <- colnames(h_test)
for (i in seq_along(train_colnames)) {
  h_train[, train_colnames[i]] <- na_impute(h_train[, train_colnames[i]])
}
for (i in seq_along(test_colnames)) {
  h_test[, test_colnames[i]] <- na_impute(h_test[, test_colnames[i]])
}

# Convert Categorical Variables to Integers
factors <- c("Sex", "ChestPainType", "FastingBS", "RestingECG", "ExerciseAngina",
             "ST_Slope", "hypertension", "stroke", "ever_married", "work_type",
             "Residence_type", "smoking_status")
for (i in seq_along(factors)) {
  h_train[factors[i]] <- as.numeric(as.factor(h_train[,factors[i]]))
  h_train[factors[i]] <- as.numeric(as.factor(h_train[,factors[i]]))
  h_test[factors[i]] <- as.numeric(as.factor(h_test[,factors[i]]))
}

# Logistic Regression
h_lr <- glm(as.factor(HeartDisease) ~ ., data = h_train, family = "binomial")
train_pred_prob <- predict(h_lr, type = "response")
train_pred <- rep("Yes", 4220)
train_pred[train_pred_prob < 0.5] <- "No"
cat("Confusion Matrix and Accuracy of Training Set:\n")
table(train_pred, factor(h_train$HeartDisease))
mean(train_pred == factor(h_train$HeartDisease))

# Create .csv with Test Predictions
test_pred_prob <- predict(h_lr, h_test, type = "response")
test_pred <- rep("Yes", 1808)
test_pred[test_pred_prob < 0.5] <- "No"
write.csv(data.frame("Ob" = h_test$Ob, "HeartDisease" = test_pred), "lr.csv",
          row.names = FALSE)
