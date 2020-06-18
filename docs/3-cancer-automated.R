
## Libraries

library(caret)
library(pROC)
library(ggplot2)
library(gridExtra)


## Functions - 


cross_val_model <- function(train, method, num_cv, folds, comp) {
  set.seed(1)
  cv_model <- train(
    x = train[4:(comp+3)],
    y = train$cancer,
    method = method,
    metric = "Accuracy",
    trControl = trainControl(method="repeatedcv",
                             number = num_cv,
                             repeats = folds),
  )
  return (cv_model)
}

conf_matrix <- function(test, cv_model, comp) {
  real <- as.factor(test$cancer)
  pred <- as.factor(predict(cv_model, test[4:(comp+3)]))
  c_matrix <- confusionMatrix(reference = real, data = pred)
  return (c_matrix)
}

get_metrics <- function(method, pca_cor, num_cat, num_cv, folds, comp) {
  
  test_path <- paste("data/processed/", pca_cor, "_", num_cat, "_test.csv", sep="")
  test <- read.csv(test_path, row.names = 1)
  train_path <- paste("data/processed/", pca_cor, "_", num_cat, "_train.csv", sep="")
  train <- read.csv(train_path, row.names = 1)
  
  cv_model <- cross_val_model(train, method, num_cv, folds, comp)
  c_matrix <- conf_matrix(test, cv_model, comp)
  
  Accuracy <- c_matrix$overall[1]
  Sensitivity <- c_matrix$byClass[1]
  Specificity <- c_matrix$byClass[2]
  F1 <- c_matrix$byClass[7]
  
  return (c(Method=method, Reduction=pca_cor, 
            Type=num_cat, Components=comp,
            Accuracy, Sensitivity, 
            Specificity, F1))
}

loop_models_and_data <- function(reduction_list, type_list, method_list, comp_list, num_cv, folds) {
  summary <- c()
  for (pca_cor in reduction_list) {
    for (num_cat in type_list){
      for (method in method_list){
        for (comp in comp_list){
          metric <- get_metrics(method, pca_cor, num_cat, num_cv, folds, comp)
          summary <- c(summary, metric)
        }
      }
    }
  }
  
  matrix_sum <- matrix(unlist(summary), ncol = 8, byrow = TRUE)
  final_df <- as.data.frame(matrix_sum)
  names(final_df) <- c("Method", "Reduction", "Type", "Component", "Accuracy", "Sensitivity", "Specificity", "F1")
  final_df$Component <- as.factor(final_df$Component)
  write.csv(final_df, file="data/processed/summary_methods_reduction_datatype.csv")
  return (final_df)
}




method_list <- c("knn", "glm", "rf", "nb", "lda", "svmLinear", "nnet")
reduction_list <- c("pca", "cor")
type_list <- c("cat", "num")
comp_list <- seq(20, 20, 5)
num_cv <- 3
folds <- 5

loop_models_and_data(reduction_list, type_list, method_list, comp_list, num_cv, folds)





summary <- read.csv("data/processed/summary_methods_reduction_datatype.csv", row.names = 1)

ggplot(data=summary, aes(x=Method, y=Accuracy, fill=Type)) +
  geom_bar(stat="identity", position=position_dodge()) +
  facet_grid(. ~ Reduction) +
  ggtitle("Comparing Reduction Method, Data Type, and Model Implementation by Test Accuracy")
ggsave(path = "images/summary/",
       filename = "summary_results.png",
       dpi = 300, 
       width = 8,
       height = 5,
       units = "in")



## Functions - Visualization


graph_model <- function(method, pca_cor, num_cat, num_cv, folds, comp) {
  test_path <- paste("data/processed/", pca_cor, "_", num_cat, "_test.csv", sep="")
  test <- read.csv(test_path, row.names = 1)
  train_path <- paste("data/processed/", pca_cor, "_", num_cat, "_train.csv", sep="")
  train <- read.csv(train_path, row.names = 1)
  
  cv_model <- cross_val_model(train, method, num_cv, folds, comp)

  predicted <- data.frame(
    probability = predict(cv_model, newdata = test[4:(comp+3)], "prob")[,1],
    cancer=test$cancer
  )

  predicted <- predicted[
    order(predicted$probability, decreasing=FALSE),]
  predicted$rank <- 1:nrow(predicted)

  graph <- ggplot(data=predicted, aes(x=rank, y=probability)) +
    geom_point(aes(color=cancer), alpha=1, shape=4, stroke=2) +
    xlab("Index") +
    ylab("Predicted Probability") +
    #xlim(0, 1) +
    ggtitle(paste("(R) ", pca_cor, 
                  ", (T) ", num_cat, 
                  #", (C) ", comp, 
                  sep="")) +
    theme(legend.position = "right",
          #legend.text=element_text(size=5),
          legend.title=element_blank())
  
  return (graph)
}

probabilities_plot_loop_models <- function(reduction_list, type_list, method_list, comp_list, num_cv, folds) {
  method_dict <- list("knn" = "K-Nearest-Neighbor",
            "rf" = "Random Forest Classification",
            "glm" = "Logistic Regression",
            "nb" = "Naive Bayes",
            'lda' = "Linear Discriminant Analysis",
            "svmLinear" = "Support Vector Machine",
            "nnet" = "Neural Network")
  
  for (method in method_list){
    g_list <- list()
    i <- 1
    for (comp in comp_list){
      for (pca_cor in reduction_list){
        for (num_cat in type_list){
          g <- graph_model(method, pca_cor, num_cat, num_cv, folds, comp)
          g_list[[i]] <- g
          i <- i+1
        }
      }
    }
    full_title <- paste(method_dict[[method]], ": Predicted Probablites of (R) Reduction Method, (T) Type of Data")
    full_plot <- do.call("grid.arrange", c(g_list, ncol=2, top = full_title))
    ggsave(path = "images/model_probabilities/",
           filename = paste(method, ".png", sep = ""),
           plot = full_plot,
           dpi = 300, 
           width = 9,
           height = 6,
           units = "in")
  }



method_list <- c("knn", "glm", "rf", "nb", "lda", "svmLinear", "nnet")
reduction_list <- c("pca", "cor")
type_list <- c("cat", "num")
comp_list <- seq(20, 20, 5)
num_cv <- 3
folds <- 5

probabilities_plot_loop_models(reduction_list, type_list, method_list, comp_list, num_cv, folds)



