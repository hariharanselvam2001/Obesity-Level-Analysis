file_path <- file.choose()
df <- read.csv(file_path)
setwd('Desktop/')
library(ggplot2)
library(reshape2)
library(dplyr)
library(rpart)
library(caret)
library(pROC)
df <-read.csv('Obesity.csv')
head(df, 5)

## EDA

# Null or missing value
sum(is.na(df))
##Found 24 duplicated rows
count_duplicated = sum(duplicated(df))
count_duplicated
#Remove duplicated rows
df=unique(df)

## Summary the data
summary(df)



## Exam the target label
target_count = as.data.frame(table(df$NObeyesdad))
ggplot(target_count, aes(x=Var1,y = Freq, label = Freq))+
         geom_bar(stat = 'identity',fill='steelblue')+
        geom_text(vjust = 2, size = 5) +
         labs(title = "Count of each Category in Target Label", x = "Target Label", y = "Count") 
## 7 categories of target label, and it's balanced dataset.


#Numerical variables
numeric_vars <- c("Age","Height","Weight")
print(numeric_vars)

#Distribution of numerical variables
for (var in numeric_vars) {
  plot = ggplot(df, aes(y = !!as.name(var))) +
          geom_boxplot(fill = "steelblue", color = "black") +
          labs(title = paste("Boxplot of", var), x = var, y = "Frequency")
  print(plot)}




#Categorical variables
df$FCVC = as.factor(as.integer(df$FCVC))
df$NCP = as.factor(as.integer(df$NCP))
df$CH2O = as.factor(as.integer(df$CH2O))
df$FAF = as.factor(as.integer(df$FAF))
df$TUE = as.factor(as.integer(df$TUE))


categorical_vars = names(df)[!(names(df) %in% numeric_vars)]
print(categorical_vars)
#Distribution of categorical variables
for (col in categorical_vars) {
  plot = ggplot(df, aes(x = !!as.name(col))) +
    geom_bar(fill = "steelblue") +
    labs(title = paste("Bar Chart of", col), x = col, y = "Count")
  print(plot)
}


##encode categorical_vars
encoded_df = data.frame(matrix(nrow = nrow(df), ncol = length(categorical_vars)))
colnames(encoded_df) <- categorical_vars
for (col in categorical_vars) {
  encoded_col = as.numeric(factor(df[[col]]))
  encoded_df[[col]] = encoded_col
}
encoded_df = cbind(encoded_df, df[numeric_vars])
print(encoded_df)



#Heat map of correlation matrix
correlation_matrix = cor(encoded_df)
round(correlation_matrix,2)
melted = melt(correlation_matrix)
ggplot(data=melted, aes(x=Var1, y=Var2, fill=value))+
  geom_tile()+
  geom_text(aes(Var1,Var2,label = round(value,2)))+
  scale_fill_gradient2(low = 'blue',high='red')+
  theme(panel.background = element_blank(),
        panel.grid = element_blank(),
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1))+
  ggtitle("Correlation Heatmap ")



#Decision Tree model

##Split data
set.seed(4644)
train_index <- createDataPartition(df$NObeyesdad, p = 0.7, list = FALSE)
train_data <- df[train_index, ]
test_data <- df[-train_index, ]



##Gain ratio Model##
####################
set.seed(4644)
model_gain_ratio = rpart(NObeyesdad ~ ., data = train_data,parms = list(split = "information"),method = 'class')
predictions_gain_ratio = predict(model_gain_ratio, newdata = test_data)
predictions_gain_ratio = colnames(predictions_gain_ratio)[max.col(predictions_gain_ratio)]
predictions_gain_ratio
##Model Summary
summary(model_gain_ratio)
##Confusion Matrix
conf_matrix_gain = table(predictions_gain_ratio, test_data$NObeyesdad)
conf_matrix_gain


#Heatmap
conf_gain = melt(conf_matrix_gain)
ggplot(data=conf_gain, aes(x=predictions_gain_ratio, y=Var2, fill=value))+
  geom_tile()+
  geom_text(aes(predictions_gain_ratio,Var2,label = value))+
  scale_fill_gradient2(low = 'blue',high='red')+
  theme(panel.background = element_blank(),
        panel.grid = element_blank(),
        axis.title.x = element_blank(),
        axis.title.y = element_blank())+
  ggtitle("Confusion Matrix_Gain Ratio ")

##Accuracy
accuracy_gain_ratio = sum(diag(conf_matrix_gain)) / sum(conf_matrix_gain)
accuracy_gain_ratio


##ROC-AUC##
###############################
class_labels = unique(predictions_gain_ratio)
roc_list1 = list()
data = test_data
for (label in class_labels) {
  data$binary_target = ifelse(data$NObeyesdad == label, 1, 0)
  probabilities = predict(model_gain_ratio, newdata = data, type = "prob")[, label]
  roc = roc(data$binary_target, probabilities)
  roc_list1[[label]] = roc
}

# Plot the stacked AUC plots
auc_values = sapply(roc_list1, function(roc) round(auc(roc), 3))
auc_df1 = data.frame(Class = unique(df$NObeyesdad), AUC = auc_values)

ggroc(roc_list1) +
  geom_line(size = .8) +
  labs(x = "False Positive Rate", y = "True Positive Rate", 
       title = "ROC-AUC Plots for Gain Model") +
  theme_minimal() +
  annotate("text", x = 0.25, y = 0.28, label = paste0(auc_df1$Class[1],"-AUC: ", round(auc_df1$AUC[1], 3)), 
          color = "black", size = 3, hjust = 1, vjust = -1) +
  annotate("text", x = 0.25, y = 0.25, label = paste0(auc_df1$Class[2],"-AUC: ", round(auc_df1$AUC[2], 3)), 
           color = "black", size = 3, hjust = 1, vjust = -1)+
  annotate("text", x = 0.25, y = 0.22, label = paste0(auc_df1$Class[3],"-AUC: ", round(auc_df1$AUC[3], 3)), 
         color = "black", size = 3, hjust = 1, vjust = -1)+
  annotate("text", x = 0.25, y = 0.19, label = paste0(auc_df1$Class[4],"-AUC: ", round(auc_df1$AUC[4], 3)), 
         color = "black", size = 3, hjust = 1, vjust = -1)+
  annotate("text", x = 0.25, y = 0.16, label = paste0(auc_df1$Class[5],"-AUC: ", round(auc_df1$AUC[5], 3)), 
         color = "black", size = 3, hjust = 1, vjust = -1)+
  annotate("text", x = 0.25, y = 0.13, label = paste0(auc_df1$Class[6],"-AUC: ", round(auc_df1$AUC[6], 3)), 
         color = "black", size = 3, hjust = 1, vjust = -1)+
  annotate("text", x = 0.25, y = 0.10, label = paste0(auc_df1$Class[7],"-AUC: ", round(auc_df1$AUC[7], 3)), 
         color = "black", size = 3, hjust = 1, vjust = -1)
  
  



##Gini Index Model##
##############
set.seed(4644)
model_gini <- rpart(NObeyesdad ~ ., data = train_data, parms = list(split = "gini"),method = 'class')
predictions_gini <- predict(model_gini, newdata = test_data)
predictions_gini = colnames(predictions_gini)[max.col(predictions_gini)]

##Model Summary
summary(model_gini)
##Confusion Matrix
conf_matrix_gini = table(predictions_gini, test_data$NObeyesdad)
conf_matrix_gini

conf_gini = melt(conf_matrix_gini)
ggplot(data=conf_gini, aes(x=predictions_gini, y=Var2, fill=value))+
  geom_tile()+
  geom_text(aes(predictions_gini,Var2,label = value))+
  scale_fill_gradient2(low = 'blue',high='red')+
  theme(panel.background = element_blank(),
        panel.grid = element_blank(),
        axis.title.x = element_blank(),
        axis.title.y = element_blank())+
  ggtitle("Confusion Matrix_Gini Index ")
##Accuracy
accuracy_gain_gini = sum(diag(conf_matrix_gini)) / sum(conf_matrix_gini)
accuracy_gain_gini


##ROC-AUC##
###############################
class_labels = unique(predictions_gini)
roc_list2 = list()
data = test_data
for (label in class_labels) {
  data$binary_target = ifelse(data$NObeyesdad == label, 1, 0)
  probabilities = predict(model_gini, newdata = data, type = "prob")[, label]
  roc = roc(data$binary_target, probabilities)
  roc_list2[[label]] = roc
}

# Plot the stacked AUC plots
auc_values2 = sapply(roc_list2, function(roc) round(auc(roc), 3))
auc_df2 = data.frame(Class = unique(df$NObeyesdad), AUC = auc_values2)
ggroc(roc_list2) +
  geom_line(size = .8) +
  labs(x = "False Positive Rate", y = "True Positive Rate", 
       title = "ROC-AUC Plots for Gini Model") +
  theme_minimal() +
  annotate("text", x = 0.25, y = 0.28, label = paste0(auc_df2$Class[1],"-AUC: ", round(auc_df2$AUC[1], 3)), 
           color = "black", size = 3, hjust = 1, vjust = -1) +
  annotate("text", x = 0.25, y = 0.25, label = paste0(auc_df2$Class[2],"-AUC: ", round(auc_df2$AUC[2], 3)), 
           color = "black", size = 3, hjust = 1, vjust = -1)+
  annotate("text", x = 0.25, y = 0.22, label = paste0(auc_df2$Class[3],"-AUC: ", round(auc_df2$AUC[3], 3)), 
           color = "black", size = 3, hjust = 1, vjust = -1)+
  annotate("text", x = 0.25, y = 0.19, label = paste0(auc_df2$Class[4],"-AUC: ", round(auc_df2$AUC[4], 3)), 
           color = "black", size = 3, hjust = 1, vjust = -1)+
  annotate("text", x = 0.25, y = 0.16, label = paste0(auc_df2$Class[5],"-AUC: ", round(auc_df2$AUC[5], 3)), 
           color = "black", size = 3, hjust = 1, vjust = -1)+
  annotate("text", x = 0.25, y = 0.13, label = paste0(auc_df2$Class[6],"-AUC: ", round(auc_df2$AUC[6], 3)), 
           color = "black", size = 3, hjust = 1, vjust = -1)+
  annotate("text", x = 0.25, y = 0.10, label = paste0(auc_df2$Class[7],"-AUC: ", round(auc_df2$AUC[7], 3)), 
           color = "black", size = 3, hjust = 1, vjust = -1)




model_gain_ratio$variable.importance
model_gini$variable.importance
