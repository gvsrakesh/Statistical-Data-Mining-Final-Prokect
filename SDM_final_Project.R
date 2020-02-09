rm(list=ls(all = TRUE))
library(tm)
#install.packages('RTextTools')
#install.packages('RTextTools_1.4.2.tar.gz', lib='/Users/sk/Desktop',repos = NULL)
#library(RTextTools)
library(plyr)
library('stringr')
library('dplyr')
library(tidytext)
library(caret)
library(plyr)
library(stringr)
library(randomForest)
library(ggplot2)
library(dplyr)

setwd("/Users/rakeshguduru/Documents/UB Grad stuff/SDM 1/Final Project")
data = read.csv('Womens Clothing E-Commerce Reviews.csv',header = T,na.strings = c("",NA))
data = data[,-1]
data$Rating = as.factor(data$Rating)
data$Recommended.IND = as.factor(data$Recommended.IND)
summary(data)
sum(is.na(data$Review.Text)) #845 observations have no reviews
#Removing data with no reviews
data = subset(data,!is.na(data$Review.Text))
table(data$Rating)

library(RColorBrewer)
coul <- brewer.pal(5, "Pastel2") 

# Transform this data in %
df_age_rating = data[,c('Age','Rating')]
summary(df_age_rating$Age)
df_age_rating$Rating = as.numeric(as.character(df_age_rating$Rating))
df_age_rating$Age_bucket = ifelse(df_age_rating$Age <= 34,1,ifelse(df_age_rating$Age <= 52,2,3))
library(plyr)
groupColumns = c("Rating","Age_bucket")
dataColumns = c("Age")
res = ddply(df_age_rating, groupColumns, function(x) count(x[dataColumns]))

data_percentage <- apply(res,2, function(x){x*100/sum(x,na.rm=T)})

# Make a stacked barplot--> it will be in %!
barplot(data_percentage, col=coul , border="white", xlab="group")

dt = as.data.frame(reviews1.DtM.C$Rating_target)

reviews = c()
#Cleaning up reviews
for(i in c(1:length(data$Review.Text))){
  # Remove punctuations
  reviews[i] = gsub('[[:punct:]]', '', data$Review.Text[i])
  # Remove control characters
  reviews[i] = gsub('[[:cntrl:]]', '', reviews[i])  
  # Remove digits
  reviews[i] = gsub('\\d+', '', reviews[i])  
  # change to lower case:  
  reviews[i] = tolower(reviews[i])  
  # split into words. str_split is in the stringr package  
  #word.list = str_split(sentence, '\\s+')  
  # sometimes a list() is one level of hierarchy too much  
  #words = unlist(word.list)
}

######## Unigrams ########
reviews1 = Corpus(VectorSource(reviews))

# Document Term Matrix using TfIDF Values
strsplit_space_tokenizer <- function(x)
  unlist(strsplit(as.character(x), "[[:space:]]+"))

reviews1.DtM <- DocumentTermMatrix(reviews1, control = list(tokenize = strsplit_space_tokenizer,wordLengths = c(6, Inf),
                                                            stemming = TRUE))

dim(reviews1.DtM)

# top 100 frequent terms
freqterms100 <- findFreqTerms(reviews1.DtM, 100)
# removes sparse terms
dt_matrix <- removeSparseTerms(reviews1.DtM, 0.99)
# terms with frequency atleast >N(1-0.95) will be retained. N is total number of docs.

# Convert to dataframe
dt_matrix <- as.data.frame(as.matrix(dt_matrix))
dim(dt_matrix)

##### Adding the calculated scores
dt_matrix <- cbind(dt_matrix,data$Rating)
dim(dt_matrix)
colnames(dt_matrix)[which(colnames(dt_matrix)%in%c('data$Rating'))] = 'Rating_target'
dt_matrix$Rating_target = as.factor(ifelse(dt_matrix$Rating_target == 5,1,ifelse(dt_matrix$Rating_target %in% c('1','2','3'),0,2)))
summary(dt_matrix$Rating_target)
reviews1.DtM.C = droplevels(dt_matrix[dt_matrix$Rating_target != 2,])
dim(reviews1.DtM.C)
nrow(reviews1.DtM.C)

# Splitting data into train and test
## 75% of the sample size
train.index <- createDataPartition(reviews1.DtM.C$Rating_target, p = .75, list = FALSE)
train <- reviews1.DtM.C[ train.index,]
test  <- reviews1.DtM.C[-train.index,]

# Building random forest model
#install.packages('tictoc')
library(tictoc)
library(rfUtilities)
set.seed(123)
tic("rf model")
rf_model <- randomForest(Rating_target ~ ., data = train,ntree=50,mtry = 12,importance = TRUE)
#rf_model_cv = rf.crossValidation(rf_model,train,train$Rating_target,p=0.1,n=10)

toc()
#rf model run time = 118.942 sec elapsed
impAttributes = round(importance(rf_model,type=2), 3)


conf.mat_train = rf_model$confusion
conf.mat_train
accuracy_train = sum(diag(conf.mat_train))/sum(conf.mat_train)
precision_train = conf.mat_train[2,2]/sum(conf.mat_train[,2])
recall_train = conf.mat_train[2,2]/sum(conf.mat_train[2,])
accuracy_train#0.8056916
precision_train#0.8357001
recall_train#0.9309136

test_pred = predict(rf_model,test,type='response')
conf.mat_test = table(test$Rating_target,test_pred)
accuracy_test = sum(diag(conf.mat_test))/sum(conf.mat_test)
precision_test = conf.mat_test[2,2]/sum(conf.mat_test[,2])
recall_test = conf.mat_test[2,2]/sum(conf.mat_test[2,])
accuracy_test#0.8063604
precision_test#0.8347683
recall_test#0.9335167

summary(data$Department.Name)

# GBM
library(gbm)
library(e1071)
summary(train$Rating_target)
y_train = train$Rating_target
y_test = test$Rating_target
train$Rating_target = as.numeric(as.character(train$Rating_target))
test$Rating_target = as.numeric(as.character(test$Rating_target))

set.seed(123)
d = c(3,5,7,9)
s = c(0.005,0.01,0.05,0.1)

accuracy_df = data.frame(d= numeric(0), s= numeric(0),train_er = numeric(0), test_er = numeric(0))

for (i in 1:length(d))
{
  for (j in 1:length(s))
  {
    boost_fit = gbm(Rating_target~., data = train, n.trees = 500, interaction.depth = d[i], shrinkage = s[j],
                    distribution = "adaboost", cv.folds = 5)
    probs_train = predict(boost_fit, newdata = train, n.trees = 500, type = "response")
    probs_test = predict(boost_fit, newdata = test, n.trees = 500, type = "response")
    
    gbm_pred_train = as.factor(ifelse(probs_train > 0.5, 1, 0))
    gbm_pred_test = as.factor(ifelse(probs_test > 0.5, 1, 0))
    
    cm_test = confusionMatrix(y_test, gbm_pred_test)
    cm_train = confusionMatrix(y_train, gbm_pred_train)
    train_accuracy = cm_train$overall['Accuracy']
    test_accuracy = cm_test$overall['Accuracy']
    
    accuracy_df = rbind(accuracy_df, c(d[i],s[j], train_accuracy, test_accuracy))
  }
}

colnames(accuracy_df) = c("depth","lambda","train_accuracy","test_accuracy")

y_train = train$Rating_target
y_test = test$Rating_target
train$Rating_target = as.numeric(as.character(train$Rating_target))
test$Rating_target = as.numeric(as.character(test$Rating_target))

boost_fit = gbm(Rating_target~., data = train, n.trees = 500, interaction.depth = 9, shrinkage = 0.1,
                distribution = "adaboost", cv.folds = 5)
probs_train = predict(boost_fit, newdata = train, n.trees = 500, type = "response")
probs_test = predict(boost_fit, newdata = test, n.trees = 500, type = "response")

gbm_pred_train = as.factor(ifelse(probs_train > 0.5, 1, 0))
gbm_pred_test = as.factor(ifelse(probs_test > 0.5, 1, 0))


cm_train = confusionMatrix(y_train, gbm_pred_train)
conf.mat_train = cm_train$table
conf.mat_train
accuracy_train = sum(diag(conf.mat_train))/sum(conf.mat_train)
precision_train = conf.mat_train[1,1]/sum(conf.mat_train[,1])
recall_train = conf.mat_train[1,1]/sum(conf.mat_train[1,])
accuracy_train
precision_train
recall_train

cm_test = confusionMatrix(y_test, gbm_pred_test)
conf.mat_test = cm_test$table
accuracy_test = sum(diag(conf.mat_test))/sum(conf.mat_test)
precision_test = conf.mat_test[1,1]/sum(conf.mat_test[,1])
recall_test = conf.mat_test[1,1]/sum(conf.mat_test[1,])
accuracy_test
precision_test
recall_test

#########Smote ############
library(DMwR)
train$Rating_target = as.factor(train$Rating_target)
new_data_train = SMOTE(Rating_target ~ ., train, perc.over = 600, k = 15, perc.under = 100)

new_data_train$Rating_target = as.numeric(as.character(new_data_train$Rating_target))

boost_fit = gbm(Rating_target~., data = new_data_train, n.trees = 500, interaction.depth = 9, shrinkage = 0.1,
                distribution = "adaboost", cv.folds = 5)
probs_train = predict(boost_fit, newdata = new_data_train, n.trees = 500, type = "response")

y_train_new = as.factor(new_data_train$Rating_target)
gbm_pred_train = as.factor(ifelse(probs_train > 0.5, 1, 0))
cm_train = confusionMatrix(y_train_new, gbm_pred_train)

conf.mat_train = cm_train$table
conf.mat_train
accuracy_train = sum(diag(conf.mat_train))/sum(conf.mat_train)
precision_train = conf.mat_train[1,1]/sum(conf.mat_train[,1])
recall_train = conf.mat_train[1,1]/sum(conf.mat_train[1,])
accuracy_train
precision_train
recall_train

test$Rating_target = as.factor(test$Rating_target)
probs_test = predict(boost_fit, newdata = test, n.trees = 500, type = "response")
gbm_pred_test = as.factor(ifelse(probs_test > 0.5, 1, 0))
cm_test = confusionMatrix(test$Rating_target, gbm_pred_test)
conf.mat_test = cm_test$table
accuracy_test = sum(diag(conf.mat_test))/sum(conf.mat_test)
precision_test = conf.mat_test[1,1]/sum(conf.mat_test[,1])
recall_test = conf.mat_test[1,1]/sum(conf.mat_test[1,])
accuracy_test
precision_test
recall_test

#probs_test = predict(boost_fit, newdata = test, n.trees = 500, type = "response")


#############Wordclouds
words = rownames(impAttributes)
tmp1 = as.data.frame(impAttributes)
tmp1$words = words
freq = tmp1[order(-tmp1$MeanDecreaseGini),]
head(freq, 14)
wf = data.frame(word=words, freq=freq$MeanDecreaseGini)
head(wf)

## Plotting
## ----plot_freq, fig.width=12
library(ggplot2)
subset(wf, freq>20)                                                  %>%
  ggplot(aes(word, freq))                                              +
  geom_bar(stat="identity")                                            +
  theme(axis.text.x=element_text(angle=45, hjust=1))

## ----wordcloud-------
library(wordcloud)
set.seed(111)
wordcloud(freq$words, freq$MeanDecreaseGini, min.freq=10, colors=brewer.pal(6, "Dark2"))
