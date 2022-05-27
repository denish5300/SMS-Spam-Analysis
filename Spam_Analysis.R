#Problem Description: Predicting whether the text is ham or spam
#Inputs:              SMS spam analysis 
#Output file:         Display values on the Console 
#========================================
#NOTE: Observations and comments are written below the respective code. 

#1.read-in the sms_spam.csv
###########################
sms_raw <- read.csv("sms_spam.csv")
View(sms_raw)

#explore data
str(sms_raw)

#there are 1 feature and 1 target feature. 

#covert type into factor
sms_raw$type <- factor(sms_raw$type)
str(sms_raw)

#how many ham and how many spam? 
table(sms_raw$type)

#percentage of each case
round(prop.table(table(sms_raw$type)) * 100, digits = 1)
#we have class imbalance issue, however we will ignore this issue in this practice for now. 

#2.data preparation - cleaning and standardizing text data because data is in unstructured format
#################################################################################################
library(NLP)
library(tm)

#the first step in processing text data involves creating: format text to corpus instead of tabular
sms_corpus <- VCorpus(VectorSource(sms_raw$text))

#examine the sms corpus: meta data and char length 
inspect(sms_corpus[1:2])

#to view an actual message
as.character(sms_corpus[[1]])

#to view multiple messages
lapply(sms_corpus[1:2], as.character)

#Text Clean-Up: 

#All lower case characters for dimensionality reduction
sms_corpus_clean <- tm_map(sms_corpus, content_transformer(tolower))

#show the difference between sms_corpus and corpus_clean
as.character(sms_corpus[[1]])
as.character(sms_corpus_clean[[1]])

#remove numbers from the SMS: 
sms_corpus_clean <- tm_map(sms_corpus_clean, removeNumbers)

#doublecheck if numbers are being removed
as.character(sms_corpus[[4]])
as.character(sms_corpus_clean[[4]])

#remove filler words such as to, and, but and or
sms_corpus_clean <- tm_map(sms_corpus_clean, removeWords, stopwords())

#doublecheck if filler words are being removed
as.character(sms_corpus[[1]])
as.character(sms_corpus_clean[[1]])

#remove punctuation
replacePunctuation <- function(x) {gsub("[[:punct:]]+"," ", x) }
sms_corpus_clean <- tm_map(sms_corpus_clean, removePunctuation)
as.character(sms_corpus[[1]])
as.character(sms_corpus_clean[[1]])

#reduce words to their root form in a process called stemming process
library(SnowballC)

sms_corpus_clean <- tm_map(sms_corpus_clean, stemDocument)
as.character(sms_corpus[[1]])
as.character(sms_corpus_clean[[1]])

#remove additional white-space
sms_corpus_clean <- tm_map(sms_corpus_clean, stripWhitespace)
as.character(sms_corpus[[55]])
as.character(sms_corpus_clean[[55]])

#splitting text documents into words --> tokenization
sms_dtm <- DocumentTermMatrix(sms_corpus_clean)
sms_dtm$ncol 
sms_dtm$nrow
sms_dtm$dimnames$Terms[1:3]

#after structuring the data, there are 6559 columns and 5559 rows in the data set. 

#create training (75%) and test (25%) data sets 
sms_dtm_train <- sms_dtm[1:4169, ]
sms_dtm_test <- sms_dtm[4170:5559, ]

#also, save the labels
sms_train_labels <- sms_raw[1:4169, ]$type
sms_test_labels <- sms_raw[4170:5559, ]$type

#check the proportion of spam to make sure it is similar
prop.table(table(sms_train_labels))
prop.table(table(sms_test_labels))

#visualize the data --> word clouds
library("wordcloud")
wordcloud(sms_corpus_clean, min.freq = 100, random.order = FALSE)

#visualize cloud from spam and ham
spam <- subset(sms_raw, type == "spam")
ham <- subset(sms_raw, type == "ham")

wordcloud(spam$text, max.words = 40, scale = c(3, 0.5))
wordcloud(ham$text, max.words = 40, scale = c(3, 0.5))

#reduce dimensionality
sms_dtm$ncol
sms_freq_words <- findFreqTerms(sms_dtm_train, 5)
sms_freq_words[1:10]

#create DTMs with only the frequent terms (i.e, words appearing at least 5 times)
sms_dtm_freq_train <- sms_dtm_train[, sms_freq_words]
sms_dtm_freq_test <- sms_dtm_test[, sms_freq_words]
sms_dtm_freq_train$ncol
          
#3.apply the Naive Bayes classifier
##################################
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

sms_train <- apply(sms_dtm_freq_train, MARGIN = 2, convert_counts)
sms_test <- apply(sms_dtm_freq_test, MARGIN = 2, convert_counts)

#train a model on the data
library(e1071)
sms_classifier <- naiveBayes(sms_train, sms_train_labels)

#predict on test data and evaluate the model performance
sms_test_pred <- predict(sms_classifier, sms_test)

#find agreement between the two vectors
library("gmodels")

CrossTable(sms_test_labels, sms_test_pred,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c("actual", "predicted"))

#improve model performance -> use Laplace estimator
sms_classifier2 <- naiveBayes(sms_train, sms_train_labels, laplace = 1)
sms_test_pred2 <- predict(sms_classifier2, sms_test)
CrossTable(sms_test_labels, sms_test_pred2,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c("actual", "predicted"))

#Observation: Looking at the table in the IMPROVED MODEL, we can see that a total 
#of only 16 + 18 = 34 of the 1,390 SMS messages were incorrectly 
#classified (2.4%) for both spam and ham.
#There are small number of spam messages slip through the spam filter and come in the ham inbox. 
#However, we still prefer the improved model because it is doing better on ham, not
#classified as spam for the fact that not many people usually check their spam inbox. 