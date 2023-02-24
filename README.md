### **Model & Classification** 
>This NLP project is basaed on 3 machine learning models for categorizing  2 different target types in binary classification for predicting the sentiments on the chats dataset from SQL validator. The main approach is [Bag of words](https://en.wikipedia.org/wiki/Bag-of-words_model), specifically [TF-IDF vectorizer](https://en.wikipedia.org/wiki/Tf%E2%80%93idf).

>>We selected 3 models to run our sentiment analysis: 
- **[Support Vector Machines](https://en.wikipedia.org/wiki/Support_vector_machine)**
- **[Multinomial Naive Bayes](https://www.mygreatlearning.com/blog/multinomial-naive-bayes-explained/)**  
- **[Complement Naive Bayes](https://www.geeksforgeeks.org/complement-naive-bayes-cnb-algorithm/)**. 

>>We have taken two separate column for 2 differnet target class, Label and Category. Label column consists of 
- _Positive_ 1
- _Neutral_ 0

>>Category column is having 
- _Collaborative_ 1 
- _Troubleshooting_ 0. 

The activity diagram given below is simply depicted to provide the understanding of the workflow of the process

![walk-through](https://user-images.githubusercontent.com/83521671/220474513-0d0505a0-7990-4cf2-9d24-56396646790a.jpg)


### **Data Preprocessing**

>>The data collected was not in the correct format. Information cleaning is the way toward guaranteeing that information is right, predictable, and usable.
For example of cleaning process; the hyperlinks do not contribute as training words, we substituted them with regular expressions(troubleshooting). 
another example is, irrelevant words such as "customers", "students", "hi", etc have been avoided because they do not have any expressions. Punctuation such as "periods", "commas", "exclamation", etc have also been avoided for this reason.
>>
The [word cloud](https://boostlabs.com/what-are-word-clouds-value-simple-visualizations/) is created to visualize the words with most occurence. It has been created after the data preprocess and cleanning.  

![download](https://user-images.githubusercontent.com/83521671/220476057-b3354b1f-fb3b-4f6e-9cca-872ed396b2ba.png)

### **Validation & Criteria**
>>The TF-IDF score of each word is then split into two
parts- one for training, and the other for testing. We split 80% of the TF-IDF score value of
the words for training, and the remaining 20% for
testing.
This table shows the frequency distribution of the number of the sample collected from the SQLValidator. 
![Frequency of datasest](https://user-images.githubusercontent.com/83521671/221268358-d66ad774-aeed-4fb3-86a9-98777f2c431d.JPG)

>>In Validation we check the [Precision](https://en.wikipedia.org/wiki/Precision_and_recall), [Recall](https://en.wikipedia.org/wiki/Precision_and_recall),
[Accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision) and [F1 score](https://en.wikipedia.org/wiki/F-score) from the results of three different models. Our result has better Accuracy and
Precision value than the Recall F1 score. We have
set the accuracy/precision/F1-score to have a thresh-
old value. If the value is higher than 75% then the
error rate will be ”LOW” and we can use the model
for prediction, And if less than 75% then error rate
will be ”HIGH” and we will need hyper-tuning the
parameters for training the model again
>>
### **Models Outcome**
> #### **Support vector machine**-
>>Table 1 shows the classification reports for the linear
SVC method. The precision and f1-score for neutral
are 86.5% and 89.4% respectively. Whereas the pre-
cision and f1-score for positive are 69.8% and 61.2%
respectively. The accuracy for the model is 83.3%.
Table 2 shows the classification reports for the
linear SVC method. The precision and f1-score for
Troubleshooting are 85.7% and 69.9% respectively.
Whereas the precision and f1-score for Collaborative
are 86.5% and 91.2% respectively. The accuracy for
>>the model is 86.4%

> #### **Multinomial Naive Bayes**-
>>Table 3 shows the classification reports for the Multi-
nomial Naive Bias method. The precision and f1-
score for neutral are 80.8% and 88.5% respectively.
Whereas the precision and f1-score for positive are
78.9% and 40.5% respectively. The accuracy for the
model is 80.7%. Table 4 shows the classification reports for the
Multinomial Naive Bias method. The precision and
f1-score for Troubleshooting are 87.5% and 69.3%
respectively. Whereas the precision and f1-score for
Collaborative are 86.2% and 91.2% respectively. The
>>accuracy for the model is 86.4%.

> #### **Complement Naive Bayes**-
>>Table 5 shows the classification reports for the Com-
plement Naive Bayes method. The precision and f1-
score for neutral are 85.9% and 81.8% respectively.
Whereas the precision and f1-score for positive are
46.5% and 52.4% respectively. The accuracy for the
model is 73.7%.
Table 6 shows the classification reports for the
Complement Naive Bayes method. The precision and
f1-score for Troubleshooting are 76.9% and 70.8%
respectively. Whereas the precision and f1-score for
Collaborative are 88.0% and 90.4% respectively. The
>>accuracy for the model is 85.5%.

The evalution matrix of 3 models are given below on tabular form
![Score](https://user-images.githubusercontent.com/83521671/220477521-6f520912-ecdd-45ed-b7b5-0ef1b221c6b4.JPG)


