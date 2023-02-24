### **Model & Classification** 
>This NLP project is basaed on 3 machine learning models for categorizing  2 x 2  different target labels the sentiments on the chats dataset from SQL validator. The main approach is Bag of words, specifically TF-IDF vectorizer.

>>We selected 3 models to run our sentiment analysis: 
- **Support Vector Machines** (SVM) 
- **Multinomial Naive Bayes**  
- **Complement Naive Bayes**. 

>>We have taken two separate column for 2 differnet target class, Label and Category. Label column consists of 
- _Positive_ 
- _Neutral_ sentiment class

>>Category column is having 
- _Collaborative_ 
- _Troubleshooting_ class. 

The activity diagram given below is simply depicted to provide the understanding of the workflow of the process

![walk-through](https://user-images.githubusercontent.com/83521671/220474513-0d0505a0-7990-4cf2-9d24-56396646790a.jpg)


### **Data Preprocessing**

>>The data collected was not in the correct format. Information cleaning is the way toward guaranteeing that information is right, predictable, and usable.
For example of cleaning process; the hyperlinks do not contribute as training words, we substituted them with regular expressions(troubleshooting). 
another example is, irrelevant words such as "customers", "students", "hi", etc have been avoided because they do not have any expressions. Punctuation such as "periods", "commas", "exclamation", etc have also been avoided for this reason.
>>
The word cloud is created to visualize the words with most occurence. It has been created after the data preprocess and cleanning.  

![download](https://user-images.githubusercontent.com/83521671/220476057-b3354b1f-fb3b-4f6e-9cca-872ed396b2ba.png)

### **Models Outcome**
#### **Support vector machine**-
Table 1 shows the classification reports for the linear
SVC method. The precision and f1-score for neutral
are 86.5% and 89.4% respectively. Whereas the pre-
cision and f1-score for positive are 69.8% and 61.2%
respectively. The accuracy for the model is 83.3%.
Table 2 shows the classification reports for the
linear SVC method. The precision and f1-score for
Troubleshooting are 85.7% and 69.9% respectively.
Whereas the precision and f1-score for Collaborative
are 86.5% and 91.2% respectively. The accuracy for
the model is 86.4%
The evalution matrix of 3 models are given below on tabular form
![Score](https://user-images.githubusercontent.com/83521671/220477521-6f520912-ecdd-45ed-b7b5-0ef1b221c6b4.JPG)


