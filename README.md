
Used 3 machine learning models for categorizing  2 x 2  different target labels the sentiments on the chats. The program is written on python. As the dataset is confidential, only the structure is provided for understanding.

We selected three models to run our sentiment analysis: Support Vector Machines (SVM), Multinomial Naive Bayes, and Complement Naive Bayes. We have taken two separate column of Label and Category column. Where, Label column consists of 
- _Positive_ 
- _Neutral_ sentiment class

Category column is having 
- _Collaborative_ 
- _Troubleshooting_ class. 

The activity diagram given below is simply depicted to provide the understanding of the workflow of the process

![walk-through](https://user-images.githubusercontent.com/83521671/220474513-0d0505a0-7990-4cf2-9d24-56396646790a.jpg)


### **Data Preprocessing**
>>
>>The data collected was not in the correct format. Information cleaning is the way toward guaranteeing that information is right, predictable, and usable.
For example of cleaning process; the hyperlinks do not contribute as training words, we substituted them with regular expressions(troubleshooting). 
another example is, irrelevant words such as "customers", "students", "hi", etc have been avoided because they do not have any expressions. Punctuation such as >>"periods", "commas", "exclamation", etc have also been avoided for this reason.

The word cloud is created to visualize the words with most occurence. It has been created after the data preprocess and cleanning.  

![download](https://user-images.githubusercontent.com/83521671/220476057-b3354b1f-fb3b-4f6e-9cca-872ed396b2ba.png)



![Score](https://user-images.githubusercontent.com/83521671/220477521-6f520912-ecdd-45ed-b7b5-0ef1b221c6b4.JPG)


