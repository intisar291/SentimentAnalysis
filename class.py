class Sentiment():

  stop=set(stopwords.words("english"))
  target_label_name=["Neutral","Positive"]
  target_category_name=["Troubleshooting","Collaborative"]


  def __init__(self,test):
    self.test=test
    input_test=pd.read_excel(test)
    self.df=pd.DataFrame(input_test)
    self.df.dropna(inplace=True)
    #print(self.df.head(2))
    #print(f"len of Dataframe index - {len(self.df.index)}")    


  def head(self,quantity):
   print(self.df.head(quantity))


  def label_encoder(self):
    encoder=LabelEncoder()
    self.df["new_label"]=self.df["label"].apply(lambda x: 1 if x=="positive" else 0) #encoder.fit_transform(df["label"])
    self.df["new_category"]=self.df["Category"].apply(lambda x: 1 if x=="collaborative" else 0 )
    #print(self.df.head(5))  


  def pre_processing(self,size_1=.2,state_1=30,size_2=0.2,state_2=30):
    self.df["pre_process"] = self.df["Updated post"].apply(lambda x: " ".join(x.lower() for x in str(x).split()))
    self.df["pre_process"]=self.df["pre_process"].apply(lambda x: " ".join([re.sub("[^A-Za-z]+"," ", x) for x in nltk.word_tokenize(x)]))
    self.df["pre_process"]=self.df["pre_process"].apply(lambda x: " ".join([x for x in x.split() if x not in self.stop]))
    
    self.X_train,self.X_test,self.Y_train,self.Y_test = train_test_split(self.df["pre_process"], self.df["new_label"], test_size=size_1, random_state=state_1)
    vectorizer_label=TfidfVectorizer()  
    self.tf_x_train = vectorizer_label.fit_transform(self.X_train)
    self.tf_x_test = vectorizer_label.transform(self.X_test) 

    self.X1_train,self.X1_test,self.Y1_train, self.Y1_test = train_test_split(self.df["pre_process"], self.df["new_category"], test_size=size_1, random_state=state_1)
    vectorizer_category=TfidfVectorizer()
    self.tf_x1_train = vectorizer_category.fit_transform(self.X1_train)
    self.tf_x1_test = vectorizer_category.transform(self.X1_test)

    words = " ".join(v for v in self.df["pre_process"])
    stop=set(STOPWORDS)
    stop = stop.union(["customers","students","insert","hi","varchar","create","table","primary", "null","foreign", "key","select","int","id","references","customer","product","name","supplier","student"])
    wordcloud = WordCloud(stopwords=stop,background_color='white',colormap="inferno", max_font_size = 30, max_words = 1000).generate(words)
    plt.figure(figsize=(12, 10))
    plt.imshow(wordcloud, interpolation= 'spline36')
    plt.axis('off')
    plt.show()
    
    fig, ax = plt.subplots()
    self.df['new_label'].value_counts().plot(ax=ax, kind='bar', xlabel='0=Neutral   1=Positive', ylabel='Frequency')
    plt.show()
    fig1, ax1 = plt.subplots()
    self.df['new_category'].value_counts().plot(ax=ax1, kind='bar', xlabel='0=Troubleshooting   1=Collaborative', ylabel='Frequency')
    plt.show()

  
  def svm(self,p_1="l1",c_1=1.4,l_1='squared_hinge',p_2="l1",c_2=0.8,l_2='squared_hinge',d=False,state=0):
    
    # Linear SVM for "label" column
    svm_model_label=LinearSVC(penalty=p_1,C=c_1,loss=l_1,dual=d,random_state=state)     
    svm_model_label.fit(self.tf_x_train,self.Y_train)
    y_test_pred=svm_model_label.predict(self.tf_x_test)

    # Linear SVM for "Category" column
    svm_model_category=LinearSVC(penalty=p_2,C=c_2,loss=l_2,dual=d,random_state=state)
    svm_model_category.fit(self.tf_x1_train,self.Y1_train)
    y1_test_pred=svm_model_category.predict(self.tf_x1_test)

    # error measurement
    report_label=classification_report(self.Y_test, y_test_pred,output_dict=False,target_names=self.target_label_name)   # "output_dict=False" will convert the output of into dictionary instead of table
    report_category=classification_report(self.Y1_test, y1_test_pred,output_dict=False,target_names=self.target_category_name)
    print(f"Validation matrix of Label column for SVM\n{report_label}\n")
    print(f"Validation matrix of Category column for SVM\n{report_category}\n")
    #g1 =  {k: report_label[k] for k in list(report_label)[:2]}
    #g2 =  {k: report_category[k] for k in list(report_category)[:2]}
    #return g1,g2


  def multi_NB(self,a_1=0.8,fp_1=False,cp_1=[.9,.3],a_2=0.5,fp_2=False,cp_2=[.3,.85]):
    
    # Multinomial Naive Bayes for "label" column
    model_mnb_label=MultinomialNB(alpha=a_1,fit_prior=fp_1,class_prior=cp_1)
    model_mnb_label.fit(self.tf_x_train,self.Y_train)
    y_test_predict=model_mnb_label.predict(self.tf_x_test)

    # Multinomial Naive Bayes for "Category" column
    model_mnb_category=MultinomialNB(alpha=a_2,fit_prior=fp_2,class_prior=cp_2)
    model_mnb_category.fit(self.tf_x1_train,self.Y1_train)
    y1_test_predict=model_mnb_category.predict(self.tf_x1_test)

    # Error measurement
    report_label=classification_report(self.Y_test,y_test_predict,output_dict=False,target_names=self.target_label_name)
    report_category=classification_report(self.Y1_test,y1_test_predict,output_dict=False,target_names=self.target_category_name)

    print(f"Validation matrix of Label column Multinomial Naive Bayes\n{report_label}\n")
    print(f"Validation matrix of Category column for Multinomial Naive Bayes\n{report_category}\n")


  def comp_NB(self):

    # Complement Naive Bayes for "label" column
    model_cnb_label=ComplementNB(alpha=1,fit_prior=False,class_prior=[.8,.5])
    model_cnb_label.fit(self.tf_x_train,self.Y_train)
    y_test_predict=model_cnb_label.predict(self.tf_x_test)

    # Complement Naive Bayes for "Category" column
    model_cnb_category=ComplementNB(alpha=1,fit_prior=False,class_prior=[.1,.5])
    model_cnb_category.fit(self.tf_x1_train,self.Y1_train)
    y1_test_predict=model_cnb_category.predict(self.tf_x1_test)

    # Error measurement
    report_label=classification_report(self.Y_test,y_test_predict,output_dict=False,target_names=self.target_label_name)
    report_category=classification_report(self.Y1_test,y1_test_predict,output_dict=False,target_names=self.target_category_name)

    print(f"Validation matrix of Label column for Complement Naive Bayes \n{report_label}\n")
    print(f"Validation matrix of Category column for Complement Naive Bayes \n{report_category}\n")

