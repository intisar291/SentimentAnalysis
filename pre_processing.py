 def pre_processing(size_1=.2,state_1=30,size_2=0.2,state_2=30):
    df["pre_process"] = df["Updated post"].apply(lambda x: " ".join(x.lower() for x in str(x).split()))
    df["pre_process"]=df["pre_process"].apply(lambda x: " ".join([re.sub("[^A-Za-z]+"," ", x) for x in nltk.word_tokenize(x)]))
    df["pre_process"]=df["pre_process"].apply(lambda x: " ".join([x for x in x.split() if x not in stop]))
    df["pre_process"]=df["pre_process"].map(lambda x: re.sub(r'http://\S+|https://\S+',"",x))
    df.dropna(subset=["pre_process"],inplace=True)


    X_train,X_test,Y_train,Y_test = train_test_split(df["pre_process"], df["new_label"], test_size=size_1, random_state=state_1)
    vectorizer_label=TfidfVectorizer()  #use tfidf vectorizer methods or printout the shape of vectorizer
    tf_x_train = vectorizer_label.fit_transform(X_train)
    tf_x_test = vectorizer_label.transform(X_test) 

    X1_train,X1_test,Y1_train, Y1_test = train_test_split(df["pre_process"], df["new_category"], test_size=size_1, random_state=state_1)
    vectorizer_category=TfidfVectorizer()
    tf_x1_train = vectorizer_category.fit_transform(X1_train)
    tf_x1_test = vectorizer_category.transform(X1_test)

    words = " ".join(v for v in df["pre_process"])
    stop=set(STOPWORDS)
    stop = stop.union(["customers","students","insert","hi","varchar","create","table","primary", "null","foreign", "key","select","int","id","references","customer","product","name","supplier","student"])
    wordcloud = WordCloud(stopwords=stop,background_color='white',colormap="inferno", max_font_size = 30, max_words = 1000).generate(words)
    plt.figure(figsize=(12, 10))
    plt.imshow(wordcloud, interpolation= 'spline36')
    plt.axis('off')
    plt.show()
    
    fig, ax = plt.subplots()
    df['new_label'].value_counts().plot(ax=ax, kind='bar', xlabel='0=Neutral   1=Positive', ylabel='Frequency')
    plt.show()
    fig1, ax1 = plt.subplots()
    df['new_category'].value_counts().plot(ax=ax1, kind='bar', xlabel='0=Troubleshooting   1=Collaborative', ylabel='Frequency')
    plt.show()

    print(f'Total number of samples {len(df)}')
