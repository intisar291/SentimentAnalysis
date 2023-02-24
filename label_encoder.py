def label_encoder():
    encoder=LabelEncoder()
    df["new_label"]=df["label"].apply(lambda x: 1 if x=="positive" else 0) 
    df["new_category"]=df["Category"].apply(lambda x: 1 if x=="collaborative" else 0 )
    print(self.df.head(5))
