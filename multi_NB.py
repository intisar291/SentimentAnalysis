def multi_NB(self,a_1=0.8,fp_1=False,cp_1=[.9,.3],a_2=0.5,fp_2=False,cp_2=[.3,.85]):
    
    # Multinomial Naive Bayes for "label" column
    mnb_model_label=MultinomialNB(alpha=a_1,fit_prior=fp_1,class_prior=cp_1)
    mnb_model_label.fit(tf_x_train,Y_train)
    y_test_predict=mnb_model_label.predict(tf_x_test)

    # Multinomial Naive Bayes for "Category" column
    mnb_model_category=MultinomialNB(alpha=a_2,fit_prior=fp_2,class_prior=cp_2)
    mnb_model_category.fit(tf_x1_train,Y1_train)
    y1_test_predict=mnb_model_category.predict(tf_x1_test)

    # Error measurement
    report_mnb_label=classification_report(Y_test,y_test_predict,output_dict=False,target_names=target_label_name)
    report_mnb_category=classification_report(Y1_test,y1_test_predict,output_dict=False,target_names=target_category_name)

    report_3=classification_report(Y_test,y_test_predict,output_dict=True,target_names=target_label_name)
    report_3.update({"accuracy": {"precision": None, "recall": None, "f1-score": report_3["accuracy"], "support": None}})
    report_3=pd.DataFrame(report_3).transpose()
    report_4=classification_report(Y1_test,y1_test_predict,output_dict=True,target_names=target_category_name)
    report_4.update({"accuracy": {"precision": None, "recall": None, "f1-score": report_4["accuracy"], "support": None}})
    report_4=pd.DataFrame(report_4).transpose()
    
    ConfusionMatrixDisplay.from_predictions(Y_test,y_test_predict,cmap="plasma")
    plt.show()
    print(f"\nValidation matrix of Label column Multinomial Naive Bayes\n{report_mnb_label}\n")
    ConfusionMatrixDisplay.from_predictions(Y1_test,y1_test_predict,cmap="plasma")
    plt.show()
    print(f"\nValidation matrix of Category column for Multinomial Naive Bayes\n{report_mnb_category}\n")
