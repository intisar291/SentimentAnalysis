def comp_NB(self,a_1=1.7,fp_1=False,cp_1=[.4,.8],a_2=1,fp_2=False,cp_2=[.1,.5]):

    # Complement Naive Bayes for "label" column
    cnb_model_label=ComplementNB(alpha=a_1,fit_prior=fp_1,class_prior=cp_1)
    cnb_model_label.fit(tf_x_train,Y_train)
    y_test_predict=cnb_model_label.predict(tf_x_test)

    # Complement Naive Bayes for "Category" column
    cnb_model_category=ComplementNB(alpha=a_2,fit_prior=fp_2,class_prior=cp_2)
    cnb_model_category.fit(tf_x1_train,Y1_train)
    y1_test_predict=cnb_model_category.predict(tf_x1_test)

    # Error measurement
    report_cnb_label=classification_report(Y_test,y_test_predict,output_dict=False,target_names=target_label_name)
    report_cnb_category=classification_report(Y1_test,y1_test_predict,output_dict=False,target_names=target_category_name)
    
    report_5= classification_report(Y_test,y_test_predict,output_dict=True,target_names=target_label_name)
    report_5.update({"accuracy": {"precision": None, "recall": None, "f1-score": report_5["accuracy"], "support": None}})
    report_5=pd.DataFrame(report_5).transpose()
    report_6=classification_report(Y1_test,y1_test_predict,output_dict=True,target_names=target_category_name)
    report_6.update({"accuracy": {"precision": None, "recall": None, "f1-score": report_6["accuracy"], "support": None}})
    report_6=pd.DataFrame(report_6).transpose() 

    ConfusionMatrixDisplay.from_predictions(Y_test,y_test_predict,cmap="plasma")
    plt.show()
    print(f"\nValidation matrix of Label column for Complement Naive Bayes \n{report_cnb_label}\n")
    ConfusionMatrixDisplay.from_predictions(Y1_test,y1_test_predict,cmap="plasma")
    plt.show()
    print(f"\nValidation matrix of Category column for Complement Naive Bayes \n{report_cnb_category}\n")
