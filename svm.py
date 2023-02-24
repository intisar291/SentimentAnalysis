def svm(p_1="l1",c_1=1.4,l_1='squared_hinge',p_2="l1",c_2=0.8,l_2='squared_hinge',d=False,state=0):
    
    # Linear SVM for "label" column
    svm_model_label=LinearSVC(penalty=p_1,C=c_1,loss=l_1,dual=d,random_state=state)     
    svm_model_label.fit(tf_x_train,Y_train)
    y_test_pred=svm_model_label.predict(tf_x_test)

    # Linear SVM for "Category" column
    svm_model_category=LinearSVC(penalty=p_2,C=c_2,loss=l_2,dual=d,random_state=state)
    svm_model_category.fit(tf_x1_train,Y1_train)
    y1_test_pred=svm_model_category.predict(tf_x1_test)

    # error measurement
    report_svm_label=classification_report(Y_test, y_test_pred,output_dict=False,target_names=target_label_name)   # "output_dict=False" will convert the output of into dictionary instead of table
    report_svm_category=classification_report(Y1_test, y1_test_pred,output_dict=False,target_names=target_category_name)
    
    report_1=classification_report(Y_test, y_test_pred,output_dict=True,target_names=target_label_name)
    report_1.update({"accuracy": {"precision": None, "recall": None, "f1-score": report_1["accuracy"], "support": None}})
    report_1=pd.DataFrame(report_1).transpose()
    report_2=classification_report(Y1_test, y1_test_pred,output_dict=True,target_names=target_category_name)
    report_2.update({"accuracy": {"precision": None, "recall": None, "f1-score": report_2["accuracy"], "support": None}})
    report_2=pd.DataFrame(report_2).transpose()


    ConfusionMatrixDisplay.from_predictions(Y_test,y_test_pred,cmap="plasma")
    plt.show()  
    print(f"\nValidation matrix of Label column for SVM\n{report_svm_label}\n")
    ConfusionMatrixDisplay.from_predictions(Y1_test,y1_test_pred,cmap="plasma")
    plt.show()
    print(f"\nValidation matrix of Category column for SVM\n{report_svm_category}\n")
