  def export_to_xl(self,existing_file_path):
    with pd.ExcelWriter(existing_file_path,mode="a", engine="openpyxl",if_sheet_exists="replace") as ex:
      report_1.to_excel(ex,sheet_name="SVM_label",startcol=2)   
      report_2.to_excel(ex,sheet_name="SVM_category",startcol=2)
      report_3.to_excel(ex,sheet_name="MNB_label",startcol=2)
      report_4.to_excel(ex,sheet_name="MNB_category",startcol=2)
      report_5.to_excel(ex,sheet_name="CNB_label",startcol=2)
      report_6.to_excel(ex,sheet_name="CNB_category",startcol=2)
