# 存储全局变量
import pandas as pd

anomaly_list = []
anomaly_config = 'config/anomaly.yaml'
predict_periods = 4
predict_dataframe = None
before_df_list = [None] * 7
after_df_list = [None] * 7
expand_df_list = [None] * 7
mining_root_cause = [None] * 7
final_res = pd.DataFrame(columns=['predict_cause'])
row_num = 0
col_num = 0
derived_measure = True
