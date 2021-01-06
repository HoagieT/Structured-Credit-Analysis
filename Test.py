import pandas as pd
import numpy as np
import os
from datetime import datetime
import tkinter.filedialog
from datetime import timedelta
import math
import time
import calendar
from numba import jit
import CreditCardCF
import CLOPassThrough


os.chdir('D:\\办公同步云盘\\另类投资\\Structured Investment\\Credit_Card_CLO')
Collateral_df= pd.read_excel('Huixin33.xlsx', sheetname = 'Sheet1', encoding = 'gb18030')
PrinStartDate = datetime.strptime('2018-5-7', '%Y-%m-%d')
IntStartDate = datetime.strptime('2018-6-1', '%Y-%m-%d')
FeeStartDate = datetime.strptime('2018-6-1','%Y-%m-%d')
RampUpDate = datetime.strptime('2018-6-1', '%Y-%m-%d')
ClosingDate = datetime.strptime('2018-5-7', '%Y-%m-%d')
ClassA_Info = Data_Input('Bond_Info_Huixin33.xlsx')[0]
ClassB_Info = Data_Input('Bond_Info_Huixin33.xlsx')[1]
ClassC_Info = Data_Input('Bond_Info_Huixin33.xlsx')[2]
ClassSub_Info = Data_Input('Bond_Info_Huixin33.xlsx')[3]
Fee_Info = Data_Input('Bond_Info_Huixin33.xlsx')[4]



import CreditCardCF
#Static pool data input
StaticPool = pd.read_excel('Static_Pool.xlsx', sheetname = 'StaticPool', encoding = 'gb18030')
#Scenarios_V3=[[1.0,1.0],[1.0,2.0],[1.0,3.0],[2.0,1.0],[3.0,1.0]]
Scenarios_V3=[[1.0,1.0]]

HazardRate_Default_Scenarios = [HazardRates(StaticPool['CumulativeDefaultRate'], Scenarios_V3[i][0]) for i in range(len(Scenarios_V3))]
HazardRate_PrePayment_Scenarios = [HazardRates(StaticPool['CumulativePrePaymentRate'], Scenarios_V3[i][1]) for i in range(len(Scenarios_V3))]
HazardRate_Sum = [HazardRate_Default_Scenarios[0][i] + HazardRate_PrePayment_Scenarios[0][i] for i in range(len(HazardRate_Default_Scenarios[0]))]
Portion = [HazardRate_Default_Scenarios[0][i] / HazardRate_Sum[i] for i in range(len(HazardRate_Sum))]
np.random.uniform(0,1,1)[0] <= Portion[2]

CF_Collateral_Tong_Scenarios_V3 = [CreditCardCF.GenerateEmptyCollateralCFSpreadSheet('2018-5-7', '2021-11-30', 'M') for i in range(len(Scenarios_V3))]

for i in range(len(Scenarios_V3)):
    x = CreditCardCF.CalculateCF_SurvivalAnalysis(Collateral_DF = Collateral_df, CF_Sum_df= CF_Collateral_Tong_Scenarios_V3[i], Prin_Start_Date=PrinStartDate, Int_Start_Date=IntStartDate, Fee_Start_Date=FeeStartDate, HazardRate_Default = HazardRate_Default_Scenarios[i], HazardRate_PrePayment = HazardRate_PrePayment_Scenarios[i])
    
Result = [CF_Collateral_Tong_Scenarios_V3[0]['Default'].sum()/CF_Collateral_Tong_Scenarios_V3[0]['Balance'][0], CF_Collateral_Tong_Scenarios_V3[0]['PrePayment'].sum()/CF_Collateral_Tong_Scenarios_V3[0]['Balance'][0]]



for i in range(len(Scenarios_V3)):
    x = CreditCardCF.CalculateCF_SurvivalFunc(Collateral_DF = Collateral_df, CF_Sum_df= CF_Collateral_Tong_Scenarios_V3[i], Prin_Start_Date=PrinStartDate, Int_Start_Date=IntStartDate, Fee_Start_Date=FeeStartDate, HazardRate_List = HazardRate_Sum)