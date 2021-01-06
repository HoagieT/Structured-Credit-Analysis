import pandas as pd
import numpy as np
import os
from datetime import datetime
import tkinter.filedialog
from datetime import timedelta
import math
import ctypes
import time
import calendar
from numba import jit
import sklearn
from sklearn.externals import joblib
import scipy


@jit
def CalculateFirstBillDate (LoanIssueDate, BillDay):
    FirstBillDate = LoanIssueDate
    if (LoanIssueDate.day < BillDay):
        FirstBillDate = datetime(LoanIssueDate.year, LoanIssueDate.month, BillDay)
    if (LoanIssueDate.day >= BillDay):
        FirstBillDate = NextMonth(datetime(LoanIssueDate.year, LoanIssueDate.month, BillDay))
    return FirstBillDate

def NextMonth (CurrentDate, Day):
    while True:
        try:
            if (CurrentDate.month == 12):
                return datetime(CurrentDate.year+1, 1, Day)
            else:
                return datetime(CurrentDate.year, CurrentDate.month + 1, Day)
        except ValueError:
                return datetime(CurrentDate.year, CurrentDate.month + 2, 1)

def NextDate(CurrentDate, Day):
    Next = CurrentDate
    if (CurrentDate.day < Day):
        return datetime(CurrentDate.year, CurrentDate.month, Day)
    elif (CurrentDate.day > Day):
        return NextMonth(CurrentDate, Day)
    return Next

@jit
def LastDate(CurrentDate, Day):
    Last = CurrentDate
    if (CurrentDate.day > Day):
        Last = datetime(CurrentDate.year, CurrentDate.month, Day)
    else:
        if (CurrentDate.month == 1):
            Last = datetime(CurrentDate.year - 1, 12, Day)
        if (CurrentDate.month != 1):
            Last = datetime(CurrentDate.year, CurrentDate.month - 1, Day)
    return Last

def DateSeries(StartDate, NumberOfMonths, Lag):
    initial_value = 0
    DateList = [initial_value for j in range(NumberOfMonths)]
    DateList[0] = StartDate + timedelta(days = Lag)
    for i in range(NumberOfMonths):
        if (i==0):
            DateList[i] = StartDate + timedelta(days = Lag)
        if (i!=0):
            DateList[i] = NextMonth(DateList[i-1]-timedelta(days = Lag)) + timedelta(days = Lag)
    return  DateList

def Last_Day(CurrentDay):
    _,Number_of_Days = calendar.monthrange(CurrentDay.year, CurrentDay.month)
    Last_Day = datetime(CurrentDay.year, CurrentDay.month, Number_of_Days)
    return Last_Day
vLast_Day = np.vectorize(Last_Day)

def MonthsPassed(StartDate, EndDate, Day):
    First_Date = NextDate(StartDate, Day)
    Last_Date = LastDate(EndDate, Day)
    month = (Last_Date.year - First_Date.year)*12 + (Last_Date.month - First_Date.month) + 1
    return month

@jit
def DateCol(DateList, Date):
    DateCol = 0
    for k in range(len(DateList)-1):
        if ((Date > DateList[k]) & (Date <= DateList[k+1])):
            DateCol = k+1
    return DateCol


@jit
def CF_Expand(Loan_ID, Loan_df, Lag):
    Index = Loan_df[Loan_df['Loan_ID']==Loan_ID].index.values[0]
    IssueDate = Loan_df['Issue_Date'].iloc[Index]
    BillDate = Loan_df['Bill_Date'].iloc[Index]
    NumberOfPeriods = Loan_df['NPeriods'].iloc[Index]
    Prin = Loan_df['Monthly_Prin_Amnt'].iloc[Index]
    Fee = Loan_df['Monthly_Fee_Amnt'].iloc[Index]
    "Billing Date List"
    DateList_Bill = DateSeries(CalculateFirstBillDate(IssueDate, BillDate), NumberOfPeriods, Lag)
    "Initialize CF for the loan"
    CF_Loan = pd.DataFrame(data=np.zeros((len(DateList_Bill),4)),columns = ['Date','Principal','Fee','Interest'])
    CF_Loan['Date'] = DateList_Bill
    CF_Loan['Principal']=float(Prin)
    CF_Loan['Fee']=float(Fee)
    return CF_Loan

vCF_Expand = np.vectorize(CF_Expand, excluded =['Loan_df', 'Lag'])


def CF_Translate(CF_Expanded):
    Temp = CF_Expanded
    Last_Day(Temp['Date'][0])
    Temp['Date'] = vLast_Day(list(Temp['Date']))
    Temp = pd.pivot_table(Temp,index=['Date'],values=['Principal','Fee','Interest'],aggfunc=np.sum)
    return Temp

def OneFactorCopula(Factor, Rho, Lambda):
    Epsilon = math.sqrt(Rho)*Factor + math.sqrt(1-Rho)*np.random.normal(0, 1)
    Probability = 1/math.sqrt(2*3.14)*math.exp(0-Epsilon**2/2)
    Time_of_Default = 0 - 1/Lambda*math.log(Probability)
    Months_Survival = math.floor(Time_of_Default)
    return Months_Survival

def CDR(CashFlow, CDR, CPR):
    CF_CDR = pd.DataFrame(data=np.zeros((len(CashFlow['CollectionDate']),7)),columns = ['CollectionDate','Interest','Fee','Principal','Default','PrePayment','Balance'])
    CF_CDR['CollectionDate'] = CashFlow['CollectionDate']
    CF_CDR = CF_CDR.fillna(0.0)
    CF_CDR['Balance'][0] = CashFlow['Balance'][0]
    for i in range(len(CashFlow.index)):
        if (i==0):
            continue
        Default = CF_CDR['Balance'][i-1]*(1-(1-CDR)**(1/12))
        if (CashFlow['Principal'][i]/CashFlow['Balance'][i-1] < 1):
            Prepayment = CF_CDR['Balance'][i-1]*(1-(CashFlow['Principal'][i]/CashFlow['Balance'][i-1]))*(1-(1-CPR)**(1/12))
        else:
            Prepayment = 0
        CF_CDR['Fee'][i] = (CF_CDR['Balance'][i-1] - Default)/CF_CDR['Balance'][i-1]*CashFlow['Fee'][i]
        CF_CDR['Interest'][i] = (CF_CDR['Balance'][i-1] - Default)/CF_CDR['Balance'][i-1]*CashFlow['Interest'][i]
        CF_CDR['Principal'][i] = CashFlow['Principal'][i]*(CF_CDR['Balance'][i-1]-Default)/CashFlow['Balance'][i-1] + Prepayment
        CF_CDR['Balance'][i] = CF_CDR['Balance'][i-1] - CF_CDR['Principal'][i] - Default
        CF_CDR['Default'][i] = Default
    return CF_CDR


def ClearCFSpreadSheet(CFSpreadSheet):
    CFSpreadSheet['Interest']=0.0
    CFSpreadSheet['Principal']=0.0
    CFSpreadSheet['Default']=0.0
    CFSpreadSheet['PrePayment']=0.0
    CFSpreadSheet['Balance']=0.0
    return

@jit
def DateLoc(DateList, Date):
    DateCol = 0
    for k in range(len(DateList)-1):
        if ((Date > DateList[k]) & (Date <= DateList[k+1])):
            DateCol = k+1
    return DateCol


@jit
def GenerateEmptyCollateralCFSpreadSheet(Closing_Date, End_Date, Frequency):
    EndDate = datetime.strptime(End_Date, '%Y-%m-%d')
    ClosingDate = datetime.strptime(Closing_Date, '%Y-%m-%d')
    Freq = Frequency
    
    DateList_Sch = pd.date_range(Closing_Date, EndDate + timedelta(days=180), freq=Freq)
    CF_Empty = pd.DataFrame(data=np.zeros((len(DateList_Sch)+1,7)),columns = ['CollectionDate','Interest','Fee','Principal','Default','PrePayment','Balance'])
    CF_Empty['CollectionDate'][0] = ClosingDate
    CF_Empty['CollectionDate'][1:] = DateList_Sch
    CF_Empty = CF_Empty.fillna(0.0)
    return CF_Empty


@jit
def CalculateCF_Scheduled(Collateral_DF, CF_Sum_df, Prin_Start_Date, Int_Start_Date, Fee_Start_Date):
    ClearCFSpreadSheet(CF_Sum_df)
    CF_Sum_Dict_Prin = dict(zip(range(CF_Sum_df.shape[0]), np.zeros((1,CF_Sum_df.shape[0]))[0]))
    CF_Sum_Dict_Fee = dict(zip(range(CF_Sum_df.shape[0]), np.zeros((1,CF_Sum_df.shape[0]))[0]))
    
    for i in Collateral_DF.iterrows():
        print(i[1]['Loan_ID'])
        
        "Principal"
        TermsRemained_Prin = i[1]['NPeriods']- MonthsPassed(i[1]['Issue_Date'], Prin_Start_Date, i[1]['Bill_Date'])
        FirstPrinDateAfterClosing = Last_Day(NextDate(Prin_Start_Date, i[1]['Bill_Date']))
        PrinStartLoc = CF_Sum_df[CF_Sum_df['CollectionDate'] == FirstPrinDateAfterClosing].index.values[0]
        for j in range(PrinStartLoc,(PrinStartLoc+TermsRemained_Prin)):
            CF_Sum_Dict_Prin[j]+=i[1]['Monthly_Prin_Amnt']
        
        "Fee"
        TermsRemained_Fee = i[1]['NPeriods']- MonthsPassed(i[1]['Issue_Date'], Fee_Start_Date, i[1]['Bill_Date'])
        FirstFeeDateAfterClosing = Last_Day(NextDate(Fee_Start_Date, i[1]['Bill_Date']))
        FeeStartLoc = CF_Sum_df[CF_Sum_df['CollectionDate'] == FirstFeeDateAfterClosing].index.values[0]
        for j in range(FeeStartLoc,(FeeStartLoc+TermsRemained_Fee)):
            CF_Sum_Dict_Fee[j]+=i[1]['Monthly_Fee_Amnt']
        
    CF_Sum_df['Principal'] = pd.DataFrame.from_dict(CF_Sum_Dict_Prin, orient='index')
    CF_Sum_df['Fee'] = pd.DataFrame.from_dict(CF_Sum_Dict_Fee, orient='index')
    
    "Balance Update"
    CF_Sum_df['Balance'][0] = CF_Sum_df['Principal'].sum() + CF_Sum_df['Default'].sum()
    for i in range(1,len(CF_Sum_df['CollectionDate'])):
        CF_Sum_df['Balance'][i] = CF_Sum_df['Balance'][i-1] - CF_Sum_df['Principal'][i] - CF_Sum_df['Default'][i]
    return


def HazardRates(StaticPool, Amplify):
    ConditionalDefaultProbability = [StaticPool[0]*Amplify]
    L2 = [(StaticPool[i]*Amplify-StaticPool[i-1]*Amplify)/(1-StaticPool[i-1]*Amplify) for i in range(1,len(StaticPool))]
    ConditionalDefaultProbability.extend(L2)
    HazardRate = [0-12*math.log(1-ConditionalDefaultProbability[i]) for i in range (len(ConditionalDefaultProbability))]
    return HazardRate

def PrePaymentRates(MonthlyPrePaymentRateList, Amplify):
    PrePaymentRates = [MonthlyPrePaymentRateList['PrePaymentRate'][i]*Amplify for i in range(0,len(MonthlyPrePaymentRateList))]
    return PrePaymentRates


def HazardRateFunction(t, HazardRateList):
    N = math.floor(t/(1/12))
    return HazardRateList[N]


def SurvivalFunction(x, t, HazardRateList):
    Integration,_  = scipy.integrate.quad(HazardRateFunction, x, x+t,  HazardRateList)
    return math.exp(0-Integration)


def InverseSurvivalFunction(Prob, x, HazardRate_List, HRIntegration_df):
    S,_ = scipy.integrate.quad(HazardRateFunction, 0, x,  HazardRate_List)
    Temp = S - math.log(Prob)
    return ((HRIntegration_df <= Temp).sum()/12 - x)[0]



def CalculateCF_SurvivalAnalysis(Collateral_DF, CF_Sum_df, Prin_Start_Date, Int_Start_Date, Fee_Start_Date, HazardRate_Default, HazardRate_PrePayment):
    ClearCFSpreadSheet(CF_Sum_df)
    CF_Sum_Dict_Prin = dict(zip(range(CF_Sum_df.shape[0]), np.zeros((1,CF_Sum_df.shape[0]))[0]))
    CF_Sum_Dict_Fee = dict(zip(range(CF_Sum_df.shape[0]), np.zeros((1,CF_Sum_df.shape[0]))[0]))
    CF_Sum_Dict_Default = dict(zip(range(CF_Sum_df.shape[0]), np.zeros((1,CF_Sum_df.shape[0]))[0]))
    CF_Sum_Dict_PrePayment = dict(zip(range(CF_Sum_df.shape[0]), np.zeros((1,CF_Sum_df.shape[0]))[0]))
    
    CF_Sum_Dict_Default_Count = dict(zip(range(CF_Sum_df.shape[0]), np.zeros((1,CF_Sum_df.shape[0]))[0]))
    CF_Sum_Dict_PrePayment_Count = dict(zip(range(CF_Sum_df.shape[0]), np.zeros((1,CF_Sum_df.shape[0]))[0]))
    
    DefaultPortion = [HazardRate_Default[i] / (HazardRate_Default[i]+HazardRate_PrePayment[i]) for i in range(len(HazardRate_Default))]
    HazardRateList = [HazardRate_Default[i] + HazardRate_PrePayment[i] for i in range(len(HazardRate_Default))]
    HazardRateFuncIntegrationDF = pd.DataFrame([scipy.integrate.quad(HazardRateFunction, 0, (i+1)/12,  HazardRateList)[0] for i in range(len(HazardRateList))])
    q=0
    for i in Collateral_DF.iterrows():
        Probability,Probability_Default = np.random.uniform(0,1,2)
        SurvivalTime = InverseSurvivalFunction(Probability, (CF_Sum_df['CollectionDate'][0]-i[1]['Issue_Date']).days/365, HazardRateList, HazardRateFuncIntegrationDF)
        t = min(int(SurvivalTime + (CF_Sum_df['CollectionDate'][0]-i[1]['Issue_Date']).days/30),i[1]['NPeriods'])
        print(q)
        Indicator = (Probability_Default > DefaultPortion[t-1])
        print(Indicator, t)
      
        "Principal"
        TermsRemained_Prin = int(i[1]['Closing_Balance']/i[1]['Monthly_Prin_Amnt'])
        FirstPrinDateAfterClosing = Last_Day(NextDate(Prin_Start_Date, i[1]['Bill_Date']))
        PrinStartLoc = CF_Sum_df[CF_Sum_df['CollectionDate'] == FirstPrinDateAfterClosing].index.values[0]
        n =  int(min(SurvivalTime*12, TermsRemained_Prin))
        
        if (n<TermsRemained_Prin):
            if (Indicator==1):
                CF_Sum_Dict_Default_Count[PrinStartLoc + n] += 1
            if (Indicator==0):
                CF_Sum_Dict_PrePayment_Count[PrinStartLoc + n] += 1
        
        for j in range(PrinStartLoc, (PrinStartLoc + n) ):
            CF_Sum_Dict_Prin[j]+=i[1]['Monthly_Prin_Amnt']
        CF_Sum_Dict_PrePayment[PrinStartLoc + n ] += i[1]['Monthly_Prin_Amnt']*(TermsRemained_Prin-n)*(1-Indicator)
        CF_Sum_Dict_Default[PrinStartLoc + n ] += i[1]['Monthly_Prin_Amnt']*(TermsRemained_Prin-n)*Indicator
               
        "Fee"
        TermsRemained_Fee = i[1]['NPeriods']- MonthsPassed(i[1]['Issue_Date'], Fee_Start_Date, i[1]['Bill_Date'])
        FirstFeeDateAfterClosing = Last_Day(NextDate(Fee_Start_Date, i[1]['Bill_Date']))
        FeeStartLoc = CF_Sum_df[CF_Sum_df['CollectionDate'] == FirstFeeDateAfterClosing].index.values[0]
        for j in range(FeeStartLoc,(FeeStartLoc + int(min(SurvivalTime*12, TermsRemained_Fee)))):
            CF_Sum_Dict_Fee[j]+=i[1]['Monthly_Fee_Amnt']
        q+=1
    CF_Sum_df['Principal'] = pd.DataFrame.from_dict(CF_Sum_Dict_Prin, orient='index')
    CF_Sum_df['Fee'] = pd.DataFrame.from_dict(CF_Sum_Dict_Fee, orient='index')
    CF_Sum_df['Default'] = pd.DataFrame.from_dict(CF_Sum_Dict_Default, orient='index')
    CF_Sum_df['PrePayment'] = pd.DataFrame.from_dict(CF_Sum_Dict_PrePayment, orient='index')
    
    "Balance Update"
    CF_Sum_df['Balance'][0] = CF_Sum_df['Principal'].sum() + CF_Sum_df['Default'].sum() + CF_Sum_df['PrePayment'].sum()
    for i in range(1,len(CF_Sum_df['CollectionDate'])):
        CF_Sum_df['Balance'][i] = CF_Sum_df['Balance'][i-1] - CF_Sum_df['Principal'][i] - CF_Sum_df['Default'][i] - CF_Sum_df['PrePayment'][i]
    
    return [CF_Sum_Dict_Default_Count, CF_Sum_Dict_PrePayment_Count]