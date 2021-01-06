import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import math
from numba import jit

@jit
def DateRow(DateList, Date):
    DateRow = 0
    for k in range(len(DateList)-1):
        if ((Date > DateList[k]) & (Date <= DateList[k+1])):
            DateRow = k+1
    return DateRow

@jit
def CDR(CF_Sch, CDR, CPR):
    CF_CDR = pd.DataFrame(columns = ['CollectionDate','PaymentDate','Interest','Principal','Default','PrePayment','Balance'])
    CF_CDR['CollectionDate'] = CF_Sch['CollectionDate']
    CF_CDR['PaymentDate'] = CF_Sch['PaymentDate']
    CF_CDR = CF_CDR.fillna(0.0)
    CF_CDR['Balance'][0] = CF_Sch['Balance'][0]
    
    for i in range(len(CF_CDR.index)):
        if (i==0):
            continue
        Default = round(CF_CDR['Balance'][i-1]*(1-(1-CDR)**(1/4)),2)
        if (CF_Sch['Principal'][i]/CF_Sch['Balance'][i-1] < 1):
            Prepayment = round(CF_CDR['Balance'][i-1]*(1-(CF_Sch['Principal'][i]/CF_Sch['Balance'][i-1]))*(1-(1-CPR)**(1/4)),2)
        else:
            Prepayment = 0.00
        CF_CDR['Interest'][i] = round((CF_CDR['Balance'][i-1] - Default)/CF_CDR['Balance'][i-1]*CF_Sch['Interest'][i],2)
        CF_CDR['Principal'][i] = CF_Sch['Principal'][i]*(CF_CDR['Balance'][i-1]-Default)/CF_Sch['Balance'][i-1] + Prepayment
        CF_CDR['Balance'][i] = CF_CDR['Balance'][i-1] - CF_CDR['Principal'][i] - Default
        CF_CDR['Default'][i] = Default
        CF_CDR['PrePayment'][i] = Prepayment
    return CF_CDR

@jit
def Pass_Through_Waterfall(CF, Bond_Info, ClassA_Info, ClassB_Info, ClassC_Info):
    """Stressed Cash Flow"""
    CF_Stressed = CF
    
    """Calculation Datelist and Distribution Datelist"""
    Dates_Cal = CF['CollectionDate']
    Dates_Dist = CF['PaymentDate']
    
    """Principal Amount Initializing"""
    PrinAmnt_A = float(Bond_Info['PrinAmnt_A'][0])
    PrinAmnt_B = float(Bond_Info['PrinAmnt_B'][0])
    PrinAmnt_C = float(Bond_Info['PrinAmnt_C'][0])
    PrinAmnt_Sub = float(Bond_Info['PrinAmnt_Sub'][0])
    
    """Interest Rate Initializing"""
    IntRate_A = float(Bond_Info['IntRate_A'][0])
    IntRate_B = float(Bond_Info['IntRate_B'][0])
    IntRate_C = float(Bond_Info['IntRate_C'][0])
    IntRate_Sub = float(Bond_Info['IntRate_Sub'][0])
    
    """Fixed Fee"""
    InitialRatingFee = float(Bond_Info['InitialRatingFee'][0])
    TrackingRatingFee = float(Bond_Info['TrackingRatingFee'][0])
    LegalFee = float(Bond_Info['LegalFee'][0])
    AuditingFee = float(Bond_Info['AuditingFee'][0])
    AccountantFee = float(Bond_Info['AccountantFee'][0])
    ExchangeFee = float(Bond_Info['ExchangeFee'][0])
    
    """Floating Fee Rate"""
    TaxRate = float(Bond_Info['TaxRate'][0])
    SPVFeeRate = float(Bond_Info['SPVFeeRate'][0])
    CustodianFeeRate = float(Bond_Info['CustodianFeeRate'][0])
    ServiceFeeRate = float(Bond_Info['ServiceFeeRate'][0])
    
    """Cash Flow Dataframes Initializing"""
    Class_A = pd.DataFrame(columns = ['CollectionDate','PaymentDate','Interest','Principal','Balance'])
    Class_B = pd.DataFrame(columns = ['CollectionDate','PaymentDate','Interest','Principal','Balance'])
    Class_C = pd.DataFrame(columns = ['CollectionDate','PaymentDate','Interest','Principal','Balance'])
    Class_A_SubClass = pd.DataFrame(columns = ['CollectionDate','PaymentDate','A1_Interest','A1_Principal','A1_Balance','A2_Interest','A2_Principal','A2_Balance','A3_Interest','A3_Principal','A3_Balance','A4_Interest','A4_Principal','A4_Balance','A5_Interest','A5_Principal','A5_Balance','A6_Interest','A6_Principal','A6_Balance'])
    Class_B_SubClass = pd.DataFrame(columns = ['CollectionDate','PaymentDate','B1_Interest','B1_Principal','B1_Balance','B2_Interest','B2_Principal','B2_Balance','B3_Interest','B3_Principal','B3_Balance','B4_Interest','B4_Principal','B4_Balance','B5_Interest','B5_Principal','B5_Balance','B6_Interest','B6_Principal','B6_Balance'])
    Class_C_SubClass = pd.DataFrame(columns = ['CollectionDate','PaymentDate','C1_Interest','C1_Principal','C1_Balance','C2_Interest','C2_Principal','C2_Balance','C3_Interest','C3_Principal','C3_Balance','C4_Interest','C4_Principal','C4_Balance','C5_Interest','C5_Principal','C5_Balance','C6_Interest','C6_Principal','C6_Balance'])
    Subordinate = pd.DataFrame(columns = ['CollectionDate','PaymentDate','Interest','Principal','Balance','Residual'])
    WaterFall = pd.DataFrame(columns = ['CollectionDate','PaymentDate','Int_Account_CF','Prin_Account_CF','Default','PrePayment','Amnt_Brought_Forward','Tax','SPVFee','CustodianFee','ServiceFee','InitialRatingFee','TrackingRatingFee','LegalFee','AuditingFee','AccountantFee','ExchangeFee','Int_A','Int_B','Int_C','Int_Sub','Prin_A','Prin_B','Prin_C','Prin_Sub','Residual'])
    
    """Update dates for each CF dataframe"""
    Class_A['CollectionDate'] = Dates_Cal
    Class_A_SubClass['CollectionDate'] = Dates_Cal
    Class_B['CollectionDate'] = Dates_Cal
    Class_B_SubClass['CollectionDate'] = Dates_Cal
    Class_C['CollectionDate'] = Dates_Cal
    Class_C_SubClass['CollectionDate'] = Dates_Cal
    Subordinate['CollectionDate'] = Dates_Cal
    WaterFall['CollectionDate'] = Dates_Cal
    Class_A['PaymentDate'] = Dates_Dist
    Class_A_SubClass['PaymentDate'] = Dates_Dist
    Class_B['PaymentDate'] = Dates_Dist
    Class_B_SubClass['PaymentDate'] = Dates_Dist
    Class_C['PaymentDate'] = Dates_Dist
    Class_C_SubClass['PaymentDate'] = Dates_Dist
    Subordinate['PaymentDate'] = Dates_Dist
    WaterFall['PaymentDate'] = Dates_Dist
    WaterFall['Default'] = CF_Stressed['Default']
    WaterFall['PrePayment'] = CF_Stressed['PrePayment']
    
    Class_A = Class_A.fillna(0.0)
    Class_A_SubClass = Class_A_SubClass.fillna(0.0)
    Class_B = Class_B.fillna(0.0)
    Class_B_SubClass = Class_B_SubClass.fillna(0.0)
    Class_C = Class_C.fillna(0.0)
    Class_C_SubClass = Class_C_SubClass.fillna(0.0)
    Subordinate = Subordinate.fillna(0.0)
    WaterFall = WaterFall.fillna(0.0)
    #Balance Input
    Class_A['Balance'][0] = PrinAmnt_A
    Class_B['Balance'][0] = PrinAmnt_B
    Class_C['Balance'][0] = PrinAmnt_C
    Subordinate['Balance'][0] = PrinAmnt_Sub
    for i in range(6):
        Class_A_SubClass.iloc[0,3*(i+1)+1] = ClassA_Info.iloc[0,2*i]
        Class_B_SubClass.iloc[0,3*(i+1)+1] = ClassB_Info.iloc[0,2*i]
        Class_C_SubClass.iloc[0,3*(i+1)+1] = ClassC_Info.iloc[0,2*i]
    
    WaterFall['Int_Account_CF'] = CF_Stressed['Interest']
    WaterFall['Prin_Account_CF'] = CF_Stressed['Principal']
    
    """Waterfall"""
    for i in range(len(WaterFall.index)):
        if (i==0):
            continue
        WaterFall['Amnt_Brought_Forward'][i] = WaterFall['Residual'][i-1] - Subordinate['Residual'][i-1]
        Days_Dist = WaterFall['PaymentDate'][i] - WaterFall['PaymentDate'][i-1]
        Days_Dist = float(Days_Dist.days)
        Days_Cal = WaterFall['CollectionDate'][i] - WaterFall['CollectionDate'][i-1]
        Days_Cal = float(Days_Cal.days)
        CF_Total = CF_Stressed['Principal'][i] + CF_Stressed['Interest'][i] + WaterFall['Amnt_Brought_Forward'][i]
        if ((CF_Stressed['Principal'][i] + CF_Stressed['Interest'][i])==0):
            continue
        """Initial Balance"""
        Sum_ClassA_Bal =  Class_A_SubClass['A1_Balance'][i-1]+Class_A_SubClass['A2_Balance'][i-1]+Class_A_SubClass['A3_Balance'][i-1]+Class_A_SubClass['A4_Balance'][i-1]+Class_A_SubClass['A5_Balance'][i-1]+Class_A_SubClass['A6_Balance'][i-1]
        Sum_ClassB_Bal =  Class_B_SubClass['B1_Balance'][i-1]+Class_B_SubClass['B2_Balance'][i-1]+Class_B_SubClass['B3_Balance'][i-1]+Class_B_SubClass['B4_Balance'][i-1]+Class_B_SubClass['B5_Balance'][i-1]+Class_B_SubClass['B6_Balance'][i-1]
        Sum_ClassC_Bal =  Class_C_SubClass['C1_Balance'][i-1]+Class_C_SubClass['C2_Balance'][i-1]+Class_C_SubClass['C3_Balance'][i-1]+Class_C_SubClass['C4_Balance'][i-1]+Class_C_SubClass['C5_Balance'][i-1]+Class_C_SubClass['C6_Balance'][i-1]
        
        Balance_Begin = Sum_ClassA_Bal + Sum_ClassB_Bal  + Sum_ClassC_Bal + Subordinate['Balance'][i-1]
        
        """Floating fees before senior interest"""
        WaterFall['Tax'][i] = round(CF_Stressed['Interest'][i]*TaxRate,2)
        WaterFall['SPVFee'][i] = round(Balance_Begin*SPVFeeRate*Days_Dist/365,2)
        WaterFall['CustodianFee'][i] = round(Balance_Begin*CustodianFeeRate*Days_Dist/365,2)
        WaterFall['ServiceFee'][i] = round(Balance_Begin*ServiceFeeRate*Days_Dist/365,2)
        
        """Fixed fees before senior interest"""
        if (i==1):
            WaterFall['InitialRatingFee'][i] = InitialRatingFee
            WaterFall['LegalFee'][i] = LegalFee
            WaterFall['AuditingFee'][i] = AuditingFee
            WaterFall['AccountantFee'][i] = AccountantFee
            WaterFall['ExchangeFee'][i] = ExchangeFee
        if ((WaterFall['PaymentDate'][i].month == 7)&(Balance_Begin!=0)):
            WaterFall['TrackingRatingFee'][i] = TrackingRatingFee
        
        Fees_Before_Int = WaterFall['Tax'][i] + WaterFall['SPVFee'][i] + WaterFall['CustodianFee'][i] + WaterFall['ServiceFee'][i] + WaterFall['InitialRatingFee'][i] + WaterFall['LegalFee'][i] + WaterFall['AuditingFee'][i] + WaterFall['AccountantFee'][i] + WaterFall['ExchangeFee'][i] + WaterFall['TrackingRatingFee'][i]
        
        """Senior Interest"""
        for j in range(6):
            Class_A_SubClass.iloc[i,3*(j+1)-1] = round(ClassA_Info.iloc[0,2*j+1]*Days_Dist/365*Class_A_SubClass.iloc[i-1,3*(j+1)+1],2)
            Class_B_SubClass.iloc[i,3*(j+1)-1] = round(ClassB_Info.iloc[0,2*j+1]*Days_Dist/365*Class_B_SubClass.iloc[i-1,3*(j+1)+1],2)
            Class_C_SubClass.iloc[i,3*(j+1)-1] = round(ClassC_Info.iloc[0,2*j+1]*Days_Dist/365*Class_C_SubClass.iloc[i-1,3*(j+1)+1],2)
        WaterFall['Int_A'][i] =  Class_A_SubClass.iloc[i,2:].sum()
        WaterFall['Int_B'][i] =  Class_B_SubClass.iloc[i,2:].sum()
        WaterFall['Int_C'][i] =  Class_C_SubClass.iloc[i,2:].sum()
            
        """Senior Interest Sum"""
        Sum_Senior_Int = WaterFall['Int_A'][i] + WaterFall['Int_B'][i] + WaterFall['Int_C'][i]
        if (Sum_Senior_Int>CF_Total):
            print("Unsufficient cash flow in %s period" %(i+1))
        """Subordinate Interest"""
        if ((Sum_ClassA_Bal + Sum_ClassB_Bal + Sum_ClassC_Bal) != 0.0):
            WaterFall['Int_Sub'][i] = round(min(Subordinate['Balance'][i-1]*Days_Dist*IntRate_Sub/365, max(0.0,CF_Stressed['Interest'][i]-Fees_Before_Int-Sum_Senior_Int+ WaterFall['Residual'][i-1]-Subordinate['Residual'][i-1])),2)
        else:
            WaterFall['Int_Sub'][i] = 0.0
        Sum_Int = Sum_Senior_Int + WaterFall['Int_Sub'][i]
        Subordinate['Interest'][i]=WaterFall['Int_Sub'][i]
        
        """Class A Principals"""
        Funds_For_Prin_A = CF_Total - Fees_Before_Int - Sum_Int
        if (Funds_For_Prin_A<0):
            print("Unsufficient cash flow in %s period" %(i+1))
        Funds_For_SubClass1 = max(0.0,Funds_For_Prin_A)
        Class_A_SubClass['A1_Principal'][i] = math.floor(min(Funds_For_SubClass1, Class_A_SubClass['A1_Balance'][i-1]))
        Class_A_SubClass['A1_Balance'][i] = Class_A_SubClass['A1_Balance'][i-1] - Class_A_SubClass['A1_Principal'][i]
        Funds_For_SubClass2 = Funds_For_SubClass1 - Class_A_SubClass['A1_Principal'][i]
        Class_A_SubClass['A2_Principal'][i] = math.floor(min(Funds_For_SubClass2, Class_A_SubClass['A2_Balance'][i-1]))
        Class_A_SubClass['A2_Balance'][i] = Class_A_SubClass['A2_Balance'][i-1] - Class_A_SubClass['A2_Principal'][i]
        Funds_For_SubClass3 = Funds_For_SubClass2 - Class_A_SubClass['A2_Principal'][i]
        Class_A_SubClass['A3_Principal'][i] = math.floor(min(Funds_For_SubClass3, Class_A_SubClass['A3_Balance'][i-1]))
        Class_A_SubClass['A3_Balance'][i] = Class_A_SubClass['A3_Balance'][i-1] - Class_A_SubClass['A3_Principal'][i]
        Funds_For_SubClass4 = Funds_For_SubClass3 - Class_A_SubClass['A3_Principal'][i]
        Class_A_SubClass['A4_Principal'][i] = math.floor(min(Funds_For_SubClass4, Class_A_SubClass['A4_Balance'][i-1]))
        Class_A_SubClass['A4_Balance'][i] = Class_A_SubClass['A4_Balance'][i-1] - Class_A_SubClass['A4_Principal'][i]
        Funds_For_SubClass5 = Funds_For_SubClass4 - Class_A_SubClass['A4_Principal'][i]
        Class_A_SubClass['A5_Principal'][i] = math.floor(min(Funds_For_SubClass5, Class_A_SubClass['A5_Balance'][i-1]))
        Class_A_SubClass['A5_Balance'][i] = Class_A_SubClass['A5_Balance'][i-1] - Class_A_SubClass['A5_Principal'][i]
        Funds_For_SubClass6 = Funds_For_SubClass5 - Class_A_SubClass['A5_Principal'][i]
        Class_A_SubClass['A6_Principal'][i] = math.floor(min(Funds_For_SubClass6, Class_A_SubClass['A6_Balance'][i-1]))
        Class_A_SubClass['A6_Balance'][i] = Class_A_SubClass['A6_Balance'][i-1] - Class_A_SubClass['A6_Principal'][i]
        WaterFall['Prin_A'][i] = Class_A_SubClass['A1_Principal'][i]+Class_A_SubClass['A2_Principal'][i]+Class_A_SubClass['A3_Principal'][i]+Class_A_SubClass['A4_Principal'][i]+Class_A_SubClass['A5_Principal'][i]+Class_A_SubClass['A6_Principal'][i]

        """Class B Principals"""
        Funds_For_Prin_B = Funds_For_Prin_A -  WaterFall['Prin_A'][i]
        Funds_For_SubClass1 = Funds_For_Prin_B
        Class_B_SubClass['B1_Principal'][i] = math.floor(min(Funds_For_SubClass1, Class_B_SubClass['B1_Balance'][i-1]))
        Class_B_SubClass['B1_Balance'][i] = Class_B_SubClass['B1_Balance'][i-1] - Class_B_SubClass['B1_Principal'][i]
        Funds_For_SubClass2 = Funds_For_SubClass1 - Class_B_SubClass['B1_Principal'][i]
        Class_B_SubClass['B2_Principal'][i] = math.floor(min(Funds_For_SubClass2, Class_B_SubClass['B2_Balance'][i-1]))
        Class_B_SubClass['B2_Balance'][i] = Class_B_SubClass['B2_Balance'][i-1] - Class_B_SubClass['B2_Principal'][i]
        Funds_For_SubClass3 = Funds_For_SubClass2 - Class_B_SubClass['B2_Principal'][i]
        Class_B_SubClass['B3_Principal'][i] = math.floor(min(Funds_For_SubClass3, Class_B_SubClass['B3_Balance'][i-1]))
        Class_B_SubClass['B3_Balance'][i] = Class_B_SubClass['B3_Balance'][i-1] - Class_B_SubClass['B3_Principal'][i]
        Funds_For_SubClass4 = Funds_For_SubClass3 - Class_B_SubClass['B3_Principal'][i]
        Class_B_SubClass['B4_Principal'][i] = math.floor(min(Funds_For_SubClass4, Class_B_SubClass['B4_Balance'][i-1]))
        Class_B_SubClass['B4_Balance'][i] = Class_B_SubClass['B4_Balance'][i-1] - Class_B_SubClass['B4_Principal'][i]
        Funds_For_SubClass5 = Funds_For_SubClass4 - Class_B_SubClass['B4_Principal'][i]
        Class_B_SubClass['B5_Principal'][i] = math.floor(min(Funds_For_SubClass5, Class_B_SubClass['B5_Balance'][i-1]))
        Class_B_SubClass['B5_Balance'][i] = Class_B_SubClass['B5_Balance'][i-1] - Class_B_SubClass['B5_Principal'][i]
        Funds_For_SubClass6 = Funds_For_SubClass5 - Class_B_SubClass['B5_Principal'][i]
        Class_B_SubClass['B6_Principal'][i] = math.floor(min(Funds_For_SubClass6, Class_B_SubClass['B6_Balance'][i-1]))
        Class_B_SubClass['B6_Balance'][i] = Class_B_SubClass['B6_Balance'][i-1] - Class_B_SubClass['B6_Principal'][i]
        WaterFall['Prin_B'][i] = Class_B_SubClass['B1_Principal'][i]+Class_B_SubClass['B2_Principal'][i]+Class_B_SubClass['B3_Principal'][i]+Class_B_SubClass['B4_Principal'][i]+Class_B_SubClass['B5_Principal'][i]+Class_B_SubClass['B6_Principal'][i]
        
        
        """Class C Principals"""
        Funds_For_Prin_C = Funds_For_Prin_B -  WaterFall['Prin_B'][i]
        Funds_For_SubClass1 = Funds_For_Prin_C
        Class_C_SubClass['C1_Principal'][i] = math.floor(min(Funds_For_SubClass1, Class_C_SubClass['C1_Balance'][i-1]))
        Class_C_SubClass['C1_Balance'][i] = Class_C_SubClass['C1_Balance'][i-1] - Class_C_SubClass['C1_Principal'][i]
        Funds_For_SubClass2 = Funds_For_SubClass1 - Class_C_SubClass['C1_Principal'][i]
        Class_C_SubClass['C2_Principal'][i] = math.floor(min(Funds_For_SubClass2, Class_C_SubClass['C2_Balance'][i-1]))
        Class_C_SubClass['C2_Balance'][i] = Class_C_SubClass['C2_Balance'][i-1] - Class_C_SubClass['C2_Principal'][i]
        Funds_For_SubClass3 = Funds_For_SubClass2 - Class_C_SubClass['C2_Principal'][i]
        Class_C_SubClass['C3_Principal'][i] = math.floor(min(Funds_For_SubClass3, Class_C_SubClass['C3_Balance'][i-1]))
        Class_C_SubClass['C3_Balance'][i] = Class_C_SubClass['C3_Balance'][i-1] - Class_C_SubClass['C3_Principal'][i]
        Funds_For_SubClass4 = Funds_For_SubClass3 - Class_C_SubClass['C3_Principal'][i]
        Class_C_SubClass['C4_Principal'][i] = math.floor(min(Funds_For_SubClass4, Class_C_SubClass['C4_Balance'][i-1]))
        Class_C_SubClass['C4_Balance'][i] = Class_C_SubClass['C4_Balance'][i-1] - Class_C_SubClass['C4_Principal'][i]
        Funds_For_SubClass5 = Funds_For_SubClass4 - Class_C_SubClass['C4_Principal'][i]
        Class_C_SubClass['C5_Principal'][i] = math.floor(min(Funds_For_SubClass5, Class_C_SubClass['C5_Balance'][i-1]))
        Class_C_SubClass['C5_Balance'][i] = Class_C_SubClass['C5_Balance'][i-1] - Class_C_SubClass['C5_Principal'][i]
        Funds_For_SubClass6 = Funds_For_SubClass5 - Class_C_SubClass['C5_Principal'][i]
        Class_C_SubClass['C6_Principal'][i] = math.floor(min(Funds_For_SubClass6, Class_C_SubClass['C6_Balance'][i-1]))
        Class_C_SubClass['C6_Balance'][i] = Class_C_SubClass['C6_Balance'][i-1] - Class_C_SubClass['C6_Principal'][i]
        WaterFall['Prin_C'][i] = Class_C_SubClass['C1_Principal'][i]+Class_C_SubClass['C2_Principal'][i]+Class_C_SubClass['C3_Principal'][i]+Class_C_SubClass['C4_Principal'][i]+Class_C_SubClass['C5_Principal'][i]+Class_C_SubClass['C6_Principal'][i]
        
        """Subordinate Principal"""
        Funds_For_Prin_Sub = Funds_For_Prin_C - WaterFall['Prin_C'][i]
        WaterFall['Prin_Sub'][i] = math.floor(min(Funds_For_Prin_Sub, Subordinate['Balance'][i-1]))
        Subordinate['Principal'][i] = WaterFall['Prin_Sub'][i]
        Subordinate['Balance'][i] = Subordinate['Balance'][i-1] - Subordinate['Principal'][i]
        
        """Residuals for subordinate certificate holders"""
        Funds_For_Residual = Funds_For_Prin_Sub - WaterFall['Prin_Sub'][i]
        WaterFall['Residual'][i] = Funds_For_Residual
        if (Subordinate['Balance'][i]==0):
            Subordinate['Residual'][i] = WaterFall['Residual'][i]
    
    CF = [Class_A_SubClass, Class_B_SubClass, Class_C_SubClass, Subordinate, WaterFall]
    return CF

@jit
def LoanStartDate(CollateralCF):
    Date = CollateralCF['Date'].iloc[0]
    for i in range(len(CollateralCF.index)):
        if (i==0):
            continue
    else:
        if ((CollateralCF['Balance'].iloc[i]!=0) & (CollateralCF['Balance'].iloc[i]==0)):
            Date =  datetime.strptime(CollateralCF['Date'].iloc[i], '%Y-%m-%d')
    return Date
"""
CollateralCF = Asset_CF
CurrentDateRow=j
"""
@jit
def LastInterestDate(CollateralCF, CurrentDateRow):
    Date = CollateralCF['Date'].iloc[CurrentDateRow]
    if (CurrentDateRow ==0):
        Date = LoanStartDate(CollateralCF)
    else:
        LastInterestDateRow = CurrentDateRow -1
        while ((CollateralCF['DateType'].iloc[LastInterestDateRow]!='IntDue') and (CollateralCF['Date'].iloc[LastInterestDateRow]!=LoanStartDate(CollateralCF))):
            LastInterestDateRow  = LastInterestDateRow -1
        Date = CollateralCF['Date'].iloc[LastInterestDateRow]
    return Date
"""        
Collateral_CashFlow = CashFlow_Collateral
Closing_Date = '2018-6-11'
RampUp_Date = '2018-8-6'
InterestStart_Date = '2018-7-23'
Frequency = 'Q'
"""
@jit
def CF_Translation_By_Asset(Collateral_CashFlow, Closing_Date, RampUp_Date, InterestStart_Date, Frequency):
    """All kinds of key dates"""
    EndDate = Collateral_CashFlow['Date'].max()
    ClosingDate = datetime.strptime(Closing_Date, '%Y-%m-%d')
    RampUpDate = datetime.strptime(RampUp_Date, '%Y-%m-%d')
    InterestStartDate = datetime.strptime(InterestStart_Date, '%Y-%m-%d')
    Freq = Frequency
    """Initialize scheduled cashflow dataframe"""
    DateList_Sch = pd.date_range(RampUpDate, EndDate + timedelta(days=180), freq=Freq)
    CF_Sch = pd.DataFrame(data=np.zeros((len(DateList_Sch)+1,7)),columns = ['CollectionDate','PaymentDate','Interest','Principal','Default','PrePayment','Balance'])
    CF_Sch['CollectionDate'][0] = ClosingDate
    CF_Sch['CollectionDate'][1:] = DateList_Sch
    for i in range(len(CF_Sch.index)):
        if (i==0):
            CF_Sch['PaymentDate'].iloc[i] = RampUpDate
        else:
            CF_Sch['PaymentDate'].iloc[i] = CF_Sch['CollectionDate'].iloc[i] + timedelta(days=17)
    
    Number_Of_Assets = int((len(Collateral_CashFlow.columns)-2)/3)
    
    for i in range(Number_Of_Assets):
        """Extract CF dataframe for each asset"""
        Asset_CF = pd.DataFrame(columns = ['DateType','Date','Interest','Principal','Balance'])
        Asset_CF['Date'] = Collateral_CashFlow['Date']
        Asset_CF['DateType'] = Collateral_CashFlow['DateType']
        Asset_CF['Interest'] = Collateral_CashFlow.iloc[:,3*(i+1)-1]
        Asset_CF['Principal'] = Collateral_CashFlow.iloc[:,3*(i+1)]
        
        """Translate"""
        for j in range(len(Asset_CF.index)):
            Row = DateRow(CF_Sch['CollectionDate'], Asset_CF['Date'].iloc[j])
            if (Asset_CF['Date'].iloc[j] > ClosingDate):
                CF_Sch['Principal'].iloc[Row] = CF_Sch['Principal'].iloc[Row] + Asset_CF['Principal'].iloc[j]
                if (Asset_CF['Date'].iloc[j] > InterestStartDate):
                    Last_Interest_Date = LastInterestDate(Asset_CF, j)
                    if (Row==1):
                        Portion = (Asset_CF['Date'].iloc[j]-InterestStartDate).days/(Asset_CF['Date'].iloc[j] - Last_Interest_Date).days
                        CF_Sch['Interest'].iloc[Row] = CF_Sch['Interest'].iloc[Row] + round(Asset_CF['Interest'].iloc[j]*Portion,2)
                    else:
                        CF_Sch['Interest'].iloc[Row] = CF_Sch['Interest'].iloc[Row]+  round(Asset_CF['Interest'].iloc[j],2)
    
    """Update balance"""
    CF_Sch['Balance'].iloc[0] = CF_Sch['Principal'].sum()
    for i in range(len(CF_Sch.index)):
        if(i!=0):
            CF_Sch['Balance'].iloc[i] = CF_Sch['Balance'].iloc[i-1] - CF_Sch['Principal'].iloc[i]
    CF_Sch = CF_Sch.fillna(0.0)
    return CF_Sch

@jit
def XNPV(Dates, Flows, Rate):
    NPV=0.0
    for i in range(len(Dates.index)):
        if (i==0):
            continue
        NPV=NPV+Flows[i]/((1+Rate)**((Dates[i]-Dates[0]).days/365))
    return NPV
"""
CF=CF_Sch
Bond_Info=BondInfo
ClassA_Info=ClassAInfo
ClassB_Info=ClassBInfo
ClassC_Info=ClassCInfo
Scenarios=Scenarios_Audit
InterestRateBenchMark=0.04
"""
@jit
def Risk_Award_Transfer(CF, Bond_Info, ClassA_Info, ClassB_Info, ClassC_Info, Scenarios, InterestRateBenchMark):
    Probability_Weighted_PV = pd.DataFrame(data=np.zeros((len(Scenarios.index),5)), columns = ['Scenario','Probability','Total','Transferred','Retained'])
    Probability_Weighted_PV['Scenario']=Scenarios['Scenario']
    Probability_Weighted_PV['Probability']=Scenarios['Probability']
    Probability_Weighted_SD = pd.DataFrame(data=np.zeros((len(Scenarios.index),5)), columns = ['Scenario','Probability','Total','Transferred','Retained'])
    Probability_Weighted_SD['Scenario']=Scenarios['Scenario']
    Probability_Weighted_SD['Probability']=Scenarios['Probability']
    """Present value"""
    for i in range(len(Scenarios.index)):
        CDR_Rate = Scenarios['CDR'][i]
        CPR_Rate = Scenarios['CPR'][i]
        CF_Stressed = CDR(CF, CDR_Rate, CPR_Rate).fillna(0.0)
        CF_Stressed['Interest'] = CF_Stressed['Interest']*(1+Scenarios['InterestRate'][i])
        Temp = Pass_Through_Waterfall(CF_Stressed, Bond_Info, ClassA_Info, ClassB_Info, ClassC_Info)
        WaterFall = Temp[4]
        Sub = Temp[3]
        Rate = InterestRateBenchMark + Scenarios['InterestRate'][i]
        """Total"""
        Probability_Weighted_PV['Total'][i]=XNPV(WaterFall['PaymentDate'], WaterFall['Int_Account_CF'], Rate)+XNPV(WaterFall['PaymentDate'], WaterFall['Prin_Account_CF'], Rate)
        
        """Retained"""
        Sub_Retained =  pd.DataFrame(data=Sub['Interest'])
        for j in range(len(Sub_Retained.index)):
            if (j==0):
                Sub_Retained.iloc[j]=0.0
            Sub_Retained.iloc[j] = min(WaterFall['Int_Sub'][j], Sub['Balance'].iloc[j-1]*0.005)
        CF_Retained = WaterFall['ServiceFee'] + WaterFall['Int_B'] + WaterFall['Prin_B'] + WaterFall['Int_C']+WaterFall['Prin_C'] #+ Sub_Retained['Interest']
        Probability_Weighted_PV['Retained'][i]=XNPV(WaterFall['PaymentDate'], CF_Retained, Rate)
        
        """Transferred"""
        Probability_Weighted_PV['Transferred'][i]=Probability_Weighted_PV['Total'][i]-Probability_Weighted_PV['Retained'][i]
    
    """Squared deviations"""
    Total_Mean = np.dot(Probability_Weighted_SD['Probability'],Probability_Weighted_PV['Total'])
    Transferred_Mean = np.dot(Probability_Weighted_SD['Probability'],Probability_Weighted_PV['Transferred'])
    Retained_Mean = np.dot(Probability_Weighted_SD['Probability'],Probability_Weighted_PV['Retained'])
    for i in range(len(Scenarios.index)):
        Probability_Weighted_SD['Total'][i]=Probability_Weighted_SD['Probability'][i]*((Probability_Weighted_PV['Total'][i]-Total_Mean)**2)
        Probability_Weighted_SD['Transferred'][i]=Probability_Weighted_SD['Probability'][i]*((Probability_Weighted_PV['Transferred'][i]-Transferred_Mean )**2)
        Probability_Weighted_SD['Retained'][i]=Probability_Weighted_SD['Probability'][i]*((Probability_Weighted_PV['Retained'][i]-Retained_Mean)**2)
    Risk_Retained_Delloite=round(0.5+(Probability_Weighted_SD['Retained'].sum()-Probability_Weighted_SD['Transferred'].sum())/(2*Probability_Weighted_SD['Total'].sum()),4)
    Risk_Retained_PWC=round(Probability_Weighted_SD['Retained'].sum()/Probability_Weighted_SD['Total'].sum(),4)
    return [Risk_Retained_Delloite,Risk_Retained_PWC, Probability_Weighted_PV,Probability_Weighted_SD]

@jit
def Data_Input(FileName):
    BondInfo = pd.read_excel(FileName, sheetname = 'Bond_Info', encoding = 'gb18030')
    ClassAInfo = pd.read_excel(FileName, sheetname = 'ClassA', encoding = 'gb18030')
    ClassBInfo = pd.read_excel(FileName, sheetname = 'ClassB', encoding = 'gb18030')
    ClassCInfo = pd.read_excel(FileName, sheetname = 'ClassC', encoding = 'gb18030')
    CashFlow_Collateral = pd.read_excel(FileName, sheetname = 'Collateral_CF', encoding = 'gb18030').fillna(0.0)
    Scenarios_Audit = pd.read_excel(FileName, sheetname = 'Risk_Award_Transfer', encoding = 'gb18030')
    return [BondInfo, ClassAInfo, ClassBInfo, ClassCInfo, CashFlow_Collateral, Scenarios_Audit]

@jit
def OneFactorCopula(Factor, Rho, Lambda):
    Epsilon = math.sqrt(Rho)*Factor + math.sqrt(1-Rho)*np.random.normal(0, 1)
    Probability = 1/(2*math.sqrt(2*3.14*1))*math.exp(0-Epsilon**2/2)
    Time_of_Default = 0 - 1/Lambda*math.log(Probability)
    Months_Survival = math.floor(Time_of_Default)
    return Months_Survival

@jit
def CF_Translation_By_Asset_Copula(Collateral_CashFlow, Closing_Date, RampUp_Date, InterestStart_Date, Frequency, Factor, Rho, Lambda):
    """All kinds of key dates"""
    EndDate = Collateral_CashFlow['Date'].max()
    ClosingDate = datetime.strptime(Closing_Date, '%Y-%m-%d')
    RampUpDate = datetime.strptime(RampUp_Date, '%Y-%m-%d')
    InterestStartDate = datetime.strptime(InterestStart_Date, '%Y-%m-%d')
    Freq = Frequency
    """Initialize scheduled cashflow dataframe"""
    DateList_Sch = pd.date_range(RampUpDate, EndDate + timedelta(days=180), freq=Freq)
    CF_Sch = pd.DataFrame(data=np.zeros((len(DateList_Sch)+1,7)),columns = ['CollectionDate','PaymentDate','Interest','Principal','Default','PrePayment', 'Balance'])
    CF_Sch['CollectionDate'][0] = ClosingDate
    CF_Sch['CollectionDate'][1:] = DateList_Sch
    for i in range(len(CF_Sch.index)):
        if (i==0):
            CF_Sch['PaymentDate'].iloc[i] = RampUpDate
        else:
            CF_Sch['PaymentDate'].iloc[i] = CF_Sch['CollectionDate'].iloc[i] + timedelta(days=17)
    
    Number_Of_Assets = int((len(Collateral_CashFlow.columns)-2)/3)
    
    for i in range(Number_Of_Assets):
        """Extract CF dataframe for each asset"""
        Asset_CF = pd.DataFrame(columns = ['DateType','Date','Interest','Principal','Balance'])
        Asset_CF['Date'] = Collateral_CashFlow['Date']
        Asset_CF['DateType'] = Collateral_CashFlow['DateType']
        Asset_CF['Interest'] = Collateral_CashFlow.iloc[:,3*(i+1)-1]
        Asset_CF['Principal'] = Collateral_CashFlow.iloc[:,3*(i+1)]
        
        FTD_Month = OneFactorCopula(Factor, Rho, Lambda)
        Last_Surviving_Date = RampUpDate + timedelta(days = FTD_Month*30)
        
        """Translate"""
        for j in range(len(Asset_CF.index)):
            Row = DateRow(CF_Sch['CollectionDate'], Asset_CF['Date'].iloc[j])
            if (Asset_CF['Date'].iloc[j] > ClosingDate):
                
                """Principal"""
                if(Asset_CF['Date'].iloc[j] <= Last_Surviving_Date):
                    CF_Sch['Principal'].iloc[Row] = CF_Sch['Principal'].iloc[Row] + Asset_CF['Principal'].iloc[j]
                else:
                    CF_Sch['Default'].iloc[Row] = CF_Sch['Default'].iloc[Row] + Asset_CF['Principal'].iloc[j]
                
                
                if ((Asset_CF['Date'].iloc[j] > InterestStartDate) & (Asset_CF['Date'].iloc[j] <= Last_Surviving_Date)):
                    Last_Interest_Date = LastInterestDate(Asset_CF, j)
                    if (Row==1):
                        Portion = (Asset_CF['Date'].iloc[j]-InterestStartDate).days/(Asset_CF['Date'].iloc[j] - Last_Interest_Date).days
                        CF_Sch['Interest'].iloc[Row] = CF_Sch['Interest'].iloc[Row] + round(Asset_CF['Interest'].iloc[j]*Portion,2)
                    else:
                        CF_Sch['Interest'].iloc[Row] = CF_Sch['Interest'].iloc[Row]+  round(Asset_CF['Interest'].iloc[j],2)
    """Update balance"""
    CF_Sch['Balance'].iloc[0] = CF_Sch['Principal'].sum() + CF_Sch['Default'].sum()
    for i in range(len(CF_Sch.index)):
        if(i!=0):
            CF_Sch['Balance'].iloc[i] = CF_Sch['Balance'].iloc[i-1] - CF_Sch['Principal'].iloc[i] - CF_Sch['Default'].iloc[i]
    CF_Sch = CF_Sch.fillna(0.0)
    return CF_Sch

@jit
def Standardized_SpreadSheets(FileName, Closing_Date, RampUp_Date, IntStart_Date, Frequency, BenchmarkRate):
    """Data Input""" 
    Info = Data_Input(FileName) #0 BondInfo, 1 ClassAInfo, 2 ClassBInfo, 3 ClassCInfo, 4 CashFlow_Collateral, 5 Scenarios_Audit
     
    """All kinds of key dates"""
    EndDate = Info[4]['Date'].max()    
    CF_Sch = CF_Translation_By_Asset(Info[4], Closing_Date, RampUp_Date, IntStart_Date, Frequency)
    """CF of CDO notes"""
    CF_Note_Sch = Pass_Through_Waterfall(CF_Sch, Info[0], Info[1], Info[2], Info[3])# 0 CF_Sch_A, 1 CF_Sch_B, 2 CF_Sch_C, 3 CF_Sch_Sub, 4 CF_Sch_WaterFall
    
    """Risk Award Transfer"""
    RiskAward = Risk_Award_Transfer(CF_Sch, Info[0], Info[1], Info[2], Info[3], Info[5], BenchmarkRate) # 0 Delloite, 1 PwC, 2 PV, 3 SD
    
    """Spread sheets export"""
    Writer_CF_Note = pd.ExcelWriter('CashFlow_Scheduled.xlsx', engine='xlsxwriter')
    CF_Note_Sch[4].to_excel(Writer_CF_Note, sheet_name = 'Waterfall')
    CF_Note_Sch[0].to_excel(Writer_CF_Note, sheet_name = 'Class-A')
    CF_Note_Sch[1].to_excel(Writer_CF_Note, sheet_name = 'Class-B')
    CF_Note_Sch[2].to_excel(Writer_CF_Note, sheet_name = 'Class-C')
    CF_Note_Sch[3].to_excel(Writer_CF_Note, sheet_name = 'Subordinate')
    RiskAward[3].to_excel(Writer_CF_Note, sheet_name = 'Prob_Wght_SD_RR=%s' %RiskAward[1])
    RiskAward[2].to_excel(Writer_CF_Note, sheet_name = 'Prob_Wght_PV_RR=%s' %RiskAward[1])
    Writer_CF_Note.save()
    
    """Scenarios"""
    CF_Scenario1 = CDR(CF_Sch, Info[5]['CDR'][0], Info[5]['CPR'][0]).fillna(0.0)
    CF_Scenario1['Interest']=CF_Scenario1['Interest']*(1+Info[5]['InterestRate'][0])
    CF_Note_Scenario1 = Pass_Through_Waterfall(CF_Scenario1, Info[0], Info[1], Info[2], Info[3])
    Writer_CF_Scenario1 = pd.ExcelWriter('CashFlow_Scenario1.xlsx', engine='xlsxwriter')
    CF_Note_Scenario1[4].to_excel(Writer_CF_Scenario1, sheet_name = 'Waterfall')
    CF_Note_Scenario1[0].to_excel(Writer_CF_Scenario1, sheet_name = 'Class-A')
    CF_Note_Scenario1[1].to_excel(Writer_CF_Scenario1, sheet_name = 'Class-B')
    CF_Note_Scenario1[2].to_excel(Writer_CF_Scenario1, sheet_name = 'Class-C')
    CF_Note_Scenario1[3].to_excel(Writer_CF_Scenario1, sheet_name = 'Subordinate')
    Writer_CF_Scenario1.save()
    
    CF_Scenario2 = CDR(CF_Sch, Info[5]['CDR'][1], Info[5]['CPR'][1]).fillna(0.0)
    CF_Scenario2['Interest']=CF_Scenario2['Interest']*(1+Info[5]['InterestRate'][1])
    CF_Note_Scenario2 = Pass_Through_Waterfall(CF_Scenario2, Info[0], Info[1], Info[2], Info[3])
    Writer_CF_Scenario2 = pd.ExcelWriter('CashFlow_Scenario2.xlsx', engine='xlsxwriter')
    CF_Note_Scenario2[4].to_excel(Writer_CF_Scenario2, sheet_name = 'Waterfall')
    CF_Note_Scenario2[0].to_excel(Writer_CF_Scenario2, sheet_name = 'Class-A')
    CF_Note_Scenario2[1].to_excel(Writer_CF_Scenario2, sheet_name = 'Class-B')
    CF_Note_Scenario2[2].to_excel(Writer_CF_Scenario2, sheet_name = 'Class-C')
    CF_Note_Scenario2[3].to_excel(Writer_CF_Scenario2, sheet_name = 'Subordinate')
    Writer_CF_Scenario2.save()
    
    return [CF_Sch, CF_Note_Sch, RiskAward[1]]
    
    