import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import math
from numba import jit


def Data_Input(FileName):
    ClassAInfo = pd.read_excel(FileName, sheetname = 'ClassA', encoding = 'gb18030')
    ClassBInfo = pd.read_excel(FileName, sheetname = 'ClassB', encoding = 'gb18030')
    ClassCInfo = pd.read_excel(FileName, sheetname = 'ClassC', encoding = 'gb18030')
    ClassSubInfo = pd.read_excel(FileName, sheetname = 'Subordinate', encoding = 'gb18030')
    FeeInfo = pd.read_excel(FileName, sheetname = 'Fees', encoding = 'gb18030')
    return [ClassAInfo, ClassBInfo, ClassCInfo, ClassSubInfo, FeeInfo]



def GenerateEmptyWaterFall(CF, Closing_Date, RampUp_Date, Freq, PaymentDate):
    WaterFall = pd.DataFrame(columns = ['CollectionDate','PaymentDate','Int_Account_CF','Prin_Account_CF','Tax','SPVFee','CustodianFee','InitialRatingFee','TrackingRatingFee','LegalFee','AdvisoryFee','AuditingFee','AccountantFee','ExchangeFee','Int_A','Int_B','Int_C','ServiceFee1','Int_Sub','Prin_A','Prin_B','Prin_C','ServiceFee2','ServiceFee2Owed','Prin_Sub','SubExptCompensation','Residual','OtherFee1','OtherFee2'])
    DateList = pd.date_range(RampUpDate, CF['CollectionDate'].max() + timedelta(days=60), freq=Freq)
    WaterFall['CollectionDate'] = [0 for i in range(len(DateList)+1)]
    WaterFall['CollectionDate'][0] = CF['CollectionDate'][0]
    WaterFall['CollectionDate'][1:] = DateList
    for i in range(len(WaterFall)):
        if (i==0):
            WaterFall['PaymentDate'].iloc[i] = RampUp_Date
        else:
            WaterFall['PaymentDate'].iloc[i] = WaterFall['CollectionDate'].iloc[i] + timedelta(days=PaymentDate)
    WaterFall = WaterFall.fillna(0.0)    
    for i in range(len(CF)):
        Row = (WaterFall['CollectionDate'] <= CF['CollectionDate'][i]).sum()
        WaterFall['Prin_Account_CF'][Row] += (CF['Principal'][i]+CF['PrePayment'][i])
        WaterFall['Int_Account_CF'][Row] += (CF['Fee'][i]+CF['Interest'][i])
    return WaterFall.fillna(0.0)


def ClearWaterFall(WaterFall):
    WaterFall.ix[:,5:] = 0.0
    return


@jit
def CLOPassThroughWaterfall(WaterFall, ClassA_Info, ClassB_Info, ClassC_Info, ClassSub_Info, Fee_Info):
    ClearWaterFall(WaterFall)
    """Cash Flow Dataframes Initializing"""
    Class_A = pd.DataFrame(columns = ['CollectionDate','PaymentDate','A1_Interest','A1_Principal','A1_Balance','A2_Interest','A2_Principal','A2_Balance','A3_Interest','A3_Principal','A3_Balance','A4_Interest','A4_Principal','A4_Balance','A5_Interest','A5_Principal','A5_Balance','A6_Interest','A6_Principal','A6_Balance'])
    Class_B = pd.DataFrame(columns = ['CollectionDate','PaymentDate','B1_Interest','B1_Principal','B1_Balance','B2_Interest','B2_Principal','B2_Balance','B3_Interest','B3_Principal','B3_Balance','B4_Interest','B4_Principal','B4_Balance','B5_Interest','B5_Principal','B5_Balance','B6_Interest','B6_Principal','B6_Balance'])
    Class_C = pd.DataFrame(columns = ['CollectionDate','PaymentDate','C1_Interest','C1_Principal','C1_Balance','C2_Interest','C2_Principal','C2_Balance','C3_Interest','C3_Principal','C3_Balance','C4_Interest','C4_Principal','C4_Balance','C5_Interest','C5_Principal','C5_Balance','C6_Interest','C6_Principal','C6_Balance'])
    Class_Sub = pd.DataFrame(columns = ['CollectionDate','PaymentDate','Interest','Principal','Balance','InterestOwed','ExptCompensation','Residual'])
    
    """Fixed Fee"""
    InitialRatingFee = float(Fee_Info['InitialRatingFee'][0])
    TrackingRatingFee = float(Fee_Info['TrackingRatingFee'][0])
    LegalFee = float(Fee_Info['LegalFee'][0])
    AdvisoryFee = float(Fee_Info['AdvisoryFee'][0])
    AuditingFee = float(Fee_Info['AuditingFee'][0])
    AccountantFee = float(Fee_Info['AccountantFee'][0])
    ExchangeFee = float(Fee_Info['ExchangeFee'][0])
    
    """Floating Fee Rate"""
    TaxRate = float(Fee_Info['TaxRate'][0])
    SPVFeeRate = float(Fee_Info['SPVFeeRate'][0])
    CustodianFeeRate = float(Fee_Info['CustodianFeeRate'][0])
    ServiceFee1Rate = float(Fee_Info['ServiceFee1Rate'][0])
    ServiceFee2Rate = float(Fee_Info['ServiceFee2Rate'][0])
    OtherFee1Rate = float(Fee_Info['OtherFee1Rate'][0])
    OtherFee2Rate = float(Fee_Info['OtherFee2Rate'][0])
    
    """Update dates for each CF dataframe"""
    Class_A['CollectionDate'] = Class_B['CollectionDate'] = Class_C['CollectionDate'] = Class_Sub['CollectionDate'] = WaterFall['CollectionDate']
    Class_A['PaymentDate'] = Class_B['PaymentDate'] = Class_C['PaymentDate'] = Class_Sub['PaymentDate'] = WaterFall['PaymentDate']
    
    """Fill in zeroes"""
    Class_A = Class_A.fillna(0.0)
    Class_B = Class_B.fillna(0.0)
    Class_C = Class_C.fillna(0.0)
    Class_Sub = Class_Sub.fillna(0.0)
    
    """Initialize Balance"""
    Class_Sub['Balance'][0] = ClassSub_Info['PrinAmnt_Sub'][0]
    for i in range(6):
        Class_A.iloc[0,3*(i+1)+1] = ClassA_Info.iloc[0,2*i]
        Class_B.iloc[0,3*(i+1)+1] = ClassB_Info.iloc[0,2*i]
        Class_C.iloc[0,3*(i+1)+1] = ClassC_Info.iloc[0,2*i]
    
    """Calculate the waterfall"""
    for i in range(1,len(WaterFall.index)):
        Balance_Begin = Class_A.iloc[i-1,[4,7,10,13,16,19]].sum()+Class_B.iloc[i-1,[4,7,10,13,16,19]].sum() + Class_C.iloc[i-1,[4,7,10,13,16,19]].sum() + Class_Sub.iloc[i-1,4]
        Days_Dist = float((WaterFall['PaymentDate'][i] - WaterFall['PaymentDate'][i-1]).days)
        
        Funds_for_Fees = WaterFall['Int_Account_CF'][i] + WaterFall['Prin_Account_CF'][i]
        
        "Floating fees before senior interest"
        WaterFall['Tax'][i] = WaterFall['Int_Account_CF'][i]*TaxRate
        WaterFall['SPVFee'][i] = Balance_Begin*SPVFeeRate*Days_Dist/365
        WaterFall['CustodianFee'][i] = Balance_Begin*CustodianFeeRate*Days_Dist/365
        WaterFall['ServiceFee1'][i] = Balance_Begin*ServiceFee1Rate*Days_Dist/365
        FloatingFees = WaterFall['Tax'][i] + WaterFall['SPVFee'][i] + WaterFall['CustodianFee'][i] + WaterFall['ServiceFee1'][i]
        "Fixed fees before senior interest"
        if (i==1):
            WaterFall['InitialRatingFee'][i] = InitialRatingFee
            WaterFall['LegalFee'][i] = LegalFee
            WaterFall['AuditingFee'][i] = AuditingFee
            WaterFall['AdvisoryFee'][i] = AdvisoryFee
            WaterFall['AccountantFee'][i] = AccountantFee
            WaterFall['ExchangeFee'][i] = ExchangeFee
        if ((WaterFall['PaymentDate'][i].month == 7)&(Balance_Begin!=0)):
            WaterFall['TrackingRatingFee'][i] = TrackingRatingFee
        FixedFees = WaterFall['InitialRatingFee'][i] + WaterFall['LegalFee'][i] + WaterFall['AdvisoryFee'][i] + WaterFall['AuditingFee'][i] + WaterFall['AccountantFee'][i] + WaterFall['ExchangeFee'][i] + WaterFall['TrackingRatingFee'][i]
        
        Funds_for_SeniorInt = Funds_for_Fees - FloatingFees - FixedFees
        "Senior Interests"
        for j in range(6):
            Class_A.iloc[i,3*(j+1)-1] = ClassA_Info.iloc[0,2*j+1]*Days_Dist/365*Class_A.iloc[i-1,3*(j+1)+1]
            Class_B.iloc[i,3*(j+1)-1] = ClassB_Info.iloc[0,2*j+1]*Days_Dist/365*Class_B.iloc[i-1,3*(j+1)+1]
            Class_C.iloc[i,3*(j+1)-1] = ClassC_Info.iloc[0,2*j+1]*Days_Dist/365*Class_C.iloc[i-1,3*(j+1)+1]
        WaterFall['Int_A'][i] =  Class_A.iloc[i,[2,5,8,11,14,17]].sum()
        WaterFall['Int_B'][i] =  Class_B.iloc[i,[2,5,8,11,14,17]].sum()
        WaterFall['Int_C'][i] =  Class_C.iloc[i,[2,5,8,11,14,17]].sum()
        SeniorInt_Sum = WaterFall['Int_A'][i] + WaterFall['Int_B'][i] + WaterFall['Int_C'][i]
        
        Funds_for_SubInt = max(0, WaterFall['Int_Account_CF'][i] - FixedFees - FloatingFees - SeniorInt_Sum)
        if (Balance_Begin!=0):
            Class_Sub['Interest'][i] = min(ClassSub_Info['IntRate_Sub'][0]*Days_Dist/365*Class_Sub['Balance'][i-1]*ClassSub_Info['Premium_Price'][0], Funds_for_SubInt)
        else:
            Class_Sub['Interest'][i] = 0.0
        
        Funds_for_SeniorPrin = Funds_for_SeniorInt - SeniorInt_Sum - Class_Sub['Interest'][i]
        
        """Class A Principals"""
        Funds_For_Prin_A = Funds_for_SeniorPrin
        if (Funds_For_Prin_A<0):
            print("Unsufficient cash flow in %s period" %(i))
        Funds_For_SubClass1 = max(0.0,Funds_For_Prin_A)
        Class_A['A1_Principal'][i] = min(Funds_For_SubClass1, Class_A['A1_Balance'][i-1])
        Class_A['A1_Balance'][i] = Class_A['A1_Balance'][i-1] - Class_A['A1_Principal'][i]
        Funds_For_SubClass2 = Funds_For_SubClass1 - Class_A['A1_Principal'][i]
        Class_A['A2_Principal'][i] = min(Funds_For_SubClass2, Class_A['A2_Balance'][i-1])
        Class_A['A2_Balance'][i] = Class_A['A2_Balance'][i-1] - Class_A['A2_Principal'][i]
        Funds_For_SubClass3 = Funds_For_SubClass2 - Class_A['A2_Principal'][i]
        Class_A['A3_Principal'][i] = min(Funds_For_SubClass3, Class_A['A3_Balance'][i-1])
        Class_A['A3_Balance'][i] = Class_A['A3_Balance'][i-1] - Class_A['A3_Principal'][i]
        Funds_For_SubClass4 = Funds_For_SubClass3 - Class_A['A3_Principal'][i]
        Class_A['A4_Principal'][i] = min(Funds_For_SubClass4, Class_A['A4_Balance'][i-1])
        Class_A['A4_Balance'][i] = Class_A['A4_Balance'][i-1] - Class_A['A4_Principal'][i]
        Funds_For_SubClass5 = Funds_For_SubClass4 - Class_A['A4_Principal'][i]
        Class_A['A5_Principal'][i] = min(Funds_For_SubClass5, Class_A['A5_Balance'][i-1])
        Class_A['A5_Balance'][i] = Class_A['A5_Balance'][i-1] - Class_A['A5_Principal'][i]
        Funds_For_SubClass6 = Funds_For_SubClass5 - Class_A['A5_Principal'][i]
        Class_A['A6_Principal'][i] = min(Funds_For_SubClass6, Class_A['A6_Balance'][i-1])
        Class_A['A6_Balance'][i] = Class_A['A6_Balance'][i-1] - Class_A['A6_Principal'][i]
        WaterFall['Prin_A'][i] = Class_A['A1_Principal'][i]+Class_A['A2_Principal'][i]+Class_A['A3_Principal'][i]+Class_A['A4_Principal'][i]+Class_A['A5_Principal'][i]+Class_A['A6_Principal'][i]
        
        """Class B Principals"""
        Funds_For_Prin_B = Funds_For_Prin_A -  WaterFall['Prin_A'][i]
        Funds_For_SubClass1 = Funds_For_Prin_B
        Class_B['B1_Principal'][i] = min(Funds_For_SubClass1, Class_B['B1_Balance'][i-1])
        Class_B['B1_Balance'][i] = Class_B['B1_Balance'][i-1] - Class_B['B1_Principal'][i]
        Funds_For_SubClass2 = Funds_For_SubClass1 - Class_B['B1_Principal'][i]
        Class_B['B2_Principal'][i] = min(Funds_For_SubClass2, Class_B['B2_Balance'][i-1])
        Class_B['B2_Balance'][i] = Class_B['B2_Balance'][i-1] - Class_B['B2_Principal'][i]
        Funds_For_SubClass3 = Funds_For_SubClass2 - Class_B['B2_Principal'][i]
        Class_B['B3_Principal'][i] = min(Funds_For_SubClass3, Class_B['B3_Balance'][i-1])
        Class_B['B3_Balance'][i] = Class_B['B3_Balance'][i-1] - Class_B['B3_Principal'][i]
        Funds_For_SubClass4 = Funds_For_SubClass3 - Class_B['B3_Principal'][i]
        Class_B['B4_Principal'][i] = min(Funds_For_SubClass4, Class_B['B4_Balance'][i-1])
        Class_B['B4_Balance'][i] = Class_B['B4_Balance'][i-1] - Class_B['B4_Principal'][i]
        Funds_For_SubClass5 = Funds_For_SubClass4 - Class_B['B4_Principal'][i]
        Class_B['B5_Principal'][i] = min(Funds_For_SubClass5, Class_B['B5_Balance'][i-1])
        Class_B['B5_Balance'][i] = Class_B['B5_Balance'][i-1] - Class_B['B5_Principal'][i]
        Funds_For_SubClass6 = Funds_For_SubClass5 - Class_B['B5_Principal'][i]
        Class_B['B6_Principal'][i] = min(Funds_For_SubClass6, Class_B['B6_Balance'][i-1])
        Class_B['B6_Balance'][i] = Class_B['B6_Balance'][i-1] - Class_B['B6_Principal'][i]
        WaterFall['Prin_B'][i] = Class_B['B1_Principal'][i]+Class_B['B2_Principal'][i]+Class_B['B3_Principal'][i]+Class_B['B4_Principal'][i]+Class_B['B5_Principal'][i]+Class_B['B6_Principal'][i]
        
        """Class C Principals"""
        Funds_For_Prin_C = Funds_For_Prin_B -  WaterFall['Prin_B'][i]
        Funds_For_SubClass1 = Funds_For_Prin_C
        Class_C['C1_Principal'][i] = min(Funds_For_SubClass1, Class_C['C1_Balance'][i-1])
        Class_C['C1_Balance'][i] = Class_C['C1_Balance'][i-1] - Class_C['C1_Principal'][i]
        Funds_For_SubClass2 = Funds_For_SubClass1 - Class_C['C1_Principal'][i]
        Class_C['C2_Principal'][i] = min(Funds_For_SubClass2, Class_C['C2_Balance'][i-1])
        Class_C['C2_Balance'][i] = Class_C['C2_Balance'][i-1] - Class_C['C2_Principal'][i]
        Funds_For_SubClass3 = Funds_For_SubClass2 - Class_C['C2_Principal'][i]
        Class_C['C3_Principal'][i] = min(Funds_For_SubClass3, Class_C['C3_Balance'][i-1])
        Class_C['C3_Balance'][i] = Class_C['C3_Balance'][i-1] - Class_C['C3_Principal'][i]
        Funds_For_SubClass4 = Funds_For_SubClass3 - Class_C['C3_Principal'][i]
        Class_C['C4_Principal'][i] = min(Funds_For_SubClass4, Class_C['C4_Balance'][i-1])
        Class_C['C4_Balance'][i] = Class_C['C4_Balance'][i-1] - Class_C['C4_Principal'][i]
        Funds_For_SubClass5 = Funds_For_SubClass4 - Class_C['C4_Principal'][i]
        Class_C['C5_Principal'][i] = min(Funds_For_SubClass5, Class_C['C5_Balance'][i-1])
        Class_C['C5_Balance'][i] = Class_C['C5_Balance'][i-1] - Class_C['C5_Principal'][i]
        Funds_For_SubClass6 = Funds_For_SubClass5 - Class_C['C5_Principal'][i]
        Class_C['C6_Principal'][i] = min(Funds_For_SubClass6, Class_C['C6_Balance'][i-1])
        Class_C['C6_Balance'][i] = Class_C['C6_Balance'][i-1] - Class_C['C6_Principal'][i]
        WaterFall['Prin_C'][i] = Class_C['C1_Principal'][i]+Class_C['C2_Principal'][i]+Class_C['C3_Principal'][i]+Class_C['C4_Principal'][i]+Class_C['C5_Principal'][i]+Class_C['C6_Principal'][i]
        
        Funds_for_ServiceFee2 = Funds_For_Prin_C - WaterFall['Prin_C'][i]
        """Second Service Fee"""
        WaterFall['ServiceFee2'][i] = min(Balance_Begin*ServiceFee2Rate*Days_Dist/365 + WaterFall['ServiceFee2Owed'][i-1], Funds_for_ServiceFee2)
        WaterFall['ServiceFee2Owed'][i] = WaterFall['ServiceFee2Owed'][i-1] + Balance_Begin*ServiceFee2Rate*Days_Dist/365 - WaterFall['ServiceFee2'][i]
        
        """Subordinate Principal"""
        Funds_For_Prin_Sub = Funds_for_ServiceFee2 - WaterFall['ServiceFee2'][i]
        Class_Sub['Principal'][i] = math.floor(min(Funds_For_Prin_Sub, Class_Sub['Balance'][i-1]))
        Class_Sub['Balance'][i] = Class_Sub['Balance'][i-1] - Class_Sub['Principal'][i]
        WaterFall['Prin_Sub'][i] = Class_Sub['Principal'][i]
        
        Funds_For_ExptCompensation = Funds_For_Prin_Sub - Class_Sub['Principal'][i]
        """Surbordinate expected compensation"""
        if (Class_Sub['Balance'][i]==0):
            Class_Sub['ExptCompensation'][i] = min(Funds_For_Prin_Sub, Class_Sub['InterestOwed'][i-1])
        WaterFall['SubExptCompensation'][i] = Class_Sub['ExptCompensation'][i]
        
        "Sub Int owed balance"
        Class_Sub['InterestOwed'][i] = Class_Sub['InterestOwed'][i-1] + ClassSub_Info['ExptRate_Sub'][0]*Days_Dist/365*Class_Sub['Balance'][i-1]*ClassSub_Info['Premium_Price'][0] - Class_Sub['Interest'][i] - Class_Sub['ExptCompensation'][i]
        
        "Residuals"
        Class_Sub['Residual'][i] = max(0.0, ClassSub_Info['Residual_Portion'][0]*(Funds_For_ExptCompensation - Class_Sub['ExptCompensation'][i]))
        
        WaterFall['Residual'][i] = max(0.0, (Funds_For_ExptCompensation - Class_Sub['ExptCompensation'][i]))
    
    return [Class_A, Class_B, Class_C, Class_Sub]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    