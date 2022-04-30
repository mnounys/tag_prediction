import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.ensemble import RandomForestClassifier

class AirLinePrediction:
    _dfAirLineInfo = pd.DataFrame()
    _mPredict = 2
    
    def __init__(self):
        lDirPath = os.path.dirname(__file__)
        lFilePathToInfo = os.path.join(lDirPath, "../data/dfForPredict.csv")
        lFilePathToPred = os.path.join(lDirPath, "../data/predict_rf.joblib")
        self.mListForModel = ['QUARTER', 'MONTH','DAY_OF_WEEK','CARRIER', 'ORIGIN_CITY_NAME', 'DEST_CITY_NAME','DEP_HOUR_GROUP', 'ARR_HOUR_GROUP', 'DISTANCE_GROUP']
        
        if (AirLinePrediction._dfAirLineInfo.empty) :        
            AirLinePrediction._dfAirLineInfo = pd.read_csv(lFilePathToInfo,sep=',')
            for col in self.mListForModel :
                AirLinePrediction._dfAirLineInfo[col] = AirLinePrediction._dfAirLineInfo[col].astype('object')
                
            AirLinePrediction._dfAirLineInfo['PRED_IS_DELAYED'] = -1
            
            AirLinePrediction._mPredict = joblib.load(lFilePathToPred)
            print("AirLinePrediction initialised")
        
    def getPrediction(self,tail_num,day,month):
        lRow = AirLinePrediction._dfAirLineInfo[ (AirLinePrediction._dfAirLineInfo.TAIL_NUM == tail_num) &
                                            (AirLinePrediction._dfAirLineInfo.DAY_OF_MONTH == day) &
                                            (AirLinePrediction._dfAirLineInfo.MONTH == month) ]
        
        if lRow.empty : 
            lMsg = "Unknown AirLine number, please use getAirlineInfo to get your plane information"
            return lMsg, False
        
        lNbRow = len(lRow)
        lNbCalculated = len( lRow[lRow.PRED_IS_DELAYED != -1])
        
        if lNbRow != lNbCalculated : 
            #Predict the delay only if not already predicted
            lRowDum = pd.get_dummies(lRow[self.mListForModel]).reindex(columns = AirLinePrediction._mPredict.feature_names_in_).fillna(0)
            lDelay = AirLinePrediction._mPredict.predict(lRowDum)
            AirLinePrediction._dfAirLineInfo.loc[lRow.index,"PRED_IS_DELAYED"] = lDelay
        
        return AirLinePrediction._dfAirLineInfo.loc[lRow.index,['TAIL_NUM','DAY_OF_MONTH','MONTH', 'ORIGIN_CITY_NAME',
                                                                'DEST_CITY_NAME', 'CRS_DEP_TIME', 'DEP_TIME' 
                                                                'CRS_ARR_TIME','ARR_TIME', "PRED_IS_DELAYED"]], True      
    
    def getAirLineInfo(self, uFilter = None):
        if uFilter == None : 
            return AirLinePrediction._dfAirLineInfo.head(20)
        else :
            return AirLinePrediction._dfAirLineInfo.query(uFilter)
    
    def getMetadata(self):
        return AirLinePrediction._dfAirLineInfo.columns
    
    

if __name__ == "__main__":
    #Unit Testing
    mv = AirLinePrediction()
    
    print("Test of getPrediction proc")
    res,ret = mv.getPrediction('N13995',18,7)
    print(res)
    
    print("\n")
    print("Test of getAirLineInfo")
    res = mv.getAirLineInfo()
    print(res)
    
    print("\n")
    print('Test of getAirLineInfo : TAIL_NUM == "N13995" & DAY_OF_MONTH == 18 & MONTH == 7')
    res = mv.getAirLineInfo('TAIL_NUM == "N13995" & DAY_OF_MONTH == 18 & MONTH == 7')
    print(res)
    
    print("\n")
    print("Test of getMetadata proc")
    res = mv.getMetadata()
    print(res)