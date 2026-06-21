import os 
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.eception import CustomException
from src.logger import logging

from src.components.Data_transformer import DataTransformation
@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join('artifacts',"train.csv")
    test_data_path=os.path.join('artifacts',"test.csv")
    raw_data_path=os.path.join('artifacts',"raw.csv")
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info("entered the data ingestion component")
        try:
            df=pd.read_csv('src/notebook/data/stud.csv')
            logging.info('read the data set')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)            
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info('split the train test data')
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('data ingestion is completed')
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data= obj.initiate_data_ingestion()
    data_transformation=DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)
            
    