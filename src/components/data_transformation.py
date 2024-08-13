# Feature engineering + data cleaning

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer # 在数据集中处理数值特征和分类特征时，分别进行不同的处理
from sklearn.impute import SimpleImputer # 用于处理缺失值
from sklearn.pipeline import Pipeline # 将多个数据处理步骤串联在一起，形成一个工作流
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

# 获取 input 的 path 路径
@dataclass # decorator 让我们可以直接在下面 define class variables
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is for Data Transformation
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            
            # 数值的 pipeline, handle missing values + perform standard scaler
            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler(with_mean=False))

                ]
            )
            
            # 类别的 pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            
            # 
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train test data completed")
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = "math_score" # 目标列
            numerical_columns = ["writing_score", "reading_score"]
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")
            
            # 
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)