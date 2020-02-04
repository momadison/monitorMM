import datetime as dt
import json
import pandas as pd
import numpy as np
import logging
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, func
from iotfunctions.db import Database
from iotfunctions.enginelog import EngineLogging

EngineLogging.configure_console_logging(logging.DEBUG)

'''
You can test functions locally before registering them on the server to
understand how they work.

Supply credentials by pasting them from the usage section into the UI.
Place your credentials in a separate file that you don't check into the repo. 



with open('credentials_as_dev.json', encoding='utf-8') as F:
    credentials = json.loads(F.read())
db_schema = None
db = Database(credentials=credentials)

'''

with open('credentials_as.json', encoding='utf-8') as F:
    credentials = json.loads(F.read())
db_schema = None
db = Database(credentials=credentials)

'''
Import and instantiate the functions to be tested 

The local test will generate data instead of using server data.
By default it will assume that the input data items are numeric.

Required data items will be inferred from the function inputs.

The function below executes an expression involving a column called x1
The local test function will generate data dataframe containing the column x1

By default test results are written to a file named df_test_entity_for_<function_name>
This file will be written to the working directory.

from customMOM.functions import HelloWorld
fn = HelloWorld(
        name = 'AS_Tester',
        greeting_col = 'greeting')
fn.execute_local_test(db=db,db_schema=db_schema)



from customMOM.logDataFrameMOM import logDataFramMOM

fn = logDataFramMOM(
    input_items=['speed', 'travel_time'],
    factor='2',
    output_items=['adjusted_speed', 'adjusted_travel_time']
)
df = fn.execute_local_test(db=db, db_schema=db_schema, generate_day=1, to_csv=True)
print(df)



from customMOM.countLabelsMOM import dropDuplicatesMOM


d = {'id': ['TestdeviceWhiBatterycritical','TestdeviceWhiBatteryLow','TestdeviceWhiNormal','TestdeviceWhiOffline','TestdeviceWhiWaterleak','TestdeviceWhiNormal','TestdeviceWhiOffline','TestdeviceWhiWaterleak'],
     'RCV_TIMESTAMP_UTC': ['2020-01-27 16:24:23.048414','2020-01-27 16:27:23.048414','2020-01-27 16:25:23.048414','2020-01-27 16:25:23.048414', '2020-01-21 10:50:36.604','2020-01-27 16:25:23.048414','2020-01-27 16:25:23.048414', '2020-01-21 10:50:36.604'],
     'alertEmail': ['mattomadison@gmail.com', 'momadison@me.com', 'momadison@gmail.com','mattomadison@me.com', 'momadison@me.com', 'momadison@gmail.com','mattomadison@me.com','mattomadison@me.com'],
     'alertPhoneNumber': ['9588473456','9847323748', '9048732312','9723450976','9847323748', '9048732312','9723450976','9723450976'],
     'appliance': ['washer','washer','washer','washer','washer','washer','washer','washer'],
     'batteryLevel': [0,1,2,1,0,2,1,0],
     'waterAlert': [True,False,True,False,True,True,False,True],
     'policyId': ['testpolicybatterycritical','testpolicywaterleak','testpolicyoffline','testpolicynormal','whitestpolicybatterylow','testpolicywaterleak','testpolicyoffline','testpolicynormal'],
     'manufacturerDeviceId': ['TestdeviceWhiBatterycritical','TestdeviceWhiBatterylow','TestdeviceWhiNormal','TestdeviceWhiOffline','TestdeviceWhiWaterleak','TestdeviceWhiNormal','TestdeviceWhiOffline','TestdeviceWhiWaterleak'],
     'descriptiveLocation': ['Bathroom', 'Bathroom', 'Kitchen', 'Bathroom', 'Basement', 'Kitchen', 'Bathroom', 'Basement'],
     'country': ['United States','United States','United States','United States','United States','United States','United States','United States'],
     'state': ['Texas','Texas','Texas','Texas','Texas','Texas','Texas','Texas'],
     }
df = pd.DataFrame(data=d)

fn = dropDuplicatesMOM(
    input_items=['policyId'],
    output_items=['myList']
)
print('this is a dataframe: ', df)
#df = fn.execute_local_test(db=db, db_schema=db_schema, generate_day=1, to_csv=True)
df = fn.execute(df)
print(df)

'''

from customMOM.functions import monthlyRate

d = {'id': ['TestdeviceWhiOffline','TestdeviceWhiOffline','TestdeviceWhiBatterycritical','TestdeviceWhiBatteryLow','TestdeviceWhiNormal','TestdeviceWhiWaterleak','TestdeviceWhiNormal','TestdeviceWhiWaterleak'],
     'RCV_TIMESTAMP_UTC': [pd.to_datetime('2020-01-21 10:50:36.604000'),pd.to_datetime('2020-01-21 10:50:44.524000'),pd.to_datetime('2020-01-27 09:53:04.067000'),pd.to_datetime(' 2020-01-27 09:53:10.130000'),pd.to_datetime(' 2020-01-27 09:53:10.130000'),pd.to_datetime(' 2020-01-27 09:53:10.130000'),pd.to_datetime(' 2020-01-27 09:53:10.130000'),pd.to_datetime('2020-01-27 09:53:10.130000')],
     'alertEmail': ['mattomadison@gmail.com', 'momadison@me.com', 'momadison@gmail.com','mattomadison@me.com', 'momadison@me.com', 'momadison@gmail.com','mattomadison@me.com','mattomadison@me.com'],
     'alertPhoneNumber': ['9588473456','9847323748', '9048732312','9723450976','9847323748', '9048732312','9723450976','9723450976'],
     'appliance': ['washer','washer','washer','washer','washer','washer','washer','washer'],
     'batteryLevel': [0,1,2,1,0,2,1,0],
     'waterAlert': [True,False,True,False,True,True,False,True],
     'policyId': ['testpolicybatterycritical','testpolicywaterleak','testpolicyoffline','testpolicynormal','whitestpolicybatterylow','testpolicywaterleak','testpolicyoffline','testpolicynormal'],
     'manufacturerDeviceId': ['TestdeviceWhiBatterycritical','TestdeviceWhiBatterylow','TestdeviceWhiNormal','TestdeviceWhiOffline','TestdeviceWhiWaterleak','TestdeviceWhiNormal','TestdeviceWhiOffline','TestdeviceWhiWaterleak'],
     'descriptiveLocation': ['Bathroom', 'Bathroom', 'Kitchen', 'Bathroom', 'Basement', 'Kitchen', 'Bathroom', 'Basement'],
     'country': ['United States','United States','United States','United States','United States','United States','United States','United States'],
     'state': ['Texas','Texas','Texas','Texas','Texas','Texas','Texas','Texas'],
     'zone': ['east coast', 'east coast', 'south', 'south', 'west coast', 'south', 'south', 'south'],
     'hazard1': [9,None,None,None,None,None,None,None],
     'hazard2': [6,'NaN','NaN','NaN','NaN','NaN','NaN','NaN'],
     'hazard3': [4,'NaN','NaN','NaN','NaN','NaN','NaN','NaN'],
     'waterAlert2': [1.000,0.000,None,1.000,0.000,None,1.000,0.000],
     'deploymentCount': [12,12,12,12,12,12,12,12]
     }
df = pd.DataFrame(data=d)

fn = monthlyRate(
     input_items=['deploymentCount'],
     output_items=['new_column'],
     #condition=9
)

#df = fn.execute_local_test(db=db, db_schema=db_schema, generate_day=1, to_csv=True)
df = fn.execute(df)
print(df)








