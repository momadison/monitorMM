import inspect
import logging
import datetime as dt
import math
import string

from sqlalchemy.sql.sqltypes import TIMESTAMP,VARCHAR
import numpy as np
import pandas as pd

from iotfunctions.base import BaseTransformer
from iotfunctions import ui

logger = logging.getLogger(__name__)

# Specify the URL to your package here.
# This URL must be accessible via pip install

PACKAGE_URL = 'git+https://github.com/momadison/monitorMM@starter_package'

class monthlyRate(BaseTransformer):

    def __init__(self, input_items, output_items):
        self.input_items = input_items
        self.output_items = output_items
        super().__init__()

    def execute(self, df):
        sources_not_in_column = df.index.names
        df.reset_index(inplace=True)
        df = df.copy()
        minDate = min(df['RCV_TIMESTAMP_UTC'])
        endDate = dt.datetime.utcnow()
        difference = (endDate - minDate)
        difference = difference / np.timedelta64(1, 'D')
        rate = (df[self.input_items]) * (30/difference)

        for i, input_item in enumerate(self.input_items):
            df[self.output_items[i]] = rate

        df.set_index(keys=sources_not_in_column, inplace=True)
        return df

    @classmethod
    def build_ui(cls):

        inputs = []
        inputs.append(ui.UIMultiItem(
            name='input_items',
            datatype=float,
            description="Data items adjust",
            output_item='output_items',
            is_output_datatype_derived=True)
        )
        outputs = []
        return (inputs, outputs)

class highestIndexOccurenceMOM(BaseTransformer):

    def __init__(self, input_items, output_items):
        self.input_items = input_items
        self.output_items = output_items
        super().__init__()

    def execute(self, df):
        df = df.copy()
        outputItem = df[self.input_items].iloc[0:,0].value_counts().idxmax()
        logger.info('the winner is: ', )
        logger.info(outputItem)

        for i, input_item in enumerate(self.input_items):
            df[self.output_items[i]] = outputItem

        return df

    @classmethod
    def build_ui(cls):

        inputs = []
        inputs.append(ui.UIMultiItem(
            name='input_items',
            datatype=str,
            description="Data items adjust",
            output_item='output_items',
            is_output_datatype_derived=True)
        )
        outputs = []
        return (inputs, outputs)

class conditionCount(BaseTransformer):

    def __init__(self, input_items, condition, output_items):
        self.input_items = input_items
        self.output_items = output_items
        self.condition = condition
        super().__init__()

    def execute(self, df):
        df = df.copy()
        count = 0
        for i,x in df[self.input_items].iterrows():
            if (str(x[0]) == str(self.condition)):
                count = count + 1
        for i, input_item in enumerate(self.input_items):
                df[self.output_items[i]] = count

        logger.info('New dataframe: ')
        logger.info(df)
        return df


    @classmethod
    def build_ui(cls):
        inputs = []
        inputs.append(ui.UIMultiItem(
            name='input_items',
            datatype=str,
            description="Data items adjust",
            output_item='output_items',
            is_output_datatype_derived=True)
        )
        inputs.append(ui.UISingle(
            name='condition',
            datatype=str)
        )
        outputs = []
        return (inputs, outputs)

class conditionCountFloat(BaseTransformer):

    def __init__(self, input_items, condition, output_items):
        self.input_items = input_items
        self.output_items = output_items
        self.condition = condition
        super().__init__()

    def execute(self, df):
        df = df.copy()
        count = 0
        for i,x in df[self.input_items].iterrows():
            if (float(x[0]) == float(self.condition)):
                count = count + 1
        for i, input_item in enumerate(self.input_items):
                df[self.output_items[i]] = count

        logger.info('New dataframe: ')
        logger.info(df)
        return df


    @classmethod
    def build_ui(cls):
        inputs = []
        inputs.append(ui.UIMultiItem(
            name='input_items',
            datatype=float,
            description="Data items adjust",
            output_item='output_items',
            is_output_datatype_derived=True)
        )
        inputs.append(ui.UISingle(
            name='condition',
            datatype=float)
        )
        outputs = []
        return (inputs, outputs)

class conditionCountBool(BaseTransformer):

    def __init__(self, input_items, condition, output_items):
        self.input_items = input_items
        self.output_items = output_items
        self.condition = condition
        super().__init__()

    def execute(self, df):
        df = df.copy()
        count = 0
        for i,x in df[self.input_items].iterrows():
            logger.info('this is x: ')
            logger.info(x)
            if (x[0]==self.condition):
                count = count + 1

        for i, input_item in enumerate(self.input_items):
            df[self.output_items[i]] = count

        return df


    @classmethod
    def build_ui(cls):
        inputs = []
        inputs.append(ui.UIMultiItem(
            name='input_items',
            datatype=bool,
            description="Data items adjust",
            output_item='output_items',
            is_output_datatype_derived=False)
        )
        inputs.append(ui.UISingle(
            name='condition',
            datatype=float)
        )
        outputs = []
        return (inputs, outputs)

class conditionCountBool2(BaseTransformer):

    def __init__(self, input_items, condition, output_items):
        self.output_items = output_items
        self.condition = condition
        self.input_items = input_items
        super().__init__()

    def execute(self, df):
        df = df.copy()
        count = 0
        for i,x in df[self.input_items].iterrows():
            if (x[0]==self.condition):
                count = count + 1

        for i, input_item in enumerate(self.input_items):
            print('the input item is: ', input_item)
            df[self.output_items[i]] = count

        logger.info('New dataframe being return is: ')
        logger.info(df)
        return df


    @classmethod
    def build_ui(cls):
        inputs = []
        inputs.append(ui.UIMultiItem(
            name='input_items',
            datatype=bool,
            description="Data items adjust",
            output_item='output_items',
            is_output_datatype_derived=False)
        )
        inputs.append(ui.UISingle(
            name='condition',
            description='check for true',
            datatype=bool)
        )
        outputs = []
        return (inputs, outputs)

class firstOccurenceRelation(BaseTransformer):

    def __init__(self, input_items, input_items2, condition, output_items):
        self.input_items = input_items
        self.input_items2 = input_items2
        self.output_items = output_items
        self.condition = condition
        super().__init__()

    def execute(self, df):
        df = df.copy()
        count = 0
        row = []
        indexKey = df[self.input_items]
        input = df[self.input_items2]
        indexKey.reset_index(inplace=True, drop=True)
        input.reset_index(inplace=True, drop=True)
        indexKey = indexKey.drop_duplicates(keep="first")
        keyValues = indexKey.index.values

        for x in keyValues:
            if (input.iloc[x,0] == self.condition):
                count = count + 1

        for i, inputItem in enumerate(self.input_items):
            df[self.output_items[i]] = count

        return df

    @classmethod
    def build_ui(cls):
        inputs = []
        inputs.append(ui.UIMultiItem(
            name='input_items',
            datatype=str,
            description="Unique Device ID Column",
            output_item='output_items',
            is_output_datatype_derived=False)
        )
        inputs.append(ui.UIMultiItem(
            name='input_items2',
            description='Search for Condition Column',
            datatype=str
        ))
        inputs.append(ui.UISingle(
            name='condition',
            datatype=str)
        )
        outputs = []
        return (inputs, outputs)

class lastOccurenceRelationCountBool(BaseTransformer):

    def __init__(self, input_items, input_items2, condition, output_items):
        self.input_items = input_items
        self.input_items2 = input_items2
        self.output_items = output_items
        self.condition = condition
        super().__init__()

    def execute(self, df):
        df = df.copy()
        count = 0
        indexKey = df[self.input_items]
        boolInput = df[self.input_items2]
        indexKey.reset_index(inplace=True)
        boolInput.reset_index(inplace=True)
        indexKey.drop('id', axis=1, inplace=True)
        indexKey.drop('RCV_TIMESTAMP_UTC', axis=1, inplace=True)
        boolInput.drop('id', axis=1, inplace=True)
        boolInput.drop('RCV_TIMESTAMP_UTC', axis=1, inplace=True)
        indexKey = indexKey.drop_duplicates(keep="last")
        keyValues = indexKey.index.values
        condition = self.condition
        if (condition > 0):
            condition = True
        else:
            condition = False
        for x in keyValues:
            if (boolInput.iloc[x,0] == None):
                boolInput.iloc[x,0] = True
            if (boolInput.iloc[x,0] == condition):
                count = count + 1

        for i, inputItem in enumerate(self.input_items):
            df[self.output_items[i]] = count

        return df

    @classmethod
    def build_ui(cls):
        inputs = []
        inputs.append(ui.UIMultiItem(
            name='input_items',
            datatype=str,
            description="DeviceId Indicator",
            output_item='output_items',
            is_output_datatype_derived=True)
        )
        inputs.append(ui.UIMultiItem(
            name='input_items2',
            datatype=bool,
            description="Boolean Column to match condition")
        )
        inputs.append(ui.UISingle(
            name='condition',
            datatype=float)
        )
        outputs = []
        return (inputs, outputs)

class lastOccurenceRelationCountFloat(BaseTransformer):

    def __init__(self, input_items, input_items2, condition, output_items):
        self.input_items = input_items
        self.input_items2 = input_items2
        self.output_items = output_items
        self.condition = condition
        super().__init__()

    def execute(self, df):
        df = df.copy()
        count = 0
        indexKey = df[self.input_items]
        input = df[self.input_items2]
        indexKey.reset_index(inplace=True, drop=True)
        input.reset_index(inplace=True, drop=True)
        #indexKey.drop('id', axis=1, inplace=True)
        #indexKey.drop('RCV_TIMESTAMP_UTC', axis=1, inplace=True)
        #boolInput.drop('id', axis=1, inplace=True)
        #boolInput.drop('RCV_TIMESTAMP_UTC', axis=1, inplace=True)
        indexKey = indexKey.drop_duplicates(keep="last")
        keyValues = indexKey.index.values
        condition = self.condition

        for x in keyValues:

            if (input.iloc[x,0] == condition):
                count = count + 1

        for i, inputItem in enumerate(self.input_items):
            df[self.output_items[i]] = count

        return df

    @classmethod
    def build_ui(cls):
        inputs = []
        inputs.append(ui.UIMultiItem(
            name='input_items',
            datatype=str,
            description="DeviceId Indicator",
            output_item='output_items',
            is_output_datatype_derived=True)
        )
        inputs.append(ui.UIMultiItem(
            name='input_items2',
            datatype=float,
            description="Condition Match")
        )
        inputs.append(ui.UISingle(
            name='condition',
            datatype=float)
        )
        outputs = []
        return (inputs, outputs)


class valueCountsMM(BaseTransformer):

    def __init__(self, input_items, data_switch, output_items):
        self.input_items = input_items
        self.output_items = output_items
        self.data_switch = data_switch
        super().__init__()
    def execute(self, df):
        df = df.copy()
        outputItem =  df[self.input_items].iloc[0:,0].value_counts(dropna=True, sort=True)
        if (self.data_switch == 1):
            MyOutput = (outputItem.index.tolist())
        else:
            MyOutput = (outputItem.tolist())
        for x in range(len(MyOutput),len(df[self.input_items])):
            MyOutput.append(None)
        df[self.output_items] = pd.DataFrame(MyOutput,index=df.index)

        return df

    @classmethod
    def build_ui(cls):
        inputs = []
        inputs.append(ui.UIMultiItem(
                name = 'input_items',
                datatype=str,
                description = "Data items adjust",
                output_item = 'output_items',
                is_output_datatype_derived = False)
                      )
        inputs.append(ui.UISingle(
            name='data_switch',
            description = "Enter 1 for index 0 for value",
            datatype=int)
        )
        outputs = []

        return (inputs,outputs)

class dropDuplicatesMOM(BaseTransformer):

    def __init__(self, input_items, output_items):

        self.input_items = input_items
        self.output_items = output_items
        super().__init__()
    def execute(self, df):
        df = df.copy()
        for i, inputItem in enumerate(self.input_items):
            df[self.output_items[i]] = df[self.input_items].drop_duplicates()
        return df

    @classmethod
    def build_ui(cls):
        #define arguments that behave as function inputs
        inputs = []
        inputs.append(ui.UIMultiItem(
                name = 'input_items',
                datatype=str,
                description = "Data items adjust",
                output_item = 'output_items',
                is_output_datatype_derived = True)
                      )
        outputs = []
        return (inputs,outputs)

class multiplyByFactorMM(BaseTransformer):

    def __init__(self, input_items, factor, output_items):
        self.input_items = input_items
        self.output_items = output_items
        self.factor = float(factor)
        super().__init__()
    def execute(self, df):
        df = df.copy()
        for i,input_item in enumerate(self.input_items):
            df[self.output_items[i]] = df[input_item] * self.factor
        print('df after with multiply by factor: ', df[self.output_items])
        return df

    @classmethod
    def build_ui(cls):
        #define arguments that behave as function inputs
        inputs = []
        inputs.append(ui.UIMultiItem(
                name = 'input_items',
                datatype=float,
                description = "Data items adjust",
                output_item = 'output_items',
                is_output_datatype_derived = True)
                      )
        inputs.append(ui.UISingle(
                name = 'factor',
                datatype=float)
                      )
        outputs = []
        return (inputs,outputs)


class countMOM(BaseTransformer):

    def __init__(self, input_items, output_items):
        self.input_items = input_items
        self.output_items = output_items
        super().__init__()

    def execute(self, df):
        df = df.copy()
        for i, input_item in enumerate(self.input_items):
            df[self.output_items[i]] = int(len(df[self.input_items]))

        return df

    @classmethod
    def build_ui(cls):
        inputs = []
        inputs.append(ui.UIMultiItem(
            name='input_items',
            datatype=str,
            description="Data items adjust",
            output_item='output_items',
            is_output_datatype_derived=False)
        )
        outputs = []
        return (inputs, outputs)

class countNotNoneMOM(BaseTransformer):

    def __init__(self, input_items, output_items):
        self.input_items = input_items
        self.output_items = output_items
        super().__init__()

    def execute(self, df):
        df = df.copy()
        lengthOfFrame = len(df[self.input_items])
        nanInFrame = df[self.input_items].isnull().sum().sum()
        notNullInFrame = lengthOfFrame - nanInFrame
        for i, input_item in enumerate(self.input_items):
            df[self.output_items[i]] = notNullInFrame

        return df

    @classmethod
    def build_ui(cls):
        inputs = []
        inputs.append(ui.UIMultiItem(
            name='input_items',
            datatype=str,
            description="Data items adjust",
            output_item='output_items',
            is_output_datatype_derived=False)
        )
        outputs = []
        return (inputs, outputs)

class countMOM2(BaseTransformer):

    def __init__(self, input_items, output_items):
        self.input_items = input_items
        self.output_items = output_items
        super().__init__()

    def execute(self, df):
        df = df.copy()
        for i, input_item in enumerate(self.input_items):
                df[self.output_items[i]] = len(df[self.input_items])

        return df

    @classmethod
    def build_ui(cls):
        inputs = []
        inputs.append(ui.UIMultiItem(
            name='input_items',
            datatype=str,
            description="Data items adjust",
            output_item='output_items',
            is_output_datatype_derived=True)
        )
        outputs = []
        return (inputs, outputs)

class HazardCount(BaseTransformer):

        def __init__(self, input_items, output_items):
            self.output_items = output_items
            self.input_items = input_items
            super().__init__()

        def execute(self, df):
            df = df.copy()
            waterAlert = df['waterAlert']
            lowBattery = df['batteryLevel']
            online = df['isOnline']
            waterHazardCount = len(np.where(waterAlert == True)[0])
            lowBatteryCount = len(np.where(lowBattery == 0)[0])
            offlineCount = len(np.where(online == False)[0])
            count = waterHazardCount + lowBatteryCount + offlineCount
            for i, input_item in enumerate(self.input_items):
                df[self.output_items[i]] = count

            return df

        @classmethod
        def build_ui(cls):
            inputs = []
            inputs.append(ui.UIMultiItem(
                name='input_items',
                datatype=str,
                description="Unique ID Column",
                output_item='output_items',
                is_output_datatype_derived=False)
            )

            outputs = []
            return (inputs, outputs)

class HazardLifeCycle(BaseTransformer):

    def __init__(self, input_items, output_items):
        self.output_items = output_items
        self.input_items = input_items
        super().__init__()



    def execute(self, df):
        sources_not_in_column = df.index.names
        df.reset_index(inplace=True)
        df = df.copy()
        count = 0
        lifeCycle = []
        waterAlert = df['waterAlert']
        lowBattery = df['batteryLevel']
        online = df['isOnline']
        deviceId = df[self.input_items]
        timeSeries = df['RCV_TIMESTAMP_UTC']
        waterHazardArr = np.where(waterAlert == True)[0]
        lowBatteryArr = np.where(lowBattery == 0)[0]
        offlineArr = np.where(online == False)[0]

        for alert in waterHazardArr:
            trueTimeStamp = timeSeries[alert]
            falseTimeStamp = dt.datetime.utcnow()
            deviceMatchArr = np.where(deviceId == deviceId.iloc[alert][0])
            deviceMatchArr = [i for i in deviceMatchArr[0] if i > alert]
            for match in deviceMatchArr:
                if waterAlert[match] == False:
                    falseTimeStamp = timeSeries[match]
                    break
            lifeCycleTime = falseTimeStamp - trueTimeStamp
            lifeCycle.append(lifeCycleTime)

        for alert in lowBatteryArr:
            trueTimeStamp = timeSeries[alert]
            falseTimeStamp = dt.datetime.utcnow()
            deviceMatchArr = np.where(deviceId == deviceId.iloc[alert][0])
            deviceMatchArr = [i for i in deviceMatchArr[0] if i > alert]
            for match in deviceMatchArr:
                if lowBattery[match] > 0:
                    falseTimeStamp = timeSeries[match]
                    break
            lifeCycleTime = falseTimeStamp - trueTimeStamp
            lifeCycle.append(lifeCycleTime)

        for alert in offlineArr:
            trueTimeStamp = timeSeries[alert]
            falseTimeStamp = dt.datetime.utcnow()
            deviceMatchArr = np.where(deviceId == deviceId.iloc[alert][0])
            deviceMatchArr = [i for i in deviceMatchArr[0] if i > alert]
            for match in deviceMatchArr:
                if online[match] == True:
                    falseTimeStamp = timeSeries[match]
                    break
            lifeCycleTime = falseTimeStamp - trueTimeStamp
            lifeCycle.append(lifeCycleTime)

        if len(lifeCycle) == 0:
            lifecycle.append(dt.datetime.utcnow() - 1)
        averageLifeCycle = sum(lifeCycle, dt.timedelta(0)) / len(lifeCycle)
        result = (averageLifeCycle.days *24) + (averageLifeCycle.seconds//3600)
        for i, input_item in enumerate(self.input_items):
            df[self.output_items[i]] = result

        df.set_index(keys=sources_not_in_column, inplace=True)
        return df

    @classmethod
    def build_ui(cls):
        inputs = []
        inputs.append(ui.UIMultiItem(
            name='input_items',
            datatype=str,
            description="Unique ID Column",
            output_item='output_items',
            is_output_datatype_derived=False)
        )

        outputs = []
        return (inputs, outputs)

class HazardType(BaseTransformer):

    def __init__(self, input_items, output_items):
        self.output_items = output_items
        self.input_items = input_items
        super().__init__()

    def execute(self, df):
        df = df.copy()
        waterAlert = df['waterAlert']
        batteryLevel = df['batteryLevel']
        isOnline = df['isOnline']
        deviceId = df[self.input_items]
        hazardType_df = []
        waterHazard = ['',False]
        batteryHazard = ['',False]
        onlineHazard = ['',False]

        for i in range (len(waterAlert)):
            if waterAlert.iloc[i] == True:
                hazardType_df.append('Water Leak')
                waterHazard = [deviceId.iloc[i,0], True]
            elif batteryLevel.iloc[i] == 0:
                hazardType_df.append('Low Battery')
                batteryHazard = [deviceId.iloc[i,0], True]
            elif isOnline.iloc[i] == False:
                hazardType_df.append('Device Offline')
                onlineHazard = [deviceId.iloc[i,0], True]
            elif waterHazard[1] == False and batteryHazard[1] == False and onlineHazard[1] == False:
                hazardType_df.append('Device Initialization')
            elif waterHazard[1] == True and waterHazard[0] == deviceId.iloc[i,0]:
                hazardType_df.append('Water Alert Resolved')
                waterHazard = ['',False]
            elif batteryHazard[1] == True and batteryHazard[0] == deviceId.iloc[i,0]:
                hazardType_df.append('Battery Alert Resolved')
                batteryHazard = ['',False]
            elif onlineHazard[1] == True and onlineHazard[0] == deviceId.iloc[i,0]:
                hazardType_df.append('Offline Alert Resolved')
                onlineHazard = ['',False]
            else:
                hazardType_df.append('Device Initialization')

        df[self.output_items] = pd.DataFrame(hazardType_df, index=df.index)

        return df

    @classmethod
    def build_ui(cls):
        inputs = []
        inputs.append(ui.UIMultiItem(
            name='input_items',
            datatype=str,
            description="Unique ID Column",
            output_item='output_items',
            is_output_datatype_derived=False)
        )

        outputs = []
        return (inputs, outputs)






