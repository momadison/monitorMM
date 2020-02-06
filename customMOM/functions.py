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

'''
class HelloWorld(BaseTransformer):
    
    #The docstring of the function will show as the function description in the UI.
   

    def __init__(self, name, greeting_col):
        # a function is expected to have at least one parameter that acts
        # as an input argument, e.g. "name" is an argument that represents the
        # name to be used in the greeting. It is an "input" as it is something
        # that the function needs to execute.

        # a function is expected to have at lease one parameter that describes
        # the output data items produced by the function, e.g. "greeting_col"
        # is the argument that asks what data item name should be used to
        # deliver the functions outputs

        # always create an instance variable with the same name as your arguments

        self.name = name
        self.greeting_col = greeting_col
        super().__init__()

        # do not place any business logic in the __init__ method  # all business logic goes into the execute() method or methods called by the  # execute() method

    def execute(self, df):
        # the execute() method accepts a dataframe as input and returns a dataframe as output
        # the output dataframe is expected to produce at least one new output column

        df[self.greeting_col] = 'Hello %s' % self.name

        # If the function has no new output data, output a status_flag instead
        # e.g. df[<self.output_col_arg>> = True

        return df

    @classmethod
    def build_ui(cls):
        # Your function will UI built automatically for configuring it
        # This method describes the contents of the dialog that will be built
        # Account for each argument - specifying it as a ui object in the "inputs" or "outputs" list

        inputs = [ui.UISingle(name='name', datatype=str, description='Name of person to greet')]
        outputs = [
            ui.UIFunctionOutSingle(name='greeting_col', datatype=str, description='Output item produced by function')]
        return (inputs, outputs)
'''

class monthlyRate(BaseTransformer):

    def __init__(self, input_items, output_items):
        self.input_items = input_items
        self.output_items = output_items
        super().__init__()

    def execute(self, df):
        sources_not_in_column = df.index.names
        print('sources not in column: ', sources_not_in_column)
        print('before reset: ', df)
        df.reset_index(inplace=True)
        print('after reset: ', df)
        df = df.copy()
        minDate = min(df['RCV_TIMESTAMP_UTC'])
        endDate = dt.datetime.utcnow()
        difference = (endDate - minDate)
        difference = difference / np.timedelta64(1, 'D')
        logger.info('Total Time of reporting: ')
        logger.info(difference)
        rate = (df[self.input_items]) * (30/difference)
        logger.info('Deployment Monthly Rate: ')
        logger.info(rate)

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
            datatype=float)
        )
        outputs = []
        return (inputs, outputs)

class firstOccurenceRelation(BaseTransformer):

    def __init__(self, input_items, condition, output_items):
        self.input_items = input_items
        self.output_items = output_items
        self.condition = condition
        super().__init__()

    def execute(self, df):
        df = df.copy()
        count = 0
        row = []
        indexKey = df[self.input_items[0]]
        input = df[self.input_items[1]]
        indexKey.reset_index(inplace=True, drop=True)
        input.reset_index(inplace=True, drop=True)
        indexKey = indexKey.drop_duplicates(keep="first")
        keyValues = indexKey.index.values

        '''
        for j, k in indexKey.iteritems():
            for i, x in input.iteritems():
                if (k == x):
                    row.append(i[0])
                    break
        '''
        for x in keyValues:
            if (input.iloc[x] == self.condition):
                count = count + 1

        for i, inputItem in enumerate(self.input_items):
            df[self.output_items[i-1]] = count

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


'''
    def _calc(self, df):
        sources_not_in_column = df.index.names
        df.reset_index(inplace=True)
        df_temp2 = df.copy()
        min_date = min(df[‘devicetimestamp’])
        df.drop(df.index[df[‘devicetimestamp’] == min_date], inplace = True)
        df[‘previous_utc2’] = df[‘devicetimestamp’].dt.date - dt.timedelta(days=1)
        df_temp2[‘previous_utc2’] = df_temp2[‘devicetimestamp’].dt.date
        df_temp2 = df_temp2.drop(columns=[“devicetimestamp”])
        df_temp2 = df_temp2.rename(columns={self.input_item: “PrevDailyUsage2
        "})
        df = df.merge(df_temp2, how=‘inner’, on = [‘previous_utc2’, ‘pipeline_system’])
        df[self.output_item] = df[self.input_item] - df[‘PrevDailyUsage2’]
        ‘’'
        today = dt.datetime.utcnow()
        prev_day = today - timedelta(days=1)
        prev_day_top_window = prev_day + timedelta(minutes=16)
        prev_day_bot_window = prev_day - timedelta(minutes=16)
        prev_day_values = df.loc[(df[‘rcv_timestamp_utc’] >= prev_day_bot_window) &
                                 (df[‘rcv_timestamp_utc’] <= prev_day_top_window)][self.input_item]
        prev_day_values_mean = prev_day_values.mean()
        df[self.output_item] = df[self.input_item] - prev_day_values_mean
        ‘’'
        df.set_index(keys=sources_not_in_column, inplace=True)
        return df
'''

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
        #count = 0
        #for i, x in df[self.input_items].iterrows():
        #    if (not(np.isnan(x[0]))):
        #        count = count + 1
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


