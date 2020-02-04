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


class CountTrue(BaseTransformer):

        def __init__(self,input_item, output_item):
            self.input_item = input_item
            self.output_item = output_item
            super().__init__()

        def execute(self, df):
            df = df.copy()
            count = 0
            for i, inputItem in enumerate(self.input_item):
                if (inputItem == True):
                    count = count + 1
            df[self.output_item] = count
            return df

        @classmethod
        def build_ui(cls):

            inputs = [ui.UISingle(name='input_item', datatype=bool, description='series of booleans')]
            outputs = [
                ui.UIFunctionOutSingle(name='output_item', datatype=int,
                                       description='Output item produced by function')]
            return (inputs, outputs)


class monthlyRate(BaseTransformer):
    import numpy as np

    def __init__(self, input_items, output_items):
        self.input_items = input_items
        self.output_items = output_items
        super().__init__()

    def execute(self, df):
        df = df.copy()
        firstDate = df['RCV_TIMESTAMP_UTC'].iloc[0]
        lastDate = df['RCV_TIMESTAMP_UTC'].iloc[-1]
        difference = (lastDate - firstDate)
        difference = difference / np.timedelta64(1, 'D')
        logger.info('Total Time of reporting: ')
        logger.info(difference)
        rate = len(df[self.input_items])
        logger.info('Deployment Monthly Rate: ')
        logger.info(rate)
        d = {'Rate':[rate]}
        df[self.output_items] = pd.DataFrame(d)

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

class highestIndexOccurence(BaseTransformer):
    import numpy as np

    def __init__(self, input_items, output_items):
        self.input_items = input_items
        self.output_items = output_items
        super().__init__()

    def execute(self, df):
        df = df.copy()
        outputItem = df[self.input_items].iloc[0:,0].value_counts().idxmax()
        logger.info('the winner is: ', )
        logger.info(outputItem)
        d = {'winner':[outputItem]}
        df[self.output_items] = pd.DataFrame(d)

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
            logger.info('x is: ')
            logger.info(x[0])
            if (x[0] != None):
                if(x[0]==self.condition):
                    count = count + 1

        for i, input_item in enumerate(self.input_items):
                df[self.output_items[i]] = count
        logger.info('count is :')
        logger.info(count)
        logger.info('New dataframe: ')
        logger.info(df)
        return df


    @classmethod
    def build_ui(cls):
        inputs = []
        inputs.append(ui.UIMultiItem(
            name='input_items',
            datatype=bool,
            description="Data items adjust",
            output_item='output_items')
            #is_output_datatype_derived=True)
        )
        inputs.append(ui.UISingle(
            name='condition',
            datatype=float)
        )
        outputs = []
        outputs.append(
            ui.UIFunctionOutSingle(name='output_items', datatype=int, description='count of bool')
        )
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
        input = df[self.input_items[0]]
        indexKey = df[self.input_items[0]].drop_duplicates()
        for j, k in indexKey.iteritems():
            for i, x in input.iteritems():
                if (k == x):
                    row.append(i)
                    break

        for reference in row:
            if (df[self.input_items[1]][reference] == self.condition):
                count = count + 1

        d = {'count':[count]}
        df[self.output_items] = pd.DataFrame(d)

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

class lastOccurenceRelation(BaseTransformer):

    def __init__(self, input_items, condition, output_items):
        self.input_items = input_items
        self.output_items = output_items
        self.condition = condition
        super().__init__()

    def execute(self, df):
        df = df.copy()
        count = 0
        row = []
        input = df[self.input_items[0]]
        input = input.iloc[::-1]
        indexKey = df[self.input_items[0]].drop_duplicates()
        for j, k in indexKey.iteritems():
            for i, x in input.iteritems():
                if (k == x):
                    row.append(i)
                    break

        for reference in row:
            if (df[self.input_items[1]][reference] == self.condition):
                count = count + 1

        d = {'count':[count]}
        df[self.output_items] = pd.DataFrame(d)
        print('the row is :', row)
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

class firstRowCount(BaseTransformer):

    def __init__(self, input_items, condition, output_items):
        self.input_items = input_items
        self.output_items = output_items
        self.condition = condition
        super().__init__()

    def execute(self, df):
        df = df.copy()
        count = 0

        for i, inputItem in enumerate(self.input_item):
            if (inputItem == True):
                count = count + 1
        d = {'count':[count]}
        df[self.output_items] = pd.DataFrame(d)
        print('the row is :', row)
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

class valueCountsValue(BaseTransformer):

    def __init__(self, input_items,  output_items):
        self.input_items = input_items
        self.output_items = output_items
        super().__init__()
    def execute(self, df):
        df2 = df.copy()
        print()
        for i, inputItem in enumerate(self.input_items):
            outputItem =  df[self.input_items].iloc[0:,0].value_counts(dropna=True, sort=True)
        MyOutput = pd.DataFrame(outputItem.index.tolist())
        logger.info('Dataframe to start: \n')
        logger.info(df2.iloc[0])
        logger.info('Dataframe to input: \n')
        logger.info(MyOutput)
        df2[self.output_items] = MyOutput
        logger.info('Dataframe after: \n')
        logger.info('First row: \n')
        logger.info(df2.iloc[0])
        logger.info('myOutput: ')
        logger.info(myOutput)
        logger.info('type of df[input: ', type(df[self.input_items]))
        logger.info('type of myOutput: ', type(MyOutput))
        df[self.output_items] = df[self.input_items]
        #merge = [df, MyOutput]
        #myString = str(self.output_items)[1:-1]
        #print(myString)
        #df = pd.concat(merge, axis=1, sort=False)
        #df.rename(columns={0:myString}, inplace=True)

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
        outputs.append(
            ui.UIFunctionOutSingle(name='output_items', datatype=str, description = 'list of value_counts() values')
        )
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

class multiplyByFactorMOM(BaseTransformer):

    def __init__(self, input_items, factor, output_items):
        self.input_items = input_items
        self.output_items = output_items
        self.factor = float(factor)
        super().__init__()
    def execute(self, df):
        df = df.copy()
        for i,input_item in enumerate(self.input_items):
            df[self.output_items[i]] = df[input_item] * self.factor
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


