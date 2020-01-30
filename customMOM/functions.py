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
                ui.UIFunctionOutSingle(name='output_item', datatype=str,
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

