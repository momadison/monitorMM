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

class valueCountsValue(BaseTransformer):

    def __init__(self, input_items,  output_items):
        self.input_items = input_items
        self.output_items = output_items
        super().__init__()
    def execute(self, df):
        df = df.copy()
        for i, inputItem in enumerate(self.input_items):
            outputItem =  df[self.input_items].iloc[0:,0].value_counts(dropna=True, sort=True)
            outputItem = outputItem.index.tolist()
            print('this is my list: ', outputItem)
            df[self.output_items[i]] = pd.Series(outputItem)
        logger.info("value counts dataframe: ")
        logger.info(df)
        logger.info('output item indexes: ')
        logger.info(outputItem.index.tolist())
        logger.info('output items alone: ')
        logger.info(outputItem)

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

