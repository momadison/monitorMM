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
        logger.info('first 3 columns: ')
        logger.info(df[df.columns[0:3]])
        logger.info('next 3 columns: ')
        logger.info(df[df.columns[3:6]])
        logger.info('next 3 columns: ')
        logger.info(df[df.columns[6:9]])
        logger.info('next 3 columns: ')
        logger.info(df[df.columns[9:12]])
        logger.info('next 3 columns: ')
        logger.info(df[df.columns[12:15]])
        logger.info('next 3 columns: ')
        logger.info(df[df.columns[15:18]])
        logger.info('next 3 columns: ')
        logger.info(df[df.columns[18:21]])
        for i, inputItem in enumerate(self.input_items):
            outputItem =  df[self.input_items].iloc[0:,0].value_counts()
            df[self.output_items[i]] = pd.DataFrame(outputItem.index.tolist())
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
                is_output_datatype_derived = False)
                      )
        outputs = []
        return (inputs,outputs)