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

class dropDuplicatesMOM(BaseTransformer):

    def __init__(self, input_items, output_items):

        self.input_items = input_items
        self.output_items = output_items
        super().__init__()
        print('this is the input items stuff: ', input_items)
    def execute(self, df):
        df = df.copy()
        for i, inputItem in enumerate(self.input_items):
            df[self.output_items[i]] = df['speed'].drop_duplicates()
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