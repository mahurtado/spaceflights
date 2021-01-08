# Copyright 2018-2019 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
#     or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Dict, List
from kedro.io import MemoryDataSet

import numpy as np
import pandas as pd
from google.cloud import bigquery

def train_model(data: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    """Split data and train the linear regression model.
        Args: parameters: Parameters defined in parameters.yml 
    """
    client = bigquery.Client()
    query_job = client.query(
        """
        CREATE OR REPLACE MODEL
            `{}`
        OPTIONS(
            model_type='linear_reg',
            input_label_cols=['price'],
            data_split_method='random',
            subsample=0.2)
        AS SELECT
            engines,
            passenger_capacity,
            crew,
            d_check_complete,
            moon_clearance_complete,
            price
        FROM
            `{}`
        """.format(parameters["bq_model_name"],parameters["bq_master_table"])
        )       
    results = query_job.result()
    input_ml_eval_data = pd.DataFrame({'state': 'ready'}, index=[0])
    input_ml_eval = MemoryDataSet(data=input_ml_eval_data)
    return input_ml_eval

def evaluate_model(data: pd.DataFrame, parameters: Dict):
    """Calculate the coefficient of determination and log the result.
        Args: parameters: Parameters defined in parameters.yml 
    """
    client = bigquery.Client()
    query_job = client.query(
        """
        SELECT r2_score FROM
        ML.EVALUATE(MODEL `{}`, (
           SELECT
               engines,
               passenger_capacity,
               crew,
               d_check_complete,
               moon_clearance_complete,
               price
           FROM
           `{}`)
        )
        """.format(parameters["bq_model_name"],parameters["bq_master_table"])
        )
    results = query_job.result()
    logger = logging.getLogger(__name__)
    for row in results:
        logger.info("Model has a coefficient R^2 of %.3f.", row.r2_score)
        