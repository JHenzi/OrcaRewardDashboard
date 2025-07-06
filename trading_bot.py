import requests
import sqlite3
import json
import time
from datetime import timedelta
import logging
from dotenv import load_dotenv
import sys
import os
import random


# Load .env
load_dotenv()

# Load .env variables, default to Jupiter Ultra API if not set
JUPITER_API_BASE_URL = os.getenv('JUPITER_API_BASE_URL', 'https://lite-api.jup.ag/ultra/v1/')
SOL_MINT = os.getenv('SOL_MINT', 'So11111111111111111111111111111111111111112')
USDC_MINT = os.getenv('USDC_MINT', 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v')
JUPITER_AMOUNT = os.getenv('JUPITER_AMOUNT', '1')

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

############ Solana Trading Bot - Leveraging Jupiter Ultra API ##############
#                                                                           #
# Usage: This is to be leveraged by the sol_price_fetcher.py script         #
# When the system makes a prediction using the contextual bandit, we want to#
# execute a trade on Jupiter based on the prediction. i.e. we will swap     #
# SOL and USDC based on the recommendations                                 #
#                                                                           #
#############################################################################

class JupiterTradingBot:
    def __init__(self):
        self.base_url = JUPITER_API_BASE_URL
        self.sol_address = SOL_MINT
        self.usdc_address = USDC_MINT
        # self.amount = amount

    def get_order(self, amount, action):
        if action == "buy":
            in_curr = self.usdc_address
            out_curr = self.sol_address
        elif action == "sell":
            in_curr = self.sol_address
            out_curr = self.usdc_address
        else:
            logger.error("Invalid action. Must be 'buy' or 'sell'. But it was: " + action)
        header = {
            'Accept': 'application/json',
        }
        params = {
            'inputMint': in_curr,
            'outputMint': out_curr,
            'amount': amount
        }
        try:
            response = requests.get(self.base_url + "order", headers=header, params=params)
        except Exception as e:
            logger.error(f"Exception in Get Order Request: {e}")
            # return None
        response.raise_for_status()
        return response.json()

    def execute_order(self, amount, action):
        """
        Execute a swap order by posting the signed transaction and requestId to the Jupiter Ultra API.

        Required Parameters:
            signed_transaction (str): The base64-encoded signed transaction.
            request_id (str): The requestId from the quote/order response.

        Returns:
            dict: The JSON response from the API.
        """

        url = self.base_url + "execute"
        signed_transaction = "" # TODO - REPLACE WITH SIGNING CEREMONY
        order_response = self.get_order(amount, action)
        order_response = self.get_order(amount, action)
        request_id = order_response.get('requestId')
        if not request_id:
            logger.error("No requestId found in order response!")
            return None
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        payload = {
            "signedTransaction": signed_transaction,
            "requestId": request_id
        }
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()
