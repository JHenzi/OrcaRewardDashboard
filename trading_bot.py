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
# Imports for Solana key management
from solders.keypair import Keypair
from solders.pubkey import Pubkey
import base64
from solders.transaction import Transaction



#### Load .env ####
load_dotenv()

# Load .env variables, default to Jupiter Ultra API if not set
JUPITER_API_BASE_URL = os.getenv('JUPITER_API_BASE_URL', 'https://lite-api.jup.ag/ultra/v1/')
SOL_MINT = os.getenv('SOL_MINT', 'So11111111111111111111111111111111111111112')
USDC_MINT = os.getenv('USDC_MINT', 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v')
JUPITER_AMOUNT = os.getenv('JUPITER_AMOUNT', '1')

#### Load Keypair ####
def load_keypair_from_env():
    priv = os.getenv("SOL_PRIVATE_KEY")
    if not priv:
        raise ValueError("SOL_PRIVATE_KEY not set in .env")
    # Try to parse as JSON array (Phantom/Sollet export)
    try:
        priv_bytes = bytes(json.loads(priv))
        kp = Keypair.from_bytes(priv_bytes)
    except Exception:
        # Otherwise, assume base58 string
        kp = Keypair.from_base58_string(priv)
    return kp

#### Load Public Key ####
def load_pubkey_from_env():
    pub = os.getenv("SOL_PUBLIC_KEY")
    if not pub:
        raise ValueError("SOL_PUBLIC_KEY not set in .env")
    return Pubkey.from_string(pub)

#### Setup Logging ####
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
        self.kp = load_keypair_from_env()
        self.public_key = load_pubkey_from_env()

    def log_trade(self, order_response, execute_response, signed_transaction="", db_path="sol_trades.db"):
        """
        Log trade details to a SQLite database, including the /execute response.
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                request_id TEXT,
                quote_id TEXT,
                mode TEXT,
                swap_type TEXT,
                router TEXT,
                maker TEXT,
                input_mint TEXT,
                output_mint TEXT,
                in_amount TEXT,
                out_amount TEXT,
                swap_mode TEXT,
                slippage_bps INTEGER,
                price_impact_pct TEXT,
                fee_bps INTEGER,
                platform_fee_amount TEXT,
                platform_fee_bps INTEGER,
                in_usd_value REAL,
                out_usd_value REAL,
                swap_usd_value REAL,
                total_time_ms INTEGER,
                route_plan TEXT,
                transaction TEXT,
                signed_transaction TEXT,
                execute_response TEXT
            )
        """)
        cursor.execute("""
            INSERT INTO trades (
                timestamp, request_id, quote_id, mode, swap_type, router, maker,
                input_mint, output_mint, in_amount, out_amount, swap_mode, slippage_bps,
                price_impact_pct, fee_bps, platform_fee_amount, platform_fee_bps,
                in_usd_value, out_usd_value, swap_usd_value, total_time_ms, route_plan,
                transaction, signed_transaction, execute_response
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            time.strftime("%Y-%m-%d %H:%M:%S"),
            order_response.get("requestId"),
            order_response.get("quoteId"),
            order_response.get("mode"),
            order_response.get("swapType"),
            order_response.get("router"),
            order_response.get("maker"),
            order_response.get("inputMint"),
            order_response.get("outputMint"),
            order_response.get("inAmount"),
            order_response.get("outAmount"),
            order_response.get("swapMode"),
            order_response.get("slippageBps"),
            order_response.get("priceImpactPct"),
            order_response.get("feeBps"),
            order_response.get("platformFee", {}).get("amount"),
            order_response.get("platformFee", {}).get("feeBps"),
            order_response.get("inUsdValue"),
            order_response.get("outUsdValue"),
            order_response.get("swapUsdValue"),
            order_response.get("totalTime"),
            json.dumps(order_response.get("routePlan")),
            order_response.get("transaction"),
            signed_transaction,
            json.dumps(execute_response) if execute_response else None
        ))
        conn.commit()
        conn.close()

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
            data = response.json()
            # self.log_quote(data, None) # - The old way
            return data
        except Exception as e:
            logger.error(f"Exception in Get Order Request: {e}")
            # return None
        # response.raise_for_status()
        # return response.json()

    def get_balances(self, address):
        url = f"{self.base_url}balances/{address}"
        headers = {'Accept': 'application/json'}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            #return data
            # Extract SOL and USDC balances using mint addresses
            sol_balance = data.get("SOL", {}).get("uiAmount", 0.0)
            usdc_balance = data.get(self.usdc_address, {}).get("uiAmount", 0.0)
            return {"SOL": sol_balance, "USDC": usdc_balance}
        except Exception as e:
            logger.error(f"Error fetching balances for {address}: {e}")
            return None

    def sign_transaction(self, transaction_b64: str) -> str:
        """
        Signs a base64-encoded transaction string and returns a base64-encoded signed transaction.
        """
        # Decode base64 to bytes
        tx_bytes = base64.b64decode(transaction_b64)
        # Deserialize to Transaction object
        tx = Transaction.from_bytes(tx_bytes)
        # Extract recent_blockhash from the transaction
        recent_blockhash = tx.message.recent_blockhash
        # Sign with keypair and recent_blockhash
        tx.sign([self.kp], recent_blockhash)
        # Serialize and encode back to base64
        signed_tx_b64 = base64.b64encode(bytes(tx)).decode("utf-8")
        return signed_tx_b64

    def execute_order(self, request_id, transaction_b64):
        """
        Execute a swap order by posting the signed transaction and requestId to the Jupiter Ultra API.

        Args:
            request_id (str): The requestId from the quote/order response.
            transaction_b64 (str): The base64-encoded unsigned transaction from the quote/order response.

        Returns:
            dict: The JSON response from the API.
        """
        url = self.base_url + "execute"
        signed_transaction = self.sign_transaction(transaction_b64)
        if not request_id:
            logger.error("No requestId provided!")
            return None
        if not signed_transaction:
            logger.error("Signing transaction failed!")
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
        execute_response = response.json()
        # You should also pass the original order_response if you want to log it here
        self.log_trade({}, execute_response, signed_transaction)
        return execute_response