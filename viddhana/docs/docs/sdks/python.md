---
sidebar_position: 2
title: Python SDK
---

# Python SDK

The official Python SDK for building on VIDDHANA.

## Installation

```bash
pip install viddhana-sdk
```

Or with poetry:

```bash
poetry add viddhana-sdk
```

## Requirements

- Python 3.9 or higher
- `web3.py` ^6.0.0

## Quick Start

```python
from viddhana import ViddhanaSDK

# Initialize the SDK
sdk = ViddhanaSDK(
    network="testnet",  # or "mainnet"
    private_key=os.environ.get("PRIVATE_KEY"),  # optional for read-only
)

# Get current block number
block_number = sdk.get_block_number()
print(f"Current block: {block_number}")
```

## Configuration

### Constructor Options

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `network` | `str` | Yes | `"mainnet"` or `"testnet"` |
| `private_key` | `str` | No | Private key for signing transactions |
| `rpc_url` | `str` | No | Custom RPC URL |
| `api_key` | `str` | No | API key for higher rate limits |

### Example Configuration

```python
sdk = ViddhanaSDK(
    network="mainnet",
    private_key=os.environ.get("PRIVATE_KEY"),
    api_key=os.environ.get("VIDDHANA_API_KEY"),
)
```

## Core Methods

### Reading Chain Data

```python
# Get block number
block_number = sdk.get_block_number()

# Get chain ID
chain_id = sdk.get_chain_id()

# Get balance (returns Decimal)
balance = sdk.get_balance("0xAddress")

# Get transaction
tx = sdk.get_transaction("0xTxHash")

# Get block
block = sdk.get_block("latest")
```

### Account Operations

```python
# Get nonce
nonce = sdk.get_nonce("0xAddress")

# Get transaction count
tx_count = sdk.get_transaction_count("0xAddress")

# Get account address from SDK
my_address = sdk.address
```

### Sending Transactions

```python
# Transfer VDH
tx_hash = sdk.transfer(
    to="0xRecipientAddress",
    amount="10",  # in VDH
)

# Wait for confirmation
receipt = sdk.wait_for_transaction(tx_hash)
print(f"Confirmed in block: {receipt['blockNumber']}")
```

### Contract Interaction

```python
# Read contract data
result = sdk.read_contract(
    address="0xContractAddress",
    abi=contract_abi,
    function_name="balanceOf",
    args=["0xAddress"],
)

# Write to contract
tx_hash = sdk.write_contract(
    address="0xContractAddress",
    abi=contract_abi,
    function_name="transfer",
    args=["0xRecipient", 1000000000000000000],
)

receipt = sdk.wait_for_transaction(tx_hash)
```

## VDH Token Methods

```python
# Get VDH balance
balance = sdk.get_vdh_balance("0xAddress")

# Transfer VDH
tx_hash = sdk.transfer_vdh("0xRecipient", "100")

# Approve spending
tx_hash = sdk.approve_vdh("0xSpender", "1000")
```

## Governance Methods

```python
# Delegate voting power
tx_hash = sdk.delegate("0xDelegateAddress")

# Get voting power
votes = sdk.get_votes("0xAddress")

# Get past votes at block
past_votes = sdk.get_past_votes("0xAddress", block_number)
```

## Async Support

The SDK supports async operations with `asyncio`:

```python
import asyncio
from viddhana import AsyncViddhanaSDK

async def main():
    sdk = AsyncViddhanaSDK(
        network="testnet",
        private_key=os.environ.get("PRIVATE_KEY"),
    )
    
    # Async operations
    block_number = await sdk.get_block_number()
    balance = await sdk.get_balance("0xAddress")
    
    # Concurrent operations
    results = await asyncio.gather(
        sdk.get_block("latest"),
        sdk.get_balance("0xAddress1"),
        sdk.get_balance("0xAddress2"),
    )
    
    print(results)

asyncio.run(main())
```

## WebSocket Subscriptions

```python
from viddhana import ViddhanaSDK

sdk = ViddhanaSDK(network="testnet")

# Subscribe to new blocks
def on_new_block(block):
    print(f"New block: {block['number']}")

subscription = sdk.subscribe_to_blocks(on_new_block)

# Subscribe to logs
def on_log(log):
    print(f"New log: {log}")

subscription = sdk.subscribe_to_logs(
    address="0xContractAddress",
    topics=[],
    callback=on_log,
)

# Unsubscribe
subscription.unsubscribe()
```

## Error Handling

```python
from viddhana import ViddhanaSDK, ViddhanaError, ErrorCodes

sdk = ViddhanaSDK(network="testnet", private_key="...")

try:
    tx_hash = sdk.transfer(to="0x...", amount="100")
except ViddhanaError as e:
    if e.code == ErrorCodes.INSUFFICIENT_FUNDS:
        print("Not enough VDH")
    elif e.code == ErrorCodes.NONCE_TOO_LOW:
        print("Nonce too low, retry")
    else:
        print(f"Error: {e.message}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Type Hints

The SDK includes full type hints for IDE support:

```python
from viddhana import ViddhanaSDK
from viddhana.types import Block, Transaction, TransactionReceipt, Log

sdk = ViddhanaSDK(network="testnet")

block: Block = sdk.get_block("latest")
tx: Transaction = sdk.get_transaction("0x...")
receipt: TransactionReceipt = sdk.wait_for_transaction("0x...")
```

## Best Practices

1. **Environment Variables** - Never hardcode private keys
2. **Error Handling** - Always wrap transactions in try/except
3. **Context Managers** - Use SDK as context manager for cleanup
4. **Connection Pooling** - Reuse SDK instances when possible
5. **Async for Scale** - Use `AsyncViddhanaSDK` for high-throughput apps

## Examples

### Full Transfer Example

```python
import os
from viddhana import ViddhanaSDK

def main():
    sdk = ViddhanaSDK(
        network="testnet",
        private_key=os.environ.get("PRIVATE_KEY"),
    )
    
    # Check balance
    balance = sdk.get_balance(sdk.address)
    print(f"Balance: {balance} VDH")
    
    # Transfer
    tx_hash = sdk.transfer(
        to="0xRecipientAddress",
        amount="1",
    )
    print(f"Transaction sent: {tx_hash}")
    
    # Wait for confirmation
    receipt = sdk.wait_for_transaction(tx_hash)
    print(f"Confirmed in block: {receipt['blockNumber']}")

if __name__ == "__main__":
    main()
```

### Contract Deployment Example

```python
from viddhana import ViddhanaSDK

def deploy_contract():
    sdk = ViddhanaSDK(
        network="testnet",
        private_key=os.environ.get("PRIVATE_KEY"),
    )
    
    # Deploy contract
    tx_hash = sdk.deploy_contract(
        abi=contract_abi,
        bytecode=contract_bytecode,
        constructor_args=["Constructor Arg"],
    )
    
    receipt = sdk.wait_for_transaction(tx_hash)
    contract_address = receipt["contractAddress"]
    
    print(f"Deployed to: {contract_address}")
    return contract_address
```

### Batch Operations Example

```python
import asyncio
from viddhana import AsyncViddhanaSDK

async def batch_operations():
    sdk = AsyncViddhanaSDK(
        network="testnet",
        private_key=os.environ.get("PRIVATE_KEY"),
    )
    
    addresses = [
        "0xAddress1",
        "0xAddress2",
        "0xAddress3",
    ]
    
    # Fetch all balances concurrently
    balances = await asyncio.gather(
        *[sdk.get_balance(addr) for addr in addresses]
    )
    
    for addr, balance in zip(addresses, balances):
        print(f"{addr}: {balance} VDH")

asyncio.run(batch_operations())
```

---

See also: [JavaScript SDK](/docs/sdks/javascript) | [API Reference](/docs/api-reference/overview)
