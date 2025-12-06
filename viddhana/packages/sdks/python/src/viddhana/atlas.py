"""Atlas Chain module for blockchain interactions."""

from typing import Optional, TYPE_CHECKING
from web3 import Web3
from web3.types import Wei

from .types import ChainInfo, BlockInfo

if TYPE_CHECKING:
    from eth_account.signers.local import LocalAccount


class AtlasModule:
    """Module for interacting with the Atlas blockchain."""

    def __init__(self, w3: Web3, account: Optional["LocalAccount"] = None):
        """
        Initialize AtlasModule.

        Args:
            w3: Web3 instance connected to Atlas chain.
            account: Optional account for signing transactions.
        """
        self._w3 = w3
        self._account = account

    def get_chain_info(self) -> ChainInfo:
        """
        Get current chain information.

        Returns:
            ChainInfo with chain ID, block number, gas price, and network name.
        """
        chain_id = self._w3.eth.chain_id
        block_number = self._w3.eth.block_number
        gas_price = self._w3.eth.gas_price

        # Determine network name based on chain ID
        network_names = {
            13370: "viddhana-mainnet",
            13371: "viddhana-testnet",
        }
        network_name = network_names.get(chain_id, f"unknown-{chain_id}")

        return ChainInfo(
            chain_id=chain_id,
            block_number=block_number,
            gas_price=gas_price,
            network_name=network_name,
        )

    def get_balance(self, address: str) -> Wei:
        """
        Get the native token balance of an address.

        Args:
            address: The address to query.

        Returns:
            Balance in Wei.
        """
        checksum_address = Web3.to_checksum_address(address)
        return self._w3.eth.get_balance(checksum_address)

    def get_balance_ether(self, address: str) -> float:
        """
        Get the native token balance in Ether units.

        Args:
            address: The address to query.

        Returns:
            Balance in Ether (float).
        """
        balance_wei = self.get_balance(address)
        return float(Web3.from_wei(balance_wei, "ether"))

    def get_block(self, block_identifier: int | str = "latest") -> BlockInfo:
        """
        Get block information.

        Args:
            block_identifier: Block number or 'latest', 'pending', 'earliest'.

        Returns:
            BlockInfo with block details.
        """
        block = self._w3.eth.get_block(block_identifier)

        return BlockInfo(
            number=block["number"],
            hash=block["hash"].hex(),
            parent_hash=block["parentHash"].hex(),
            timestamp=block["timestamp"],
            transactions=[tx.hex() if isinstance(tx, bytes) else tx for tx in block["transactions"]],
            gas_used=block["gasUsed"],
            gas_limit=block["gasLimit"],
        )

    def get_transaction_count(self, address: str) -> int:
        """
        Get the transaction count (nonce) for an address.

        Args:
            address: The address to query.

        Returns:
            Transaction count.
        """
        checksum_address = Web3.to_checksum_address(address)
        return self._w3.eth.get_transaction_count(checksum_address)

    def get_gas_price(self) -> int:
        """
        Get current gas price.

        Returns:
            Gas price in Wei.
        """
        return self._w3.eth.gas_price

    def is_connected(self) -> bool:
        """
        Check if connected to the network.

        Returns:
            True if connected, False otherwise.
        """
        return self._w3.is_connected()
