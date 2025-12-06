"""Vault module for portfolio and vault management."""

from typing import Optional, Dict, Any, List, TYPE_CHECKING
from web3 import Web3
import httpx

from .types import (
    Portfolio,
    Asset,
    PortfolioMetrics,
    VaultInfo,
    TransactionReceipt,
)

if TYPE_CHECKING:
    from eth_account.signers.local import LocalAccount


# Contract ABIs (simplified for SDK)
VAULT_MANAGER_ABI = [
    {
        "name": "createVault",
        "type": "function",
        "inputs": [{"name": "strategy", "type": "uint8"}],
        "outputs": [{"name": "vaultId", "type": "bytes32"}],
    },
    {
        "name": "deposit",
        "type": "function",
        "inputs": [
            {"name": "asset", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "outputs": [],
    },
    {
        "name": "withdraw",
        "type": "function",
        "inputs": [
            {"name": "asset", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "outputs": [],
    },
]

ERC20_ABI = [
    {
        "name": "approve",
        "type": "function",
        "inputs": [
            {"name": "spender", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "outputs": [{"name": "", "type": "bool"}],
    },
    {
        "name": "balanceOf",
        "type": "function",
        "inputs": [{"name": "account", "type": "address"}],
        "outputs": [{"name": "", "type": "uint256"}],
    },
]


class VaultModule:
    """Module for vault and portfolio management."""

    # Contract addresses (configurable per network)
    VAULT_MANAGER_ADDRESS = "0x0000000000000000000000000000000000001001"

    def __init__(
        self,
        w3: Web3,
        account: Optional["LocalAccount"] = None,
        api_url: Optional[str] = None,
    ):
        """
        Initialize VaultModule.

        Args:
            w3: Web3 instance connected to Atlas chain.
            account: Optional account for signing transactions.
            api_url: API URL for portfolio queries.
        """
        self._w3 = w3
        self._account = account
        self._api_url = api_url
        self._http_client = httpx.Client(timeout=30.0)

    def _get_vault_manager(self):
        """Get VaultManager contract instance."""
        return self._w3.eth.contract(
            address=Web3.to_checksum_address(self.VAULT_MANAGER_ADDRESS),
            abi=VAULT_MANAGER_ABI,
        )

    def _sign_and_send(self, tx: Dict[str, Any]) -> TransactionReceipt:
        """Sign and send a transaction."""
        if not self._account:
            raise ValueError("Account required for transactions")

        # Build transaction
        tx["from"] = self._account.address
        tx["nonce"] = self._w3.eth.get_transaction_count(self._account.address)
        tx["gas"] = self._w3.eth.estimate_gas(tx)
        tx["gasPrice"] = self._w3.eth.gas_price

        # Sign and send
        signed = self._account.sign_transaction(tx)
        tx_hash = self._w3.eth.send_raw_transaction(signed.raw_transaction)

        # Wait for receipt
        receipt = self._w3.eth.wait_for_transaction_receipt(tx_hash)

        return TransactionReceipt(
            tx_hash=receipt["transactionHash"].hex(),
            block_number=receipt["blockNumber"],
            status=receipt["status"] == 1,
            gas_used=receipt["gasUsed"],
            effective_gas_price=receipt.get("effectiveGasPrice", 0),
            logs=[dict(log) for log in receipt["logs"]],
        )

    def create_vault(self, strategy: int = 0) -> TransactionReceipt:
        """
        Create a new vault.

        Args:
            strategy: Vault strategy type (0=conservative, 1=balanced, 2=aggressive).

        Returns:
            Transaction receipt.

        Raises:
            ValueError: If no account is configured.
        """
        if not self._account:
            raise ValueError("Account required to create vault")

        vault_manager = self._get_vault_manager()
        tx = vault_manager.functions.createVault(strategy).build_transaction({
            "chainId": self._w3.eth.chain_id,
        })

        return self._sign_and_send(tx)

    def deposit(self, asset: str, amount: int) -> TransactionReceipt:
        """
        Deposit assets into the vault.

        Args:
            asset: Token contract address.
            amount: Amount to deposit (in token's smallest unit).

        Returns:
            Transaction receipt.

        Raises:
            ValueError: If no account is configured.
        """
        if not self._account:
            raise ValueError("Account required for deposit")

        asset_address = Web3.to_checksum_address(asset)

        # First approve the vault manager to spend tokens
        token = self._w3.eth.contract(address=asset_address, abi=ERC20_ABI)
        approve_tx = token.functions.approve(
            Web3.to_checksum_address(self.VAULT_MANAGER_ADDRESS),
            amount,
        ).build_transaction({
            "chainId": self._w3.eth.chain_id,
        })

        self._sign_and_send(approve_tx)

        # Now deposit
        vault_manager = self._get_vault_manager()
        deposit_tx = vault_manager.functions.deposit(
            asset_address, amount
        ).build_transaction({
            "chainId": self._w3.eth.chain_id,
        })

        return self._sign_and_send(deposit_tx)

    def withdraw(self, asset: str, amount: int) -> TransactionReceipt:
        """
        Withdraw assets from the vault.

        Args:
            asset: Token contract address.
            amount: Amount to withdraw (in token's smallest unit).

        Returns:
            Transaction receipt.

        Raises:
            ValueError: If no account is configured.
        """
        if not self._account:
            raise ValueError("Account required for withdrawal")

        asset_address = Web3.to_checksum_address(asset)
        vault_manager = self._get_vault_manager()

        tx = vault_manager.functions.withdraw(
            asset_address, amount
        ).build_transaction({
            "chainId": self._w3.eth.chain_id,
        })

        return self._sign_and_send(tx)

    def get_portfolio(self, address: Optional[str] = None) -> Portfolio:
        """
        Get portfolio data for an address.

        Args:
            address: Address to query (defaults to connected account).

        Returns:
            Portfolio data.

        Raises:
            ValueError: If no address provided and no account configured.
        """
        if address is None:
            if not self._account:
                raise ValueError("Address required or account must be configured")
            address = self._account.address

        # Use JSON-RPC custom method
        result = self._w3.provider.make_request(
            "vdh_getPortfolio", [address]
        )

        if "error" in result:
            raise ValueError(result["error"]["message"])

        data = result["result"]

        assets = [
            Asset(
                symbol=a["symbol"],
                address=a["address"],
                balance=a["balance"],
                value=a["value"],
                allocation=a["allocation"],
            )
            for a in data.get("assets", [])
        ]

        metrics_data = data.get("metrics", {})
        metrics = PortfolioMetrics(
            volatility_30d=metrics_data.get("volatility30d", 0.0),
            sharpe_ratio=metrics_data.get("sharpeRatio", 0.0),
            unrealized_pnl=metrics_data.get("unrealizedPnL", 0.0),
            unrealized_pnl_percent=metrics_data.get("unrealizedPnLPercent", 0.0),
        )

        return Portfolio(
            address=data["address"],
            total_value=data["totalValue"],
            currency=data["currency"],
            assets=assets,
            metrics=metrics,
            last_updated=data["lastUpdated"],
        )

    def get_vault_info(self, vault_id: str) -> VaultInfo:
        """
        Get vault information.

        Args:
            vault_id: Vault identifier.

        Returns:
            VaultInfo with vault details.
        """
        result = self._w3.provider.make_request(
            "vdh_getVaultInfo", [vault_id]
        )

        if "error" in result:
            raise ValueError(result["error"]["message"])

        data = result["result"]

        assets = [
            Asset(
                symbol=a["symbol"],
                address=a["address"],
                balance=a["balance"],
                value=a["value"],
                allocation=a["allocation"],
            )
            for a in data.get("assets", [])
        ]

        return VaultInfo(
            vault_id=data["vaultId"],
            owner=data["owner"],
            assets=assets,
            total_value=data["totalValue"],
            strategy=data["strategy"],
            created_at=data["createdAt"],
        )

    def close(self):
        """Close HTTP client."""
        self._http_client.close()
