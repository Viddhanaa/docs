"""Main ViddhanaClient for interacting with the VIDDHANA network."""

from typing import Optional
from web3 import Web3
from eth_account import Account

from .types import NetworkConfig, Network
from .atlas import AtlasModule
from .vault import VaultModule
from .ai import AIModule
from .governance import GovernanceModule


# Network configurations
NETWORKS = {
    Network.MAINNET: NetworkConfig(
        chain_id=13370,
        rpc_url="https://rpc.viddhana.network",
        ws_url="wss://ws.viddhana.network",
        api_url="https://api.viddhana.network",
    ),
    Network.TESTNET: NetworkConfig(
        chain_id=13371,
        rpc_url="https://rpc.testnet.viddhana.network",
        ws_url="wss://ws.testnet.viddhana.network",
        api_url="https://api.testnet.viddhana.network",
    ),
}


class ViddhanaClient:
    """
    Main client for interacting with the VIDDHANA network.

    Provides access to all VIDDHANA services including:
    - Atlas Chain: Blockchain operations
    - Vault: Portfolio and vault management
    - AI: Prometheus AI predictions and optimizations
    - Governance: Proposals, voting, and delegation

    Example:
        >>> client = ViddhanaClient(
        ...     rpc_url="https://rpc.testnet.viddhana.network",
        ...     api_key="your-api-key",
        ...     private_key="0x..."
        ... )
        >>> await client.connect()
        >>> balance = client.atlas.get_balance(client.address)
        >>> prediction = client.ai.predict_price("BTC", horizon=7)
        >>> await client.disconnect()
    """

    def __init__(
        self,
        rpc_url: str,
        api_key: Optional[str] = None,
        private_key: Optional[str] = None,
        network: Network = Network.MAINNET,
    ):
        """
        Initialize ViddhanaClient.

        Args:
            rpc_url: RPC endpoint URL for the Atlas chain.
            api_key: Optional API key for authenticated API access.
            private_key: Optional private key for signing transactions.
            network: Network to use for API endpoints (mainnet or testnet).
        """
        self._rpc_url = rpc_url
        self._api_key = api_key
        self._private_key = private_key
        self._network_config = NETWORKS.get(network, NETWORKS[Network.MAINNET])

        # Web3 instance
        self._w3: Optional[Web3] = None

        # Account for signing
        self._account = None
        if private_key:
            self._account = Account.from_key(private_key)

        # Modules (initialized on connect)
        self._atlas: Optional[AtlasModule] = None
        self._vault: Optional[VaultModule] = None
        self._ai: Optional[AIModule] = None
        self._governance: Optional[GovernanceModule] = None

        # Connection state
        self._connected = False

    @property
    def address(self) -> Optional[str]:
        """Get the connected account address."""
        return self._account.address if self._account else None

    @property
    def atlas(self) -> AtlasModule:
        """Get the Atlas chain module."""
        if self._atlas is None:
            raise RuntimeError("Client not connected. Call connect() first.")
        return self._atlas

    @property
    def vault(self) -> VaultModule:
        """Get the Vault module."""
        if self._vault is None:
            raise RuntimeError("Client not connected. Call connect() first.")
        return self._vault

    @property
    def ai(self) -> AIModule:
        """Get the AI module."""
        if self._ai is None:
            raise RuntimeError("Client not connected. Call connect() first.")
        return self._ai

    @property
    def governance(self) -> GovernanceModule:
        """Get the Governance module."""
        if self._governance is None:
            raise RuntimeError("Client not connected. Call connect() first.")
        return self._governance

    def connect(self) -> "ViddhanaClient":
        """
        Connect to the VIDDHANA network.

        Initializes Web3 connection and all modules.

        Returns:
            Self for method chaining.

        Raises:
            ConnectionError: If unable to connect to the RPC endpoint.
        """
        if self._connected:
            return self

        # Initialize Web3
        self._w3 = Web3(Web3.HTTPProvider(self._rpc_url))

        if not self._w3.is_connected():
            raise ConnectionError(f"Failed to connect to {self._rpc_url}")

        # Initialize modules
        self._atlas = AtlasModule(self._w3, self._account)
        self._vault = VaultModule(self._w3, self._account, self._network_config.api_url)
        self._ai = AIModule(self._network_config.api_url, self._api_key)
        self._governance = GovernanceModule(self._w3, self._account)

        self._connected = True
        return self

    def disconnect(self) -> None:
        """
        Disconnect from the VIDDHANA network.

        Closes all HTTP clients and cleans up resources.
        """
        if not self._connected:
            return

        # Close module resources
        if self._vault:
            self._vault.close()
        if self._ai:
            self._ai.close()

        # Reset state
        self._w3 = None
        self._atlas = None
        self._vault = None
        self._ai = None
        self._governance = None
        self._connected = False

    async def aconnect(self) -> "ViddhanaClient":
        """
        Async version of connect.

        Returns:
            Self for method chaining.
        """
        return self.connect()

    async def adisconnect(self) -> None:
        """
        Async version of disconnect.

        Also closes async HTTP clients.
        """
        if self._ai:
            await self._ai.aclose()
        self.disconnect()

    def is_connected(self) -> bool:
        """
        Check if connected to the network.

        Returns:
            True if connected, False otherwise.
        """
        if not self._connected or not self._w3:
            return False
        return self._w3.is_connected()

    def __enter__(self) -> "ViddhanaClient":
        """Context manager entry."""
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.disconnect()

    async def __aenter__(self) -> "ViddhanaClient":
        """Async context manager entry."""
        return await self.aconnect()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.adisconnect()

    def __repr__(self) -> str:
        """String representation."""
        status = "connected" if self._connected else "disconnected"
        addr = self.address or "no-account"
        return f"ViddhanaClient(rpc={self._rpc_url}, address={addr}, status={status})"
