"""Governance module for proposals, voting, and delegation."""

from typing import Optional, Dict, Any, List, TYPE_CHECKING
from web3 import Web3

from .types import Proposal, Vote, Delegation, TransactionReceipt

if TYPE_CHECKING:
    from eth_account.signers.local import LocalAccount


# Governance contract ABI (simplified)
GOVERNOR_ABI = [
    {
        "name": "propose",
        "type": "function",
        "inputs": [
            {"name": "targets", "type": "address[]"},
            {"name": "values", "type": "uint256[]"},
            {"name": "calldatas", "type": "bytes[]"},
            {"name": "description", "type": "string"},
        ],
        "outputs": [{"name": "proposalId", "type": "uint256"}],
    },
    {
        "name": "castVote",
        "type": "function",
        "inputs": [
            {"name": "proposalId", "type": "uint256"},
            {"name": "support", "type": "uint8"},
        ],
        "outputs": [{"name": "", "type": "uint256"}],
    },
    {
        "name": "castVoteWithReason",
        "type": "function",
        "inputs": [
            {"name": "proposalId", "type": "uint256"},
            {"name": "support", "type": "uint8"},
            {"name": "reason", "type": "string"},
        ],
        "outputs": [{"name": "", "type": "uint256"}],
    },
    {
        "name": "delegate",
        "type": "function",
        "inputs": [{"name": "delegatee", "type": "address"}],
        "outputs": [],
    },
    {
        "name": "getVotes",
        "type": "function",
        "inputs": [{"name": "account", "type": "address"}],
        "outputs": [{"name": "", "type": "uint256"}],
    },
    {
        "name": "proposals",
        "type": "function",
        "inputs": [{"name": "proposalId", "type": "uint256"}],
        "outputs": [
            {"name": "id", "type": "uint256"},
            {"name": "proposer", "type": "address"},
            {"name": "eta", "type": "uint256"},
            {"name": "startBlock", "type": "uint256"},
            {"name": "endBlock", "type": "uint256"},
            {"name": "forVotes", "type": "uint256"},
            {"name": "againstVotes", "type": "uint256"},
            {"name": "abstainVotes", "type": "uint256"},
            {"name": "canceled", "type": "bool"},
            {"name": "executed", "type": "bool"},
        ],
    },
]


class GovernanceModule:
    """Module for governance operations."""

    # Contract addresses (configurable per network)
    GOVERNOR_ADDRESS = "0x0000000000000000000000000000000000001003"
    VDH_TOKEN_ADDRESS = "0x0000000000000000000000000000000000001000"

    def __init__(self, w3: Web3, account: Optional["LocalAccount"] = None):
        """
        Initialize GovernanceModule.

        Args:
            w3: Web3 instance connected to Atlas chain.
            account: Optional account for signing transactions.
        """
        self._w3 = w3
        self._account = account

    def _get_governor(self):
        """Get Governor contract instance."""
        return self._w3.eth.contract(
            address=Web3.to_checksum_address(self.GOVERNOR_ADDRESS),
            abi=GOVERNOR_ABI,
        )

    def _sign_and_send(self, tx: Dict[str, Any]) -> TransactionReceipt:
        """Sign and send a transaction."""
        if not self._account:
            raise ValueError("Account required for transactions")

        tx["from"] = self._account.address
        tx["nonce"] = self._w3.eth.get_transaction_count(self._account.address)
        tx["gas"] = self._w3.eth.estimate_gas(tx)
        tx["gasPrice"] = self._w3.eth.gas_price

        signed = self._account.sign_transaction(tx)
        tx_hash = self._w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = self._w3.eth.wait_for_transaction_receipt(tx_hash)

        return TransactionReceipt(
            tx_hash=receipt["transactionHash"].hex(),
            block_number=receipt["blockNumber"],
            status=receipt["status"] == 1,
            gas_used=receipt["gasUsed"],
            effective_gas_price=receipt.get("effectiveGasPrice", 0),
            logs=[dict(log) for log in receipt["logs"]],
        )

    def create_proposal(
        self,
        targets: List[str],
        values: List[int],
        calldatas: List[bytes],
        description: str,
    ) -> TransactionReceipt:
        """
        Create a new governance proposal.

        Args:
            targets: List of target contract addresses.
            values: List of ETH values to send with each call.
            calldatas: List of encoded function calls.
            description: Proposal description (markdown supported).

        Returns:
            Transaction receipt containing the proposal ID in logs.

        Raises:
            ValueError: If no account is configured.
        """
        if not self._account:
            raise ValueError("Account required to create proposal")

        governor = self._get_governor()
        checksum_targets = [Web3.to_checksum_address(t) for t in targets]

        tx = governor.functions.propose(
            checksum_targets,
            values,
            calldatas,
            description,
        ).build_transaction({
            "chainId": self._w3.eth.chain_id,
        })

        return self._sign_and_send(tx)

    def cast_vote(
        self,
        proposal_id: int,
        support: int,
        reason: Optional[str] = None,
    ) -> TransactionReceipt:
        """
        Cast a vote on a proposal.

        Args:
            proposal_id: The proposal ID to vote on.
            support: Vote type (0=against, 1=for, 2=abstain).
            reason: Optional reason for the vote.

        Returns:
            Transaction receipt.

        Raises:
            ValueError: If no account is configured or invalid support value.
        """
        if not self._account:
            raise ValueError("Account required to vote")

        if support not in (0, 1, 2):
            raise ValueError("Support must be 0 (against), 1 (for), or 2 (abstain)")

        governor = self._get_governor()

        if reason:
            tx = governor.functions.castVoteWithReason(
                proposal_id, support, reason
            ).build_transaction({
                "chainId": self._w3.eth.chain_id,
            })
        else:
            tx = governor.functions.castVote(
                proposal_id, support
            ).build_transaction({
                "chainId": self._w3.eth.chain_id,
            })

        return self._sign_and_send(tx)

    def vote_for(self, proposal_id: int, reason: Optional[str] = None) -> TransactionReceipt:
        """
        Vote in favor of a proposal.

        Args:
            proposal_id: The proposal ID.
            reason: Optional reason for the vote.

        Returns:
            Transaction receipt.
        """
        return self.cast_vote(proposal_id, 1, reason)

    def vote_against(self, proposal_id: int, reason: Optional[str] = None) -> TransactionReceipt:
        """
        Vote against a proposal.

        Args:
            proposal_id: The proposal ID.
            reason: Optional reason for the vote.

        Returns:
            Transaction receipt.
        """
        return self.cast_vote(proposal_id, 0, reason)

    def vote_abstain(self, proposal_id: int, reason: Optional[str] = None) -> TransactionReceipt:
        """
        Abstain from voting on a proposal.

        Args:
            proposal_id: The proposal ID.
            reason: Optional reason.

        Returns:
            Transaction receipt.
        """
        return self.cast_vote(proposal_id, 2, reason)

    def delegate(self, delegatee: str) -> TransactionReceipt:
        """
        Delegate voting power to another address.

        Args:
            delegatee: Address to delegate voting power to.

        Returns:
            Transaction receipt.

        Raises:
            ValueError: If no account is configured.
        """
        if not self._account:
            raise ValueError("Account required to delegate")

        governor = self._get_governor()
        delegatee_address = Web3.to_checksum_address(delegatee)

        tx = governor.functions.delegate(delegatee_address).build_transaction({
            "chainId": self._w3.eth.chain_id,
        })

        return self._sign_and_send(tx)

    def self_delegate(self) -> TransactionReceipt:
        """
        Delegate voting power to self (activate voting).

        Returns:
            Transaction receipt.
        """
        if not self._account:
            raise ValueError("Account required")
        return self.delegate(self._account.address)

    def get_voting_power(self, address: Optional[str] = None) -> int:
        """
        Get voting power for an address.

        Args:
            address: Address to query (defaults to connected account).

        Returns:
            Voting power (in token units).
        """
        if address is None:
            if not self._account:
                raise ValueError("Address required or account must be configured")
            address = self._account.address

        governor = self._get_governor()
        checksum_address = Web3.to_checksum_address(address)
        return governor.functions.getVotes(checksum_address).call()

    def get_proposal(self, proposal_id: int) -> Proposal:
        """
        Get proposal details.

        Args:
            proposal_id: The proposal ID.

        Returns:
            Proposal details.
        """
        result = self._w3.provider.make_request(
            "vdh_getProposal", [proposal_id]
        )

        if "error" in result:
            raise ValueError(result["error"]["message"])

        data = result["result"]

        return Proposal(
            proposal_id=str(data["proposalId"]),
            proposer=data["proposer"],
            title=data.get("title", ""),
            description=data.get("description", ""),
            start_block=data["startBlock"],
            end_block=data["endBlock"],
            for_votes=str(data["forVotes"]),
            against_votes=str(data["againstVotes"]),
            status=data["status"],
        )

    def get_proposals(
        self,
        status: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> List[Proposal]:
        """
        Get list of proposals.

        Args:
            status: Filter by status ('active', 'pending', 'succeeded', 'defeated', 'executed').
            limit: Maximum number of proposals to return.
            offset: Number of proposals to skip.

        Returns:
            List of proposals.
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        result = self._w3.provider.make_request("vdh_getProposals", [params])

        if "error" in result:
            raise ValueError(result["error"]["message"])

        proposals = []
        for data in result["result"]:
            proposals.append(Proposal(
                proposal_id=str(data["proposalId"]),
                proposer=data["proposer"],
                title=data.get("title", ""),
                description=data.get("description", ""),
                start_block=data["startBlock"],
                end_block=data["endBlock"],
                for_votes=str(data["forVotes"]),
                against_votes=str(data["againstVotes"]),
                status=data["status"],
            ))

        return proposals
