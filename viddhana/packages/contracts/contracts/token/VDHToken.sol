// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Permit.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";

/**
 * @title VDHToken
 * @author VIDDHANA Team
 * @notice Native token for the VIDDHANA ecosystem
 * @dev Fixed supply of 1 billion tokens
 */
contract VDHToken is ERC20, ERC20Burnable, ERC20Permit, AccessControl {
    
    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    
    uint256 public constant MAX_SUPPLY = 1_000_000_000 * 10**18;  // 1 billion
    
    constructor() ERC20("Viddhana Token", "VDH") ERC20Permit("Viddhana Token") {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(MINTER_ROLE, msg.sender);
        
        // Mint initial supply to deployer
        _mint(msg.sender, MAX_SUPPLY);
    }
    
    /**
     * @notice Mint new tokens (only if under max supply)
     * @dev Only callable by addresses with MINTER_ROLE
     */
    function mint(address to, uint256 amount) external onlyRole(MINTER_ROLE) {
        require(totalSupply() + amount <= MAX_SUPPLY, "Max supply exceeded");
        _mint(to, amount);
    }
}
