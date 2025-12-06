---
sidebar_position: 2
title: Quickstart
---

# Quickstart Guide

Deploy your first smart contract on VIDDHANA in under 5 minutes.

## Initialize a Project

```bash
mkdir my-viddhana-project && cd my-viddhana-project
npm init -y
npm install ethers hardhat @nomicfoundation/hardhat-toolbox dotenv
npx hardhat init
```

Select "Create a TypeScript project" when prompted.

## Configure Hardhat

Update `hardhat.config.ts`:

```typescript
import { HardhatUserConfig } from "hardhat/config";
import "@nomicfoundation/hardhat-toolbox";
import * as dotenv from "dotenv";

dotenv.config();

const config: HardhatUserConfig = {
  solidity: "0.8.20",
  networks: {
    viddhana: {
      url: "https://rpc.viddhana.com",
      chainId: 13370,
      accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
    },
    localhost: {
      url: "http://localhost:8545",
      chainId: 1337,
    },
  },
};

export default config;
```

Create `.env` file:

```bash
PRIVATE_KEY=your_private_key_here
```

## Create a Simple Contract

Create `contracts/HelloViddhana.sol`:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract HelloViddhana {
    string public message;
    address public owner;

    event MessageUpdated(string oldMessage, string newMessage, address updatedBy);

    constructor(string memory _message) {
        message = _message;
        owner = msg.sender;
    }

    function updateMessage(string memory _newMessage) public {
        string memory oldMessage = message;
        message = _newMessage;
        emit MessageUpdated(oldMessage, _newMessage, msg.sender);
    }

    function getMessage() public view returns (string memory) {
        return message;
    }
}
```

## Compile the Contract

```bash
npx hardhat compile
```

Expected output:
```
Compiled 1 Solidity file successfully
```

## Deploy Script

Create `scripts/deploy.ts`:

```typescript
import { ethers } from "hardhat";

async function main() {
  const [deployer] = await ethers.getSigners();
  console.log("Deploying with account:", deployer.address);
  
  const balance = await ethers.provider.getBalance(deployer.address);
  console.log("Account balance:", ethers.formatEther(balance), "VDH");

  const HelloViddhana = await ethers.getContractFactory("HelloViddhana");
  const contract = await HelloViddhana.deploy("Hello, VIDDHANA!");

  await contract.waitForDeployment();
  const address = await contract.getAddress();

  console.log("HelloViddhana deployed to:", address);
  console.log("Transaction hash:", contract.deploymentTransaction()?.hash);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
```

## Deploy to Local Testnet

```bash
# Deploy to localhost (ensure local node is running)
npx hardhat run scripts/deploy.ts --network localhost
```

## Deploy to VIDDHANA Mainnet

```bash
npx hardhat run scripts/deploy.ts --network viddhana
```

Expected output:
```
Deploying with account: 0xYourAddress
Account balance: 100.0 VDH
HelloViddhana deployed to: 0xContractAddress
Transaction hash: 0x...
```

## Interact with the Contract

Create `scripts/interact.ts`:

```typescript
import { ethers } from "hardhat";

const CONTRACT_ADDRESS = "0xYourDeployedContractAddress";

async function main() {
  const HelloViddhana = await ethers.getContractAt("HelloViddhana", CONTRACT_ADDRESS);

  // Read current message
  const message = await HelloViddhana.getMessage();
  console.log("Current message:", message);

  // Update message
  console.log("Updating message...");
  const tx = await HelloViddhana.updateMessage("Building on VIDDHANA!");
  await tx.wait();
  console.log("Message updated! Tx:", tx.hash);

  // Read new message
  const newMessage = await HelloViddhana.getMessage();
  console.log("New message:", newMessage);
}

main().catch(console.error);
```

Run:
```bash
npx hardhat run scripts/interact.ts --network localhost
```

## Using ethers.js Directly

```typescript
import { ethers } from 'ethers';

// Connect to VIDDHANA
const provider = new ethers.JsonRpcProvider('https://rpc.viddhana.com');

// Create wallet
const wallet = new ethers.Wallet(process.env.PRIVATE_KEY!, provider);

// Contract ABI (minimal)
const abi = [
  "function getMessage() view returns (string)",
  "function updateMessage(string memory _newMessage)",
  "event MessageUpdated(string oldMessage, string newMessage, address updatedBy)"
];

// Connect to contract
const contract = new ethers.Contract(CONTRACT_ADDRESS, abi, wallet);

// Read message
const message = await contract.getMessage();
console.log("Message:", message);

// Update message
const tx = await contract.updateMessage("Hello from ethers.js!");
await tx.wait();
console.log("Updated!");
```

## Using the API

Interact with contracts via the VIDDHANA API:

```bash
# Get account balance
curl -X POST https://api.viddhana.com/rpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "atlas_getBalance",
    "params": ["0xYourAddress"]
  }'

# Call contract (read-only)
curl -X POST https://api.viddhana.com/rpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "eth_call",
    "params": [{
      "to": "0xContractAddress",
      "data": "0xce6d41de"
    }, "latest"]
  }'
```

## Verify Contract on Explorer

After deployment, verify your contract on the block explorer:

1. Go to https://scan.viddhana.com
2. Search for your contract address
3. Click "Verify & Publish"
4. Upload source code and compiler settings

## Project Structure

After setup, your project should look like:

```
my-viddhana-project/
├── contracts/
│   └── HelloViddhana.sol
├── scripts/
│   ├── deploy.ts
│   └── interact.ts
├── test/
│   └── HelloViddhana.test.ts
├── hardhat.config.ts
├── package.json
└── .env
```

## Write Tests

Create `test/HelloViddhana.test.ts`:

```typescript
import { expect } from "chai";
import { ethers } from "hardhat";

describe("HelloViddhana", function () {
  it("Should deploy with initial message", async function () {
    const HelloViddhana = await ethers.getContractFactory("HelloViddhana");
    const contract = await HelloViddhana.deploy("Hello, VIDDHANA!");
    
    expect(await contract.getMessage()).to.equal("Hello, VIDDHANA!");
  });

  it("Should update message", async function () {
    const HelloViddhana = await ethers.getContractFactory("HelloViddhana");
    const contract = await HelloViddhana.deploy("Hello, VIDDHANA!");
    
    await contract.updateMessage("New message");
    expect(await contract.getMessage()).to.equal("New message");
  });

  it("Should emit event on update", async function () {
    const HelloViddhana = await ethers.getContractFactory("HelloViddhana");
    const contract = await HelloViddhana.deploy("Hello, VIDDHANA!");
    
    await expect(contract.updateMessage("New message"))
      .to.emit(contract, "MessageUpdated")
      .withArgs("Hello, VIDDHANA!", "New message", await (await ethers.getSigners())[0].getAddress());
  });
});
```

Run tests:
```bash
npx hardhat test
```

## Next Steps

- Explore [Smart Contract Architecture](/docs/smart-contracts/overview)
- Learn about the [VDH Token](/docs/smart-contracts/vdh-token)
- Check out the [JSON-RPC API](/docs/api-reference/json-rpc)
- Try our [JavaScript SDK](/docs/sdks/javascript)

---

Need help? Join our [Discord](https://discord.gg/viddhana) community.
