import { ethers } from "hardhat";

async function main() {
  const [deployer] = await ethers.getSigners();
  console.log("Deploying contracts with:", deployer.address);
  console.log("Account balance:", (await ethers.provider.getBalance(deployer.address)).toString());

  // 1. Deploy VDH Token
  console.log("\n1. Deploying VDH Token...");
  const VDHToken = await ethers.getContractFactory("VDHToken");
  const vdhToken = await VDHToken.deploy();
  await vdhToken.waitForDeployment();
  const vdhTokenAddress = await vdhToken.getAddress();
  console.log("VDH Token deployed to:", vdhTokenAddress);

  // 2. Deploy VaultManager (implementation only - no proxy for testing)
  console.log("\n2. Deploying VaultManager...");
  const VaultManager = await ethers.getContractFactory("VaultManager");
  const vaultManager = await VaultManager.deploy();
  await vaultManager.waitForDeployment();
  const vaultManagerAddress = await vaultManager.getAddress();
  console.log("VaultManager deployed to:", vaultManagerAddress);

  // 3. Deploy PolicyEngine (implementation only)
  console.log("\n3. Deploying PolicyEngine...");
  const PolicyEngine = await ethers.getContractFactory("PolicyEngine");
  const policyEngine = await PolicyEngine.deploy();
  await policyEngine.waitForDeployment();
  const policyEngineAddress = await policyEngine.getAddress();
  console.log("PolicyEngine deployed to:", policyEngineAddress);

  // 4. Deploy VDHGovernance (implementation only)
  console.log("\n4. Deploying VDHGovernance...");
  const VDHGovernance = await ethers.getContractFactory("VDHGovernance");
  const governance = await VDHGovernance.deploy();
  await governance.waitForDeployment();
  const governanceAddress = await governance.getAddress();
  console.log("VDHGovernance deployed to:", governanceAddress);

  // Output deployment summary
  console.log("\n========== DEPLOYMENT SUMMARY ==========");
  console.log("Chain ID:", (await ethers.provider.getNetwork()).chainId);
  console.log("Deployer:", deployer.address);
  console.log("----------------------------------------");
  console.log("VDH Token:      ", vdhTokenAddress);
  console.log("VaultManager:   ", vaultManagerAddress);
  console.log("PolicyEngine:   ", policyEngineAddress);
  console.log("VDHGovernance:  ", governanceAddress);
  console.log("==========================================");
  console.log("\nAll contracts deployed successfully!");
  console.log("Note: Upgradeable contracts deployed as implementations.");
  console.log("For production, use OpenZeppelin Upgrades plugin with proxy pattern.");

  // Return addresses for testing
  return {
    vdhToken: vdhTokenAddress,
    vaultManager: vaultManagerAddress,
    policyEngine: policyEngineAddress,
    governance: governanceAddress
  };
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
