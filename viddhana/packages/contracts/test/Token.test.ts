import { expect } from "chai";
import { ethers } from "hardhat";
import { VDHToken, VaultManager, PolicyEngine, VDHGovernance } from "../typechain-types";
import { SignerWithAddress } from "@nomicfoundation/hardhat-ethers/signers";

describe("VIDDHANA Contracts", function () {
  let vdhToken: VDHToken;
  let vaultManager: VaultManager;
  let policyEngine: PolicyEngine;
  let governance: VDHGovernance;
  let owner: SignerWithAddress;
  let user1: SignerWithAddress;
  let user2: SignerWithAddress;
  let treasury: SignerWithAddress;

  beforeEach(async function () {
    [owner, user1, user2, treasury] = await ethers.getSigners();

    // Deploy VDH Token
    const VDHToken = await ethers.getContractFactory("VDHToken");
    vdhToken = await VDHToken.deploy();
    await vdhToken.waitForDeployment();

    // Deploy VaultManager
    const VaultManager = await ethers.getContractFactory("VaultManager");
    vaultManager = await VaultManager.deploy();
    await vaultManager.waitForDeployment();
    await vaultManager.initialize(treasury.address);

    // Deploy PolicyEngine
    const PolicyEngine = await ethers.getContractFactory("PolicyEngine");
    policyEngine = await PolicyEngine.deploy();
    await policyEngine.waitForDeployment();
    await policyEngine.initialize(await vaultManager.getAddress());

    // Deploy Governance
    const VDHGovernance = await ethers.getContractFactory("VDHGovernance");
    governance = await VDHGovernance.deploy();
    await governance.waitForDeployment();
    await governance.initialize(await vdhToken.getAddress());

    // Setup roles
    const POLICY_ENGINE_ROLE = ethers.keccak256(ethers.toUtf8Bytes("POLICY_ENGINE_ROLE"));
    await vaultManager.grantRole(POLICY_ENGINE_ROLE, await policyEngine.getAddress());

    // Add VDH as supported asset
    await vaultManager.addSupportedAsset(await vdhToken.getAddress(), "VDH", true, false);
  });

  describe("VDHToken", function () {
    it("Should have correct name and symbol", async function () {
      expect(await vdhToken.name()).to.equal("Viddhana Token");
      expect(await vdhToken.symbol()).to.equal("VDH");
    });

    it("Should have correct total supply", async function () {
      const expectedSupply = ethers.parseEther("1000000000"); // 1 billion
      expect(await vdhToken.totalSupply()).to.equal(expectedSupply);
    });

    it("Should mint initial supply to deployer", async function () {
      const expectedSupply = ethers.parseEther("1000000000");
      expect(await vdhToken.balanceOf(owner.address)).to.equal(expectedSupply);
    });

    it("Should not allow minting beyond max supply", async function () {
      await expect(
        vdhToken.mint(user1.address, ethers.parseEther("1"))
      ).to.be.revertedWith("Max supply exceeded");
    });

    it("Should allow burning tokens", async function () {
      const burnAmount = ethers.parseEther("1000");
      await vdhToken.burn(burnAmount);
      
      const expectedBalance = ethers.parseEther("1000000000") - burnAmount;
      expect(await vdhToken.balanceOf(owner.address)).to.equal(expectedBalance);
    });
  });

  describe("VaultManager", function () {
    beforeEach(async function () {
      // Transfer tokens to user1 for testing
      await vdhToken.transfer(user1.address, ethers.parseEther("10000"));
      await vdhToken.connect(user1).approve(await vaultManager.getAddress(), ethers.parseEther("10000"));
    });

    it("Should allow deposits", async function () {
      const depositAmount = ethers.parseEther("1000");
      await vaultManager.connect(user1).deposit(await vdhToken.getAddress(), depositAmount);

      // Account for 0.1% fee
      const expectedBalance = depositAmount - (depositAmount * 10n / 10000n);
      expect(await vaultManager.getUserBalance(user1.address, await vdhToken.getAddress())).to.equal(expectedBalance);
    });

    it("Should allow withdrawals", async function () {
      const depositAmount = ethers.parseEther("1000");
      await vaultManager.connect(user1).deposit(await vdhToken.getAddress(), depositAmount);

      const balance = await vaultManager.getUserBalance(user1.address, await vdhToken.getAddress());
      await vaultManager.connect(user1).withdraw(await vdhToken.getAddress(), balance);

      expect(await vaultManager.getUserBalance(user1.address, await vdhToken.getAddress())).to.equal(0);
    });

    it("Should reject unsupported assets", async function () {
      const fakeToken = ethers.Wallet.createRandom().address;
      await expect(
        vaultManager.connect(user1).deposit(fakeToken, ethers.parseEther("100"))
      ).to.be.revertedWithCustomError(vaultManager, "AssetNotSupported");
    });

    it("Should reject zero deposits", async function () {
      await expect(
        vaultManager.connect(user1).deposit(await vdhToken.getAddress(), 0)
      ).to.be.revertedWithCustomError(vaultManager, "InvalidAmount");
    });
  });

  describe("PolicyEngine", function () {
    it("Should allow users to set profile", async function () {
      await policyEngine.connect(user1).setUserProfile(5000, 24, true);

      const profile = await policyEngine.getUserProfile(user1.address);
      expect(profile.riskTolerance).to.equal(5000);
      expect(profile.timeToGoal).to.equal(24);
      expect(profile.autoRebalanceEnabled).to.equal(true);
      expect(profile.isActive).to.equal(true);
    });

    it("Should reject invalid risk tolerance", async function () {
      await expect(
        policyEngine.connect(user1).setUserProfile(15000, 24, true) // > 10000
      ).to.be.revertedWithCustomError(policyEngine, "InvalidRiskTolerance");
    });

    it("Should calculate max volatile share correctly", async function () {
      // High risk, long term = higher volatile share
      await policyEngine.connect(user1).setUserProfile(10000, 36, true);
      const profile1 = await policyEngine.getUserProfile(user1.address);
      
      // Low risk, short term = lower volatile share
      await policyEngine.connect(user2).setUserProfile(2000, 6, true);
      const profile2 = await policyEngine.getUserProfile(user2.address);

      expect(profile1.maxVolatileShare).to.be.greaterThan(profile2.maxVolatileShare);
    });

    it("Should check canRebalance correctly", async function () {
      await policyEngine.connect(user1).setUserProfile(5000, 24, true);
      
      // Initially false due to MIN_REBALANCE_INTERVAL
      expect(await policyEngine.canRebalance(user1.address)).to.equal(false);
    });

    it("Should increment totalUsers on new profile", async function () {
      expect(await policyEngine.totalUsers()).to.equal(0);
      
      await policyEngine.connect(user1).setUserProfile(5000, 24, true);
      expect(await policyEngine.totalUsers()).to.equal(1);
      
      await policyEngine.connect(user2).setUserProfile(3000, 12, true);
      expect(await policyEngine.totalUsers()).to.equal(2);
    });
  });

  describe("VDHGovernance", function () {
    beforeEach(async function () {
      // Transfer enough tokens to user1 to create proposals
      await vdhToken.transfer(user1.address, ethers.parseEther("200000")); // 200k VDH
    });

    it("Should allow creating proposals with sufficient tokens", async function () {
      await governance.connect(user1).propose(
        "Test Proposal",
        "This is a test proposal description"
      );

      const proposal = await governance.getProposal(1);
      expect(proposal.title).to.equal("Test Proposal");
      expect(proposal.proposer).to.equal(user1.address);
    });

    it("Should reject proposals from users with insufficient tokens", async function () {
      await expect(
        governance.connect(user2).propose("Test", "Description")
      ).to.be.revertedWithCustomError(governance, "InsufficientBalance");
    });

    it("Should track proposal count", async function () {
      await governance.connect(user1).propose("Proposal 1", "Description 1");
      await governance.connect(user1).propose("Proposal 2", "Description 2");

      expect(await governance.proposalCount()).to.equal(2);
    });

    it("Should allow voting after voting delay", async function () {
      await governance.connect(user1).propose("Test", "Description");

      // Move time forward past voting delay (1 day)
      await ethers.provider.send("evm_increaseTime", [86401]); // 1 day + 1 second
      await ethers.provider.send("evm_mine", []);

      // Vote
      await governance.connect(user1).castVote(1, 1); // Vote For

      const proposal = await governance.getProposal(1);
      expect(proposal.forVotes).to.be.greaterThan(0);
    });

    it("Should prevent double voting", async function () {
      await governance.connect(user1).propose("Test", "Description");

      await ethers.provider.send("evm_increaseTime", [86401]);
      await ethers.provider.send("evm_mine", []);

      await governance.connect(user1).castVote(1, 1);

      await expect(
        governance.connect(user1).castVote(1, 1)
      ).to.be.revertedWithCustomError(governance, "AlreadyVoted");
    });
  });

  describe("Integration", function () {
    it("Should complete a full deposit and profile setup flow", async function () {
      // Transfer tokens to user
      await vdhToken.transfer(user1.address, ethers.parseEther("10000"));
      await vdhToken.connect(user1).approve(await vaultManager.getAddress(), ethers.parseEther("10000"));

      // Set user profile
      await policyEngine.connect(user1).setUserProfile(5000, 24, true);

      // Deposit funds
      await vaultManager.connect(user1).deposit(await vdhToken.getAddress(), ethers.parseEther("1000"));

      // Verify state
      const profile = await policyEngine.getUserProfile(user1.address);
      expect(profile.isActive).to.equal(true);

      const balance = await vaultManager.getUserBalance(user1.address, await vdhToken.getAddress());
      expect(balance).to.be.greaterThan(0);
    });
  });
});
