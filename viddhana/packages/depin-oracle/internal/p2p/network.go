// Package p2p implements the peer-to-peer network layer for the DePIN Oracle.
package p2p

import (
	"context"
	"fmt"
	"sync"

	"github.com/libp2p/go-libp2p"
	pubsub "github.com/libp2p/go-libp2p-pubsub"
	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/multiformats/go-multiaddr"
)

// Config holds P2P network configuration
type Config struct {
	ListenAddr     string
	BootstrapPeers []string
	PrivateKeyPath string
}

// MessageHandler is a function that handles incoming P2P messages
type MessageHandler func(ctx context.Context, data []byte) error

// Network represents a P2P network node
type Network struct {
	host       host.Host
	pubsub     *pubsub.PubSub
	topics     map[string]*pubsub.Topic
	subs       map[string]*pubsub.Subscription
	handlers   map[string]MessageHandler
	topicsMu   sync.RWMutex
	config     Config
	ctx        context.Context
	cancel     context.CancelFunc
	peerCount  int
	peerCountMu sync.RWMutex
}

// NewNetwork creates a new P2P network instance
func NewNetwork(cfg Config) (*Network, error) {
	ctx, cancel := context.WithCancel(context.Background())

	// Parse listen address
	listenAddr, err := multiaddr.NewMultiaddr(cfg.ListenAddr)
	if err != nil {
		cancel()
		return nil, fmt.Errorf("invalid listen address: %w", err)
	}

	// Create libp2p host options
	opts := []libp2p.Option{
		libp2p.ListenAddrs(listenAddr),
	}

	// Load private key if provided
	if cfg.PrivateKeyPath != "" {
		// In production, load key from file
		// For now, generate a new key
		priv, _, err := crypto.GenerateKeyPair(crypto.Ed25519, -1)
		if err != nil {
			cancel()
			return nil, fmt.Errorf("failed to generate key: %w", err)
		}
		opts = append(opts, libp2p.Identity(priv))
	}

	// Create libp2p host
	h, err := libp2p.New(opts...)
	if err != nil {
		cancel()
		return nil, fmt.Errorf("failed to create host: %w", err)
	}

	// Create pubsub
	ps, err := pubsub.NewGossipSub(ctx, h)
	if err != nil {
		h.Close()
		cancel()
		return nil, fmt.Errorf("failed to create pubsub: %w", err)
	}

	return &Network{
		host:     h,
		pubsub:   ps,
		topics:   make(map[string]*pubsub.Topic),
		subs:     make(map[string]*pubsub.Subscription),
		handlers: make(map[string]MessageHandler),
		config:   cfg,
		ctx:      ctx,
		cancel:   cancel,
	}, nil
}

// Start starts the P2P network and connects to bootstrap peers
func (n *Network) Start(ctx context.Context) error {
	// Connect to bootstrap peers
	for _, peerAddr := range n.config.BootstrapPeers {
		addr, err := multiaddr.NewMultiaddr(peerAddr)
		if err != nil {
			continue
		}

		peerInfo, err := peer.AddrInfoFromP2pAddr(addr)
		if err != nil {
			continue
		}

		if err := n.host.Connect(ctx, *peerInfo); err != nil {
			// Log but don't fail - bootstrap peers might not be available
			continue
		}

		n.peerCountMu.Lock()
		n.peerCount++
		n.peerCountMu.Unlock()
	}

	return nil
}

// Subscribe subscribes to a topic and sets up a message handler
func (n *Network) Subscribe(topic string, handler MessageHandler) error {
	n.topicsMu.Lock()
	defer n.topicsMu.Unlock()

	// Join topic if not already joined
	t, exists := n.topics[topic]
	if !exists {
		var err error
		t, err = n.pubsub.Join(topic)
		if err != nil {
			return fmt.Errorf("failed to join topic %s: %w", topic, err)
		}
		n.topics[topic] = t
	}

	// Subscribe to topic
	sub, err := t.Subscribe()
	if err != nil {
		return fmt.Errorf("failed to subscribe to topic %s: %w", topic, err)
	}
	n.subs[topic] = sub
	n.handlers[topic] = handler

	// Start message handler goroutine
	go n.handleMessages(topic, sub, handler)

	return nil
}

// handleMessages processes incoming messages for a topic
func (n *Network) handleMessages(topic string, sub *pubsub.Subscription, handler MessageHandler) {
	for {
		msg, err := sub.Next(n.ctx)
		if err != nil {
			// Context cancelled or subscription closed
			return
		}

		// Skip messages from self
		if msg.ReceivedFrom == n.host.ID() {
			continue
		}

		// Handle the message
		if err := handler(n.ctx, msg.Data); err != nil {
			// Log error but continue processing
			continue
		}
	}
}

// Publish publishes a message to a topic
func (n *Network) Publish(topic string, data []byte) error {
	n.topicsMu.RLock()
	t, exists := n.topics[topic]
	n.topicsMu.RUnlock()

	if !exists {
		// Join topic if not already joined
		n.topicsMu.Lock()
		var err error
		t, err = n.pubsub.Join(topic)
		if err != nil {
			n.topicsMu.Unlock()
			return fmt.Errorf("failed to join topic %s: %w", topic, err)
		}
		n.topics[topic] = t
		n.topicsMu.Unlock()
	}

	return t.Publish(n.ctx, data)
}

// PeerCount returns the number of connected peers
func (n *Network) PeerCount() int {
	n.peerCountMu.RLock()
	defer n.peerCountMu.RUnlock()
	return n.peerCount
}

// Stop stops the P2P network
func (n *Network) Stop() error {
	n.cancel()

	// Close all subscriptions
	n.topicsMu.Lock()
	for _, sub := range n.subs {
		sub.Cancel()
	}
	for _, topic := range n.topics {
		topic.Close()
	}
	n.topicsMu.Unlock()

	return n.host.Close()
}

// ID returns the peer ID of this node
func (n *Network) ID() string {
	return n.host.ID().String()
}

// Addrs returns the listen addresses of this node
func (n *Network) Addrs() []string {
	addrs := n.host.Addrs()
	result := make([]string, len(addrs))
	for i, addr := range addrs {
		result[i] = addr.String()
	}
	return result
}
