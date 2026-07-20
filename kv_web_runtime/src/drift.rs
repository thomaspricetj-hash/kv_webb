//! drift.rs
//! Relevance drift physics for KV Web + BitDrop_v2 max‑tier compression.
//!
//! This module handles:
//! - time‑based score decay
//! - reinforcement when nodes are accessed
//! - drift curves (linear, exponential)
//! - edge drift
//!
//! With BitDrop_v2 wired in, drift updates produce compressed packets
//! that can be logged, replayed, or routed into higher‑level memory.

use kv_web_core::{KvWeb, WebNodeId, NodeDriftState, KvWebCompressor};
use std::time::Instant;

/// Drift mode for score decay.
#[derive(Debug, Clone)]
pub enum DriftMode {
    Linear,       // score -= decay_rate * elapsed
    Exponential,  // score *= (1.0 - decay_rate)^elapsed
}

/// Drift configuration.
#[derive(Debug, Clone)]
pub struct DriftConfig {
    pub decay_rate: f32,           // how fast nodes drift
    pub edge_decay_rate: f32,      // how fast edges drift
    pub mode: DriftMode,           // linear or exponential
    pub reinforcement_amount: f32, // score bump when node is accessed
}

/// Extension trait for drift physics.
pub trait KvWebDrift {
    /// Initialize drift state for all nodes.
    fn init_drift_state(&mut self);

    /// Apply drift to all nodes based on time since last access.
    /// Returns an optional compressed drift packet.
    fn apply_drift(&mut self, cfg: &DriftConfig) -> Option<Vec<u8>>;

    /// Apply drift to edges.
    /// Returns an optional compressed edge‑drift packet.
    fn apply_edge_drift(&mut self, cfg: &DriftConfig) -> Option<Vec<u8>>;

    /// Reinforce a node (called when node is accessed).
    /// Returns an optional compressed reinforcement packet.
    fn reinforce_node(&mut self, node: WebNodeId, cfg: &DriftConfig) -> Option<Vec<u8>>;
}

impl KvWebDrift for KvWeb {
    fn init_drift_state(&mut self) {
        if self.drift_state.is_none() {
            self.drift_state = Some(std::collections::HashMap::new());
        }

        let state = self.drift_state.as_mut().unwrap();

        // Ensure every node has drift state
        for id in self.nodes.keys() {
            state.entry(*id).or_insert(NodeDriftState::new());
        }
    }

    fn apply_drift(&mut self, cfg: &DriftConfig) -> Option<Vec<u8>> {
        let Some(state) = self.drift_state.as_mut() else {
            return None;
        };

        let mut updates: Vec<(WebNodeId, f32)> = Vec::new();

        for (id, node) in &mut self.nodes {
            if let Some(drift) = state.get(id) {
                let elapsed = drift.last_access.elapsed().as_secs_f32();

                match cfg.mode {
                    DriftMode::Linear => {
                        node.score -= cfg.decay_rate * elapsed;
                        if node.score < 0.0 {
                            node.score = 0.0;
                        }
                    }
                    DriftMode::Exponential => {
                        let factor = (1.0 - cfg.decay_rate).clamp(0.0, 1.0);
                        node.score *= factor.powf(elapsed);
                        if node.score < 0.0 {
                            node.score = 0.0;
                        }
                    }
                }

                updates.push((*id, node.score));
            }
        }

        // MAX‑TIER BitDrop_v2 compressed drift packet
        self.compressor.as_ref().map(|c| {
            c.compress(&(
                "apply_drift",
                cfg.decay_rate,
                &updates
            ))
        })
    }

    fn apply_edge_drift(&mut self, cfg: &DriftConfig) -> Option<Vec<u8>> {
        let mut before_after: Vec<(WebNodeId, WebNodeId, f32)> = Vec::new();

        for edge in &mut self.edges {
            let before = edge.weight;
            match cfg.mode {
                DriftMode::Linear => {
                    edge.weight -= cfg.edge_decay_rate;
                }
                DriftMode::Exponential => {
                    let factor = (1.0 - cfg.edge_decay_rate).clamp(0.0, 1.0);
                    edge.weight *= factor;
                }
            }
            before_after.push((edge.from, edge.to, before));
        }

        // Remove dead edges
        self.edges.retain(|e| e.weight > 0.0);

        // MAX‑TIER BitDrop_v2 compressed edge‑drift packet
        self.compressor.as_ref().map(|c| {
            c.compress(&(
                "apply_edge_drift",
                cfg.edge_decay_rate,
                &before_after
            ))
        })
    }

    fn reinforce_node(&mut self, node: WebNodeId, cfg: &DriftConfig) -> Option<Vec<u8>> {
        let mut new_score = None;

        if let Some(n) = self.nodes.get_mut(&node) {
            n.score += cfg.reinforcement_amount;
            new_score = Some(n.score);
        }

        if let Some(state) = self.drift_state.as_mut() {
            if let Some(s) = state.get_mut(&node) {
                s.last_access = Instant::now();

                // Also update compressed drift packet for this node
                if let Some(comp) = &self.compressor {
                    let packet = comp.compress(&(s.drift_score, s.reinforcement_score));
                    s.drift_packet_compressed = Some(packet);
                }
            }
        }

        // MAX‑TIER BitDrop_v2 compressed reinforcement packet
        self.compressor.as_ref().map(|c| {
            c.compress(&(
                "reinforce_node",
                node,
                cfg.reinforcement_amount,
                new_score
            ))
        })
    }
}


