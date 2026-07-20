//! drift.rs
//! Relevance drift physics for KV Web.
//!
//! This module handles:
//! - time‑based score decay
//! - reinforcement when nodes are accessed
//! - drift curves (linear, exponential)
//! - edge drift
//!
//! This makes the KV Web behave like a living memory system.

use kv_web_core::{KvWeb, WebNodeId, NodeDriftState};
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
    fn apply_drift(&mut self, cfg: &DriftConfig);

    /// Apply drift to edges.
    fn apply_edge_drift(&mut self, cfg: &DriftConfig);

    /// Reinforce a node (called when node is accessed).
    fn reinforce_node(&mut self, node: WebNodeId, cfg: &DriftConfig);
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

    fn apply_drift(&mut self, cfg: &DriftConfig) {
        let Some(state) = self.drift_state.as_mut() else {
            return;
        };

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
            }
        }
    }

    fn apply_edge_drift(&mut self, cfg: &DriftConfig) {
        for edge in &mut self.edges {
            match cfg.mode {
                DriftMode::Linear => {
                    edge.weight -= cfg.edge_decay_rate;
                }
                DriftMode::Exponential => {
                    let factor = (1.0 - cfg.edge_decay_rate).clamp(0.0, 1.0);
                    edge.weight *= factor;
                }
            }
        }

        // Remove dead edges
        self.edges.retain(|e| e.weight > 0.0);
    }

    fn reinforce_node(&mut self, node: WebNodeId, cfg: &DriftConfig) {
        if let Some(n) = self.nodes.get_mut(&node) {
            n.score += cfg.reinforcement_amount;
        }

        if let Some(state) = self.drift_state.as_mut() {
            if let Some(s) = state.get_mut(&node) {
                s.last_access = Instant::now();
            }
        }
    }
}

