//! drift.rs
//! Relevance drift physics for KV Web + BitDrop_v2 max‑tier compression.
//!
//! This module handles:
//! - time‑based score decay
//! - reinforcement when nodes are accessed
//! - drift curves (linear, exponential)
//! - edge drift
//!
//! Tier‑6 upgrades:
//! - parallel node drift
//! - parallel edge drift (with safe collection)
//! - dual-layer drift scratch pads (scores + normalized drift)
//! - GPU-ready compressed drift packets
//!
//! All upgrades are backwards-compatible.

use kv_web_core::{KvWeb, WebNodeId, NodeDriftState, KvWebCompressor};
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use std::time::Instant;

/// Drift mode for score decay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DriftMode {
    Linear,       // score -= decay_rate * elapsed
    Exponential,  // score *= (1.0 - decay_rate)^elapsed
}

/// Drift configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftConfig {
    pub decay_rate: f32,           // how fast nodes drift
    pub edge_decay_rate: f32,      // how fast edges drift
    pub mode: DriftMode,           // linear or exponential
    pub reinforcement_amount: f32, // score bump when node is accessed
}

/// Drift optimization configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftOptimizationConfig {
    pub min_decay_rate: f32,
    pub max_decay_rate: f32,
    pub min_edge_decay_rate: f32,
    pub max_edge_decay_rate: f32,
    pub max_allowed_score_drop: f32,
    pub min_reinforcement_amount: f32,
    pub max_reinforcement_amount: f32,
}

/// Dual-layer scratch pad for drift.
/// Layer A = node scores / edge weights
/// Layer B = normalized drift factor (0..1)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftScratchPad {
    pub layer_a: Vec<f32>,
    pub layer_b: Vec<f32>,
}

/// Build node drift scratch pad.
fn build_node_drift_scratch_pad(web: &KvWeb) -> DriftScratchPad {
    let mut layer_a = Vec::with_capacity(web.nodes.len());
    let mut min_s = f32::MAX;
    let mut max_s = f32::MIN;

    for (_, node) in &web.nodes {
        layer_a.push(node.score);
        if node.score < min_s {
            min_s = node.score;
        }
        if node.score > max_s {
            max_s = node.score;
        }
    }

    let span = if max_s > min_s { max_s - min_s } else { 1.0 };

    let mut layer_b = Vec::with_capacity(layer_a.len());
    for s in &layer_a {
        let norm = (*s - min_s) / span;
        layer_b.push(norm);
    }

    DriftScratchPad { layer_a, layer_b }
}

/// Build edge drift scratch pad.
fn build_edge_drift_scratch_pad(web: &KvWeb) -> DriftScratchPad {
    let mut layer_a = Vec::with_capacity(web.edges.len());
    let mut min_w = f32::MAX;
    let mut max_w = f32::MIN;

    for e in &web.edges {
        layer_a.push(e.weight);
        if e.weight < min_w {
            min_w = e.weight;
        }
        if e.weight > max_w {
            max_w = e.weight;
        }
    }

    let span = if max_w > min_w { max_w - min_w } else { 1.0 };

    let mut layer_b = Vec::with_capacity(layer_a.len());
    for w in &layer_a {
        let norm = (*w - min_w) / span;
        layer_b.push(norm);
    }

    DriftScratchPad { layer_a, layer_b }
}

/// Extension trait for drift physics.
pub trait KvWebDrift {
    fn init_drift_state(&mut self);

    fn apply_drift(&mut self, cfg: &DriftConfig) -> Option<Vec<u8>>;

    fn apply_edge_drift(&mut self, cfg: &DriftConfig) -> Option<Vec<u8>>;

    fn reinforce_node(&mut self, node: WebNodeId, cfg: &DriftConfig) -> Option<Vec<u8>>;

    fn optimize_drift(
        &mut self,
        cfg: &mut DriftConfig,
        opt_cfg: &DriftOptimizationConfig,
    ) -> Option<Vec<u8>>;
}

impl KvWebDrift for KvWeb {
    fn init_drift_state(&mut self) {
        if self.drift_state.is_none() {
            self.drift_state = Some(std::collections::HashMap::new());
        }

        let state = self.drift_state.as_mut().unwrap();

        for id in self.nodes.keys() {
            state.entry(*id).or_insert(NodeDriftState::new());
        }
    }

    /// Parallel node drift.
    fn apply_drift(&mut self, cfg: &DriftConfig) -> Option<Vec<u8>> {
        let Some(state) = self.drift_state.as_mut() else {
            return None;
        };

        let mut updates: Vec<(WebNodeId, f32)> = Vec::with_capacity(self.nodes.len());

        self.nodes
            .par_iter_mut()
            .for_each(|(id, node)| {
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
            });

        for (id, node) in &self.nodes {
            updates.push((*id, node.score));
        }

        let scratch = build_node_drift_scratch_pad(self);

        self.compressor.as_ref().map(|c| {
            c.compress(&(
                "apply_drift",
                cfg.decay_rate,
                &updates,
                &scratch.layer_a,
                &scratch.layer_b,
            ))
        })
    }

    /// Parallel edge drift with safe collection.
    fn apply_edge_drift(&mut self, cfg: &DriftConfig) -> Option<Vec<u8>> {
        // Compute before/after in parallel, but collect into a new Vec.
        let before_after: Vec<(WebNodeId, WebNodeId, f32)> = self
            .edges
            .par_iter_mut()
            .map(|edge| {
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
                (edge.from, edge.to, before)
            })
            .collect();

        self.edges.retain(|e| e.weight > 0.0);

        let scratch = build_edge_drift_scratch_pad(self);

        self.compressor.as_ref().map(|c| {
            c.compress(&(
                "apply_edge_drift",
                cfg.edge_decay_rate,
                &before_after,
                &scratch.layer_a,
                &scratch.layer_b,
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

                if let Some(comp) = &self.compressor {
                    let packet = comp.compress(&(s.drift_score, s.reinforcement_score));
                    s.drift_packet_compressed = Some(packet);
                }
            }
        }

        let scratch = build_node_drift_scratch_pad(self);

        self.compressor.as_ref().map(|c| {
            c.compress(&(
                "reinforce_node",
                node,
                cfg.reinforcement_amount,
                new_score,
                &scratch.layer_a,
                &scratch.layer_b,
            ))
        })
    }

    fn optimize_drift(
        &mut self,
        cfg: &mut DriftConfig,
        opt_cfg: &DriftOptimizationConfig,
    ) -> Option<Vec<u8>> {
        let mut total_score: f32 = 0.0;
        let mut min_score: f32 = f32::MAX;
        let mut max_score: f32 = f32::MIN;
        let mut count: f32 = 0.0;

        for (_, node) in &self.nodes {
            total_score += node.score;
            if node.score < min_score {
                min_score = node.score;
            }
            if node.score > max_score {
                max_score = node.score;
            }
            count += 1.0;
        }

        let avg_score = if count > 0.0 { total_score / count } else { 0.0 };
        let score_span = if max_score > min_score {
            max_score - min_score
        } else {
            0.0
        };

        if score_span > opt_cfg.max_allowed_score_drop {
            cfg.decay_rate = (cfg.decay_rate * 0.9).max(opt_cfg.min_decay_rate);
        } else if score_span < opt_cfg.max_allowed_score_drop * 0.5 {
            cfg.decay_rate = (cfg.decay_rate * 1.05).min(opt_cfg.max_decay_rate);
        }

        if score_span > opt_cfg.max_allowed_score_drop {
            cfg.edge_decay_rate =
                (cfg.edge_decay_rate * 0.9).max(opt_cfg.min_edge_decay_rate);
        } else if score_span < opt_cfg.max_allowed_score_drop * 0.5 {
            cfg.edge_decay_rate =
                (cfg.edge_decay_rate * 1.05).min(opt_cfg.max_edge_decay_rate);
        }

        if avg_score < 0.3 {
            cfg.reinforcement_amount =
                (cfg.reinforcement_amount * 1.1).min(opt_cfg.max_reinforcement_amount);
        } else if avg_score > 0.7 {
            cfg.reinforcement_amount =
                (cfg.reinforcement_amount * 0.9).max(opt_cfg.min_reinforcement_amount);
        }

        let scratch = build_node_drift_scratch_pad(self);

        self.compressor.as_ref().map(|c| {
            c.compress(&(
                "optimize_drift",
                cfg.decay_rate,
                cfg.edge_decay_rate,
                cfg.reinforcement_amount,
                avg_score,
                score_span,
                &scratch.layer_a,
                &scratch.layer_b,
            ))
        })
    }
}



