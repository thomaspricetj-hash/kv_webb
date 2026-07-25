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
//! Max-tier upgrade:
//! - Cross-link grids (nodes/edges → drift factors)
//! - Revolving-door routing over drift zones (entry/exit + flow)
//! - Fusion field over drift (scores + drift + door flow)
//! - Embedded Roundabout logic (drift + predictor + smoothing + memory + solver)
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

/// Cross-link grid over drift.
/// Links nodes and edges to normalized drift factors for routing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftCrossLinkGrid {
    pub node_indices: Vec<WebNodeId>,
    pub node_drift_norm: Vec<f32>,
    pub edge_indices: Vec<(WebNodeId, WebNodeId)>,
    pub edge_drift_norm: Vec<f32>,
}

/// Revolving door over drift zones.
/// Entry/exit node sets + flow scalar.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftRevolvingDoor {
    pub door_id: usize,
    pub entry_nodes: Vec<WebNodeId>,
    pub exit_nodes: Vec<WebNodeId>,
    pub flow_strength: f32,
}

/// Fusion field over drift.
/// Combines scores, drift factors, and door flow into a single bias.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftFusionField {
    pub fused_node_bias: Vec<f32>,
}

/// Roundabout drift predictor configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundaboutDriftPredictorConfig {
    pub passes: usize,
    pub min_bias: f32,
    pub max_bias: f32,
    pub smoothing_strength: f32,
}

/// Roundabout drift chain (a routed path through node IDs).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundaboutDriftChain {
    pub nodes: Vec<WebNodeId>,
    pub total_bias: f32,
}

/// Roundabout drift pattern memory entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundaboutDriftPattern {
    pub chain: RoundaboutDriftChain,
    pub weight: f32,
}

/// Roundabout drift pattern memory with decay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundaboutDriftPatternMemory {
    pub patterns: Vec<RoundaboutDriftPattern>,
    pub decay: f32,
}

/// Roundabout drift solver result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundaboutDriftSolverResult {
    pub chosen_node: WebNodeId,
    pub bias: f32,
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

/// Build cross-link grid from node/edge drift scratch pads.
fn build_drift_cross_link_grid(web: &KvWeb) -> DriftCrossLinkGrid {
    let node_scratch = build_node_drift_scratch_pad(web);
    let edge_scratch = build_edge_drift_scratch_pad(web);

    let node_indices: Vec<WebNodeId> = web.nodes.keys().cloned().collect();
    let mut edge_indices = Vec::with_capacity(web.edges.len());
    for e in &web.edges {
        edge_indices.push((e.from, e.to));
    }

    DriftCrossLinkGrid {
        node_indices,
        node_drift_norm: node_scratch.layer_b,
        edge_indices,
        edge_drift_norm: edge_scratch.layer_b,
    }
}

/// Build revolving doors over drift.
/// Simple heuristic: lowest-drift nodes as entry, highest-drift nodes as exit.
fn build_drift_revolving_doors(
    web: &KvWeb,
    grid: &DriftCrossLinkGrid,
) -> Vec<DriftRevolvingDoor> {
    let mut doors = Vec::new();
    if grid.node_indices.is_empty() {
        return doors;
    }

    // Sort indices by normalized drift.
    let mut idxs: Vec<usize> = (0..grid.node_indices.len()).collect();
    idxs.sort_by(|&a, &b| {
        grid.node_drift_norm[a]
            .partial_cmp(&grid.node_drift_norm[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let entry_count = (idxs.len() as f32 * 0.1).ceil() as usize;
    let exit_count = entry_count;

    let entry_nodes: Vec<WebNodeId> = idxs
        .iter()
        .take(entry_count)
        .map(|i| grid.node_indices[*i])
        .collect();

    let exit_nodes: Vec<WebNodeId> = idxs
        .iter()
        .rev()
        .take(exit_count)
        .map(|i| grid.node_indices[*i])
        .collect();

    // Flow strength: difference between avg exit drift and avg entry drift.
    let mut entry_avg = 0.0f32;
    let mut exit_avg = 0.0f32;

    for n in &entry_nodes {
        if let Some(pos) = grid.node_indices.iter().position(|id| id == n) {
            entry_avg += grid.node_drift_norm[pos];
        }
    }
    for n in &exit_nodes {
        if let Some(pos) = grid.node_indices.iter().position(|id| id == n) {
            exit_avg += grid.node_drift_norm[pos];
        }
    }

    if !entry_nodes.is_empty() {
        entry_avg /= entry_nodes.len() as f32;
    }
    if !exit_nodes.is_empty() {
        exit_avg /= exit_nodes.len() as f32;
    }

    let flow_strength = (exit_avg - entry_avg).abs();

    doors.push(DriftRevolvingDoor {
        door_id: 0,
        entry_nodes,
        exit_nodes,
        flow_strength,
    });

    doors
}

/// Build fusion field over drift from scores + drift + doors.
fn build_drift_fusion_field(
    web: &KvWeb,
    grid: &DriftCrossLinkGrid,
    doors: &[DriftRevolvingDoor],
) -> DriftFusionField {
    let mut fused = Vec::with_capacity(grid.node_indices.len());

    for (idx, id) in grid.node_indices.iter().enumerate() {
        let mut bias = if let Some(node) = web.nodes.get(id) {
            node.score
        } else {
            0.0
        };

        // Drift contribution: higher normalized drift → lower bias.
        let drift = grid.node_drift_norm[idx];
        bias *= (1.0 - drift * 0.5).max(0.0);

        // Door flow contribution: if node is exit, boost; if entry, slight damp.
        for door in doors {
            if door.exit_nodes.contains(id) {
                bias *= 1.0 + door.flow_strength * 0.1;
            } else if door.entry_nodes.contains(id) {
                bias *= 1.0 - door.flow_strength * 0.05;
            }
        }

        fused.push(bias);
    }

    // Normalize fused bias.
    let mut max = 0.0f32;
    for v in &fused {
        if *v > max {
            max = *v;
        }
    }
    if max > 0.0 {
        for v in &mut fused {
            *v /= max;
        }
    }

    DriftFusionField {
        fused_node_bias: fused,
    }
}

/// Run roundabout drift predictor: multi-pass chain over fused bias.
fn run_roundabout_drift_predictor(
    grid: &DriftCrossLinkGrid,
    fusion: &DriftFusionField,
    cfg: &RoundaboutDriftPredictorConfig,
) -> RoundaboutDriftChain {
    let mut nodes = Vec::new();
    let mut total = 0.0f32;

    let mut visited = vec![false; fusion.fused_node_bias.len()];

    for _pass in 0..cfg.passes {
        let mut best_idx = None;
        let mut best_bias = cfg.min_bias;

        for (i, b) in fusion.fused_node_bias.iter().enumerate() {
            if visited[i] {
                continue;
            }
            if *b > best_bias && *b <= cfg.max_bias {
                best_bias = *b;
                best_idx = Some(i);
            }
        }

        if let Some(idx) = best_idx {
            visited[idx] = true;
            nodes.push(grid.node_indices[idx]);
            total += best_bias;
        } else {
            break;
        }
    }

    RoundaboutDriftChain {
        nodes,
        total_bias: total,
    }
}

/// Smooth roundabout drift chain by local averaging over fused bias.
fn smooth_roundabout_drift_chain(
    grid: &DriftCrossLinkGrid,
    chain: &mut RoundaboutDriftChain,
    fusion: &DriftFusionField,
    strength: f32,
) {
    if chain.nodes.len() < 3 {
        return;
    }

    let mut new_total = 0.0f32;

    for (i, id) in chain.nodes.iter().enumerate() {
        let mut local_sum = 0.0f32;
        let mut local_count = 0.0f32;

        for j in i.saturating_sub(1)..=(i + 1).min(chain.nodes.len() - 1) {
            let nid = chain.nodes[j];
            if let Some(pos) = grid.node_indices.iter().position(|x| *x == nid) {
                local_sum += fusion.fused_node_bias[pos];
                local_count += 1.0;
            }
        }

        if local_count > 0.0 {
            let avg = local_sum / local_count;
            let base = if let Some(pos) = grid.node_indices.iter().position(|x| *x == *id) {
                fusion.fused_node_bias[pos]
            } else {
                0.0
            };
            new_total += avg * strength + base * (1.0 - strength);
        }
    }

    chain.total_bias = new_total;
}

/// Update drift pattern memory with new chain, applying decay.
fn update_roundabout_drift_pattern_memory(
    memory: &mut RoundaboutDriftPatternMemory,
    chain: &RoundaboutDriftChain,
) {
    for pattern in &mut memory.patterns {
        pattern.weight *= memory.decay;
    }

    memory.patterns.push(RoundaboutDriftPattern {
        chain: chain.clone(),
        weight: 1.0,
    });

    memory.patterns.retain(|p| p.weight > 0.01);
}

/// Apply roundabout bias to fused drift using pattern memory.
fn apply_roundabout_drift_bias(
    grid: &DriftCrossLinkGrid,
    fusion: &mut DriftFusionField,
    memory: &RoundaboutDriftPatternMemory,
) {
    let mut fused = fusion.fused_node_bias.clone();

    for pattern in &memory.patterns {
        let boost = pattern.weight * 0.05;
        for id in &pattern.chain.nodes {
            if let Some(pos) = grid.node_indices.iter().position(|x| x == id) {
                fused[pos] *= 1.0 + boost;
            }
        }
    }

    // Normalize fused bias.
    let mut max = 0.0f32;
    for v in &fused {
        if *v > max {
            max = *v;
        }
    }
    if max > 0.0 {
        for v in &mut fused {
            *v /= max;
        }
    }

    fusion.fused_node_bias = fused;
}

/// Run roundabout drift solver: choose final node using fused bias + chain + memory.
fn run_roundabout_drift_solver(
    grid: &DriftCrossLinkGrid,
    fusion: &DriftFusionField,
    chain: &RoundaboutDriftChain,
    memory: &RoundaboutDriftPatternMemory,
) -> RoundaboutDriftSolverResult {
    // Prefer last node in chain if available.
    if let Some(&last) = chain.nodes.last() {
        let bias = if let Some(pos) = grid.node_indices.iter().position(|x| *x == last) {
            fusion.fused_node_bias.get(pos).copied().unwrap_or(0.0)
        } else {
            0.0
        };
        return RoundaboutDriftSolverResult {
            chosen_node: last,
            bias,
        };
    }

    // Fallback: choose max fused bias node.
    let mut best_pos = 0usize;
    let mut best_bias = f32::MIN;

    for (i, b) in fusion.fused_node_bias.iter().enumerate() {
        if *b > best_bias {
            best_bias = *b;
            best_pos = i;
        }
    }

    let mut final_bias = best_bias;
    let chosen_node = grid.node_indices[best_pos];

    // Light bias from memory: if any pattern contains chosen_node, boost bias slightly.
    for pattern in &memory.patterns {
        if pattern.chain.nodes.contains(&chosen_node) {
            final_bias *= 1.05;
        }
    }

    RoundaboutDriftSolverResult {
        chosen_node,
        bias: final_bias,
    }
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

    /// Max-tier roundabout drift pipeline (GPU-ready).
    fn roundabout_drift_pipeline_compressed(
        &mut self,
        cfg: &DriftConfig,
        predictor_cfg: &RoundaboutDriftPredictorConfig,
        memory: &mut RoundaboutDriftPatternMemory,
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

    /// Compressed roundabout drift pipeline: drift + cross-link + doors + fusion + predictor + smoothing + memory + solver.
    fn roundabout_drift_pipeline_compressed(
        &mut self,
        cfg: &DriftConfig,
        predictor_cfg: &RoundaboutDriftPredictorConfig,
        memory: &mut RoundaboutDriftPatternMemory,
    ) -> Option<Vec<u8>> {
        // Apply node + edge drift first (physics step).
        let _ = self.apply_drift(cfg);
        let _ = self.apply_edge_drift(cfg);

        // Cross-link + doors + fusion.
        let grid = build_drift_cross_link_grid(self);
        let doors = build_drift_revolving_doors(self, &grid);
        let mut fusion = build_drift_fusion_field(self, &grid, &doors);

        // Predictor.
        let mut chain = run_roundabout_drift_predictor(&grid, &fusion, predictor_cfg);

        // Smoothing.
        smooth_roundabout_drift_chain(&grid, &mut chain, &fusion, predictor_cfg.smoothing_strength);

        // Memory update + bias.
        update_roundabout_drift_pattern_memory(memory, &chain);
        apply_roundabout_drift_bias(&grid, &mut fusion, memory);

        // Solver.
        let result = run_roundabout_drift_solver(&grid, &fusion, &chain, memory);

        self.compressor.as_ref().map(|c| {
            c.compress(&(
                "roundabout_drift_pipeline",
                cfg.decay_rate,
                cfg.edge_decay_rate,
                cfg.reinforcement_amount,
                predictor_cfg.passes,
                &grid.node_indices,
                &grid.node_drift_norm,
                &fusion.fused_node_bias,
                &chain.nodes,
                chain.total_bias,
                &memory.patterns,
                result.chosen_node,
                result.bias,
            ))
        })
    }
}

