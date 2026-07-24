//! pruning.rs
//! Polygon-aware pruning physics for KV Web + BitDrop_v2 max-tier compression.
//!
//! Tier-6 upgrades:
//! - parallel bias computation
//! - safe score decay + node pruning
//! - dual-layer pruning scratch pads (scores + weights)
//! - GPU-ready compressed pruning packets
//!
//! Max-tier upgrades:
//! - cross-link grids over pruning bias
//! - revolving-door routing over pruned vs kept nodes
//! - fusion field over pruning geometry (scores + weights + door flow)
//! - embedded Roundabout logic (predictor + smoothing + memory + solver)
//!
//! All upgrades are backwards-compatible.

use kv_web_core::{KvWeb, WebNodeId};
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use std::collections::{HashSet, HashMap};

/// Configuration for pruning physics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningConfig {
    pub score_decay: f32,
    pub min_node_score: f32,
    pub min_edge_weight: f32,

    pub face_bonus: f32,
    pub centroid_penalty: f32,
    pub radius_factor: f32,
}

/// Optimization configuration for pruning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningOptimizationConfig {
    pub min_score_decay: f32,
    pub max_score_decay: f32,

    pub min_node_score: f32,
    pub max_node_score: f32,

    pub min_edge_weight: f32,
    pub max_edge_weight: f32,

    pub target_pruned_ratio: f32,
    pub max_pruned_ratio: f32,
}

/// Dual-layer scratch pad for pruning.
/// Layer A = node scores or edge weights
/// Layer B = normalized metric (0..1) for GPU routing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningScratchPad {
    pub layer_a: Vec<f32>,
    pub layer_b: Vec<f32>,
}

/// Build node pruning scratch pad.
fn build_node_pruning_scratch_pad(web: &KvWeb) -> PruningScratchPad {
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

    PruningScratchPad { layer_a, layer_b }
}

/// Build edge pruning scratch pad.
fn build_edge_pruning_scratch_pad(web: &KvWeb) -> PruningScratchPad {
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

    PruningScratchPad { layer_a, layer_b }
}

/// Polygon-aware score modifier.
fn polygon_prune_bias(web: &KvWeb, node_id: WebNodeId, base: f32, cfg: &PruningConfig) -> f32 {
    let node = match web.nodes.get(&node_id) {
        Some(n) => n,
        None => return base,
    };

    let poly = match &node.polygon {
        Some(p) => p,
        None => return base,
    };

    let face_bonus = match poly.face_index {
        3 => cfg.face_bonus,
        2 => cfg.face_bonus * 0.6,
        1 => cfg.face_bonus * 0.3,
        _ => 0.0,
    };

    let mut centroid_mag = 0.0;
    for v in &poly.centroid {
        centroid_mag += v.abs();
    }

    let centroid_penalty =
        f32::min(centroid_mag / (poly.radius + 1.0), 0.25) * cfg.centroid_penalty;

    let radius_mult = 1.0 / (poly.radius + cfg.radius_factor);

    base + face_bonus - centroid_penalty * radius_mult
}

/// Cross-link grid over pruning bias.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningCrossLinkGrid {
    pub nodes: Vec<WebNodeId>,
    pub scores: Vec<f32>,
    pub geom_bias: Vec<f32>,
}

/// Revolving door over pruned vs kept nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningRevolvingDoor {
    pub door_id: usize,
    pub kept_nodes: Vec<WebNodeId>,
    pub pruned_nodes: Vec<WebNodeId>,
    pub flow_strength: f32,
}

/// Fusion field over pruning geometry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningFusionField {
    pub fused_node_bias: Vec<f32>,
}

/// Roundabout pruning predictor configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundaboutPruningPredictorConfig {
    pub passes: usize,
    pub min_bias: f32,
    pub max_bias: f32,
    pub smoothing_strength: f32,
}

/// Roundabout pruning chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundaboutPruningChain {
    pub nodes: Vec<WebNodeId>,
    pub total_bias: f32,
}

/// Roundabout pruning pattern memory entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundaboutPruningPattern {
    pub chain: RoundaboutPruningChain,
    pub weight: f32,
}

/// Roundabout pruning pattern memory with decay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundaboutPruningPatternMemory {
    pub patterns: Vec<RoundaboutPruningPattern>,
    pub decay: f32,
}

/// Roundabout pruning solver result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundaboutPruningSolverResult {
    pub chosen_node: WebNodeId,
    pub bias: f32,
}

/// Extension trait for pruning physics.
pub trait KvWebPruning {
    fn decay_scores(&mut self, cfg: &PruningConfig);
    fn prune_nodes(&mut self, cfg: &PruningConfig);
    fn prune_edges(&mut self, cfg: &PruningConfig);
    fn cleanup_orphan_tokens(&mut self);

    fn optimize_pruning(
        &mut self,
        cfg: &mut PruningConfig,
        opt_cfg: &PruningOptimizationConfig,
    ) -> Option<Vec<u8>>;
}

impl KvWebPruning for KvWeb {
    /// Safe polygon-aware score decay: bias computed in parallel, applied sequentially.
    fn decay_scores(&mut self, cfg: &PruningConfig) {
        let ids: Vec<WebNodeId> = self.nodes.keys().cloned().collect();

        let biased_scores: HashMap<WebNodeId, f32> = ids
            .par_iter()
            .filter_map(|id| {
                let base = match self.nodes.get(id) {
                    Some(n) => n.score - cfg.score_decay,
                    None => return None,
                };

                let biased = polygon_prune_bias(self, *id, base, cfg);
                Some((*id, biased.max(0.0)))
            })
            .collect();

        for (id, score) in biased_scores {
            if let Some(node) = self.nodes.get_mut(&id) {
                node.score = score;
            }
        }

        if let Some(comp) = &self.compressor {
            let scratch = build_node_pruning_scratch_pad(self);
            let _ = comp.compress(&(
                "decay_scores",
                cfg.score_decay,
                &scratch.layer_a,
                &scratch.layer_b,
            ));
        }
    }

    /// Parallel polygon-aware node pruning (bias map computed in parallel, retain sequential).
    fn prune_nodes(&mut self, cfg: &PruningConfig) {
        let ids: Vec<WebNodeId> = self.nodes.keys().cloned().collect();

        let bias_map: HashMap<WebNodeId, f32> = ids
            .par_iter()
            .filter_map(|id| {
                if let Some(node) = self.nodes.get(id) {
                    let biased = polygon_prune_bias(self, *id, node.score, cfg);
                    Some((*id, biased))
                } else {
                    None
                }
            })
            .collect();

        let mut removed = HashSet::new();

        self.nodes.retain(|id, _node| {
            let biased_score = bias_map[id];
            let keep = biased_score >= cfg.min_node_score;
            if !keep {
                removed.insert(*id);
            }
            keep
        });

        self.edges.retain(|e| !removed.contains(&e.to));

        self.node_index_by_token.clear();
        for (id, node) in &self.nodes {
            for t in &node.tokens {
                self.node_index_by_token.insert(*t, *id);
            }
        }

        if let Some(comp) = &self.compressor {
            let scratch = build_node_pruning_scratch_pad(self);
            let _ = comp.compress(&(
                "prune_nodes",
                cfg.min_node_score,
                &scratch.layer_a,
                &scratch.layer_b,
            ));
        }
    }

    /// Parallel edge pruning via marking + retain.
    fn prune_edges(&mut self, cfg: &PruningConfig) {
        self.edges
            .par_iter_mut()
            .for_each(|e| {
                if e.weight < cfg.min_edge_weight {
                    e.weight = -1.0;
                }
            });

        self.edges.retain(|e| e.weight >= cfg.min_edge_weight);

        if let Some(comp) = &self.compressor {
            let scratch = build_edge_pruning_scratch_pad(self);
            let _ = comp.compress(&(
                "prune_edges",
                cfg.min_edge_weight,
                &scratch.layer_a,
                &scratch.layer_b,
            ));
        }
    }

    fn cleanup_orphan_tokens(&mut self) {
        self.node_index_by_token
            .retain(|_, node_id| self.nodes.contains_key(node_id));

        if let Some(comp) = &self.compressor {
            let _ = comp.compress(&("cleanup_orphan_tokens", self.node_index_by_token.len()));
        }
    }

    fn optimize_pruning(
        &mut self,
        cfg: &mut PruningConfig,
        opt_cfg: &PruningOptimizationConfig,
    ) -> Option<Vec<u8>> {
        let total_nodes = self.nodes.len() as f32;
        if total_nodes == 0.0 {
            return None;
        }

        let mut below_threshold = 0.0;
        for (_, node) in &self.nodes {
            if node.score < cfg.min_node_score {
                below_threshold += 1.0;
            }
        }

        let pruned_ratio = below_threshold / total_nodes;

        if pruned_ratio < opt_cfg.target_pruned_ratio {
            cfg.score_decay =
                (cfg.score_decay * 1.05).min(opt_cfg.max_score_decay);
        } else if pruned_ratio > opt_cfg.max_pruned_ratio {
            cfg.score_decay =
                (cfg.score_decay * 0.9).max(opt_cfg.min_score_decay);
        }

        if pruned_ratio > opt_cfg.max_pruned_ratio {
            cfg.min_node_score =
                (cfg.min_node_score * 0.9).max(opt_cfg.min_node_score);
        } else if pruned_ratio < opt_cfg.target_pruned_ratio {
            cfg.min_node_score =
                (cfg.min_node_score * 1.05).min(opt_cfg.max_node_score);
        }

        let mut low_edges = 0.0;
        for e in &self.edges {
            if e.weight < cfg.min_edge_weight {
                low_edges += 1.0;
            }
        }

        let edge_ratio = if self.edges.is_empty() {
            0.0
        } else {
            low_edges / self.edges.len() as f32
        };

        if edge_ratio > 0.5 {
            cfg.min_edge_weight =
                (cfg.min_edge_weight * 0.9).max(opt_cfg.min_edge_weight);
        } else {
            cfg.min_edge_weight =
                (cfg.min_edge_weight * 1.05).min(opt_cfg.max_edge_weight);
        }

        let scratch_nodes = build_node_pruning_scratch_pad(self);
        let scratch_edges = build_edge_pruning_scratch_pad(self);

        self.compressor.as_ref().map(|c| {
            c.compress(&(
                "optimize_pruning",
                pruned_ratio,
                edge_ratio,
                cfg.score_decay,
                cfg.min_node_score,
                cfg.min_edge_weight,
                &scratch_nodes.layer_a,
                &scratch_nodes.layer_b,
                &scratch_edges.layer_a,
                &scratch_edges.layer_b,
            ))
        })
    }
}

/// Build cross-link grid from current pruning state.
fn build_pruning_cross_link_grid(
    web: &KvWeb,
    cfg: &PruningConfig,
) -> PruningCrossLinkGrid {
    let mut nodes = Vec::with_capacity(web.nodes.len());
    let mut scores = Vec::with_capacity(web.nodes.len());
    let mut geom_bias = Vec::with_capacity(web.nodes.len());

    for (id, node) in &web.nodes {
        nodes.push(*id);
        scores.push(node.score);
        let biased = polygon_prune_bias(web, *id, node.score, cfg);
        geom_bias.push(biased);
    }

    PruningCrossLinkGrid {
        nodes,
        scores,
        geom_bias,
    }
}

/// Build revolving doors over pruning state.
fn build_pruning_revolving_doors(
    grid: &PruningCrossLinkGrid,
    cfg: &PruningConfig,
) -> Vec<PruningRevolvingDoor> {
    let mut doors = Vec::new();
    if grid.nodes.is_empty() {
        return doors;
    }

    let mut idxs: Vec<usize> = (0..grid.nodes.len()).collect();
    idxs.sort_by(|&a, &b| {
        grid.scores[b]
            .partial_cmp(&grid.scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let cutoff = cfg.min_node_score;
    let kept_nodes: Vec<WebNodeId> = idxs
        .iter()
        .filter(|&&i| grid.scores[i] >= cutoff)
        .map(|i| grid.nodes[*i])
        .collect();

    let pruned_nodes: Vec<WebNodeId> = idxs
        .iter()
        .filter(|&&i| grid.scores[i] < cutoff)
        .map(|i| grid.nodes[*i])
        .collect();

    let mut kept_avg = 0.0f32;
    let mut pruned_avg = 0.0f32;

    for n in &kept_nodes {
        if let Some(pos) = grid.nodes.iter().position(|id| id == n) {
            kept_avg += grid.scores[pos];
        }
    }
    for n in &pruned_nodes {
        if let Some(pos) = grid.nodes.iter().position(|id| id == n) {
            pruned_avg += grid.scores[pos];
        }
    }

    if !kept_nodes.is_empty() {
        kept_avg /= kept_nodes.len() as f32;
    }
    if !pruned_nodes.is_empty() {
        pruned_avg /= pruned_nodes.len() as f32;
    }

    let flow_strength = (kept_avg - pruned_avg).abs();

    doors.push(PruningRevolvingDoor {
        door_id: 0,
        kept_nodes,
        pruned_nodes,
        flow_strength,
    });

    doors
}

/// Build fusion field over pruning geometry.
fn build_pruning_fusion_field(
    grid: &PruningCrossLinkGrid,
    doors: &[PruningRevolvingDoor],
) -> PruningFusionField {
    let mut fused = Vec::with_capacity(grid.nodes.len());

    for (idx, id) in grid.nodes.iter().enumerate() {
        let mut bias = grid.scores[idx];

        let geom = grid.geom_bias[idx];
        bias *= (0.5 + geom * 0.5);

        for door in doors {
            if door.kept_nodes.contains(id) {
                bias *= 1.0 + door.flow_strength * 0.1;
            } else if door.pruned_nodes.contains(id) {
                bias *= 1.0 - door.flow_strength * 0.05;
            }
        }

        fused.push(bias);
    }

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

    PruningFusionField {
        fused_node_bias: fused,
    }
}

/// Run roundabout pruning predictor.
fn run_roundabout_pruning_predictor(
    grid: &PruningCrossLinkGrid,
    fusion: &PruningFusionField,
    cfg: &RoundaboutPruningPredictorConfig,
) -> RoundaboutPruningChain {
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
            nodes.push(grid.nodes[idx]);
            total += best_bias;
        } else {
            break;
        }
    }

    RoundaboutPruningChain {
        nodes,
        total_bias: total,
    }
}

/// Smooth roundabout pruning chain.
fn smooth_roundabout_pruning_chain(
    grid: &PruningCrossLinkGrid,
    chain: &mut RoundaboutPruningChain,
    fusion: &PruningFusionField,
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
            if let Some(pos) = grid.nodes.iter().position(|x| *x == nid) {
                local_sum += fusion.fused_node_bias[pos];
                local_count += 1.0;
            }
        }

        if local_count > 0.0 {
            let avg = local_sum / local_count;
            let base = if let Some(pos) = grid.nodes.iter().position(|x| *x == *id) {
                fusion.fused_node_bias[pos]
            } else {
                0.0
            };
            new_total += avg * strength + base * (1.0 - strength);
        }
    }

    chain.total_bias = new_total;
}

/// Update pruning pattern memory.
fn update_roundabout_pruning_pattern_memory(
    memory: &mut RoundaboutPruningPatternMemory,
    chain: &RoundaboutPruningChain,
) {
    for pattern in &mut memory.patterns {
        pattern.weight *= memory.decay;
    }

    memory.patterns.push(RoundaboutPruningPattern {
        chain: chain.clone(),
        weight: 1.0,
    });

    memory.patterns.retain(|p| p.weight > 0.01);
}

/// Apply roundabout bias to fused pruning geometry.
fn apply_roundabout_pruning_bias(
    grid: &PruningCrossLinkGrid,
    fusion: &mut PruningFusionField,
    memory: &RoundaboutPruningPatternMemory,
) {
    let mut fused = fusion.fused_node_bias.clone();

    for pattern in &memory.patterns {
        let boost = pattern.weight * 0.05;
        for id in &pattern.chain.nodes {
            if let Some(pos) = grid.nodes.iter().position(|x| x == id) {
                fused[pos] *= 1.0 + boost;
            }
        }
    }

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

/// Run roundabout pruning solver.
fn run_roundabout_pruning_solver(
    grid: &PruningCrossLinkGrid,
    fusion: &PruningFusionField,
    chain: &RoundaboutPruningChain,
    memory: &RoundaboutPruningPatternMemory,
) -> RoundaboutPruningSolverResult {
    if let Some(&last) = chain.nodes.last() {
        let bias = if let Some(pos) = grid.nodes.iter().position(|x| *x == last) {
            fusion.fused_node_bias.get(pos).copied().unwrap_or(0.0)
        } else {
            0.0
        };
        return RoundaboutPruningSolverResult {
            chosen_node: last,
            bias,
        };
    }

    let mut best_pos = 0usize;
    let mut best_bias = f32::MIN;

    for (i, b) in fusion.fused_node_bias.iter().enumerate() {
        if *b > best_bias {
            best_bias = *b;
            best_pos = i;
        }
    }

    let mut final_bias = best_bias;
    let chosen_node = grid.nodes[best_pos];

    for pattern in &memory.patterns {
        if pattern.chain.nodes.contains(&chosen_node) {
            final_bias *= 1.05;
        }
    }

    RoundaboutPruningSolverResult {
        chosen_node,
        bias: final_bias,
    }
}

/// Packet type for compressed roundabout pruning pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningRoundaboutPipelinePacket {
    pub tag: &'static str,
    pub pruned_ratio: f32,
    pub edge_ratio: f32,
    pub score_decay: f32,
    pub min_node_score: f32,
    pub min_edge_weight: f32,
    pub grid_nodes: Vec<WebNodeId>,
    pub grid_scores: Vec<f32>,
    pub grid_geom_bias: Vec<f32>,
    pub fused_node_bias: Vec<f32>,
    pub chain_nodes: Vec<WebNodeId>,
    pub chain_total_bias: f32,
    pub patterns: Vec<RoundaboutPruningPattern>,
    pub chosen_node: WebNodeId,
    pub chosen_bias: f32,
}

/// Compressed roundabout pruning pipeline (GPU-ready).
pub fn pruning_roundabout_pipeline_compressed(
    web: &KvWeb,
    cfg: &PruningConfig,
    opt_cfg: &PruningOptimizationConfig,
    predictor_cfg: &RoundaboutPruningPredictorConfig,
    memory: &mut RoundaboutPruningPatternMemory,
) -> Option<Vec<u8>> {
    let total_nodes = web.nodes.len() as f32;
    if total_nodes == 0.0 {
        return None;
    }

    let mut below_threshold = 0.0;
    for (_, node) in &web.nodes {
        if node.score < cfg.min_node_score {
            below_threshold += 1.0;
        }
    }
    let pruned_ratio = below_threshold / total_nodes;

    let mut low_edges = 0.0;
    for e in &web.edges {
        if e.weight < cfg.min_edge_weight {
            low_edges += 1.0;
        }
    }
    let edge_ratio = if web.edges.is_empty() {
        0.0
    } else {
        low_edges / web.edges.len() as f32
    };

    let grid = build_pruning_cross_link_grid(web, cfg);
    let doors = build_pruning_revolving_doors(&grid, cfg);
    let mut fusion = build_pruning_fusion_field(&grid, &doors);

    let mut chain = run_roundabout_pruning_predictor(&grid, &fusion, predictor_cfg);
    smooth_roundabout_pruning_chain(&grid, &mut chain, &fusion, predictor_cfg.smoothing_strength);

    update_roundabout_pruning_pattern_memory(memory, &chain);
    apply_roundabout_pruning_bias(&grid, &mut fusion, memory);

    let result = run_roundabout_pruning_solver(&grid, &fusion, &chain, memory);

    let packet = PruningRoundaboutPipelinePacket {
        tag: "pruning_roundabout_pipeline",
        pruned_ratio,
        edge_ratio,
        score_decay: cfg.score_decay,
        min_node_score: cfg.min_node_score,
        min_edge_weight: cfg.min_edge_weight,
        grid_nodes: grid.nodes.clone(),
        grid_scores: grid.scores.clone(),
        grid_geom_bias: grid.geom_bias.clone(),
        fused_node_bias: fusion.fused_node_bias.clone(),
        chain_nodes: chain.nodes.clone(),
        chain_total_bias: chain.total_bias,
        patterns: memory.patterns.clone(),
        chosen_node: result.chosen_node,
        chosen_bias: result.bias,
    };

    web.compressor.as_ref().map(|c| c.compress(&packet))
}

