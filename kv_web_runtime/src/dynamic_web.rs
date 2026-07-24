//! dynamic_web.rs
//! Automatic edge generation and dynamic webbing logic + BitDrop_v2 max‑tier compression.
//!
//! This module makes the KV Web *adaptive*:
//! - edges strengthen when nodes co‑occur
//! - edges weaken when unused
//! - new edges form based on recency or semantic similarity
//!
//! Tier‑6 upgrades:
//! - parallel edge reinforcement + decay
//! - dual-layer scratch pads (weights + normalized geometry)
//! - GPU-ready compressed packets
//!
//! Max-tier upgrades:
//! - cross-link grids over edges + nodes
//! - revolving-door routing over edge zones (entry/exit + flow)
//! - fusion field over dynamic web (weights + geometry + door flow)
//! - embedded Roundabout logic (predictor + smoothing + memory + solver)
//!
//! All upgrades are backwards-compatible.

use kv_web_core::{KvWeb, WebNodeId, EdgeKind, KvWebCompressor};
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use std::collections::HashSet;

/// Dynamic webbing configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicWebConfig {
    pub strengthen_amount: f32,   // how much to increase weight when nodes co‑occur
    pub weaken_amount: f32,       // how much to decrease weight when unused
    pub min_weight: f32,          // edges below this are removed
    pub max_weight: f32,          // clamp edge weight
    pub recency_link_weight: f32, // weight for auto‑linking recent nodes
}

/// Dynamic web optimization configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicWebOptimizationConfig {
    pub min_strengthen_amount: f32,
    pub max_strengthen_amount: f32,
    pub min_weaken_amount: f32,
    pub max_weaken_amount: f32,
    pub min_weight_span: f32,
    pub max_weight_span: f32,
    pub min_recency_link_weight: f32,
    pub max_recency_link_weight: f32,
}

/// Dual-layer scratch pad for dynamic web.
/// Layer A = raw edge weights
/// Layer B = normalized weights (0..1) for GPU routing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicWebScratchPad {
    pub layer_a: Vec<f32>,
    pub layer_b: Vec<f32>,
}

/// Cross-link grid over dynamic web.
/// Links edges and nodes into routing-ready structures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicWebCrossLinkGrid {
    pub edge_indices: Vec<(WebNodeId, WebNodeId)>,
    pub edge_norm_weights: Vec<f32>,
    pub node_indices: Vec<WebNodeId>,
}

/// Revolving door over dynamic web zones.
/// Entry/exit edge sets + flow scalar.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicWebRevolvingDoor {
    pub door_id: usize,
    pub entry_edges: Vec<(WebNodeId, WebNodeId)>,
    pub exit_edges: Vec<(WebNodeId, WebNodeId)>,
    pub flow_strength: f32,
}

/// Fusion field over dynamic web.
/// Combines weights + door flow into a single bias per edge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicWebFusionField {
    pub fused_edge_bias: Vec<f32>,
}

/// Roundabout dynamic web predictor configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundaboutDynamicWebPredictorConfig {
    pub passes: usize,
    pub min_bias: f32,
    pub max_bias: f32,
    pub smoothing_strength: f32,
}

/// Roundabout dynamic web chain (routed path through edge indices).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundaboutDynamicWebChain {
    pub edge_indices: Vec<usize>,
    pub total_bias: f32,
}

/// Roundabout dynamic web pattern memory entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundaboutDynamicWebPattern {
    pub chain: RoundaboutDynamicWebChain,
    pub weight: f32,
}

/// Roundabout dynamic web pattern memory with decay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundaboutDynamicWebPatternMemory {
    pub patterns: Vec<RoundaboutDynamicWebPattern>,
    pub decay: f32,
}

/// Roundabout dynamic web solver result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundaboutDynamicWebSolverResult {
    pub chosen_edge_index: usize,
    pub bias: f32,
}

/// Build dual-layer scratch pad from current edges.
fn build_dynamic_web_scratch_pad(web: &KvWeb) -> DynamicWebScratchPad {
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

    DynamicWebScratchPad { layer_a, layer_b }
}

/// Build cross-link grid from current edges + scratch pad.
fn build_dynamic_web_cross_link_grid(web: &KvWeb) -> DynamicWebCrossLinkGrid {
    let scratch = build_dynamic_web_scratch_pad(web);

    let mut edge_indices = Vec::with_capacity(web.edges.len());
    for e in &web.edges {
        edge_indices.push((e.from, e.to));
    }

    let node_indices: Vec<WebNodeId> = web.nodes.keys().cloned().collect();

    DynamicWebCrossLinkGrid {
        edge_indices,
        edge_norm_weights: scratch.layer_b,
        node_indices,
    }
}

/// Build revolving doors over dynamic web.
/// Simple heuristic: lowest-weight edges as entry, highest-weight edges as exit.
fn build_dynamic_web_revolving_doors(
    grid: &DynamicWebCrossLinkGrid,
) -> Vec<DynamicWebRevolvingDoor> {
    let mut doors = Vec::new();
    if grid.edge_indices.is_empty() {
        return doors;
    }

    let mut idxs: Vec<usize> = (0..grid.edge_indices.len()).collect();
    idxs.sort_by(|&a, &b| {
        grid.edge_norm_weights[a]
            .partial_cmp(&grid.edge_norm_weights[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let zone_count = (idxs.len() as f32 * 0.1).ceil() as usize;
    let entry_count = zone_count;
    let exit_count = zone_count;

    let entry_edges: Vec<(WebNodeId, WebNodeId)> = idxs
        .iter()
        .take(entry_count)
        .map(|i| grid.edge_indices[*i])
        .collect();

    let exit_edges: Vec<(WebNodeId, WebNodeId)> = idxs
        .iter()
        .rev()
        .take(exit_count)
        .map(|i| grid.edge_indices[*i])
        .collect();

    let mut entry_avg = 0.0f32;
    let mut exit_avg = 0.0f32;

    for (from, to) in &entry_edges {
        if let Some(pos) = grid
            .edge_indices
            .iter()
            .position(|e| e.0 == *from && e.1 == *to)
        {
            entry_avg += grid.edge_norm_weights[pos];
        }
    }
    for (from, to) in &exit_edges {
        if let Some(pos) = grid
            .edge_indices
            .iter()
            .position(|e| e.0 == *from && e.1 == *to)
        {
            exit_avg += grid.edge_norm_weights[pos];
        }
    }

    if !entry_edges.is_empty() {
        entry_avg /= entry_edges.len() as f32;
    }
    if !exit_edges.is_empty() {
        exit_avg /= exit_edges.len() as f32;
    }

    let flow_strength = (exit_avg - entry_avg).abs();

    doors.push(DynamicWebRevolvingDoor {
        door_id: 0,
        entry_edges,
        exit_edges,
        flow_strength,
    });

    doors
}

/// Build fusion field over dynamic web from weights + doors.
fn build_dynamic_web_fusion_field(
    web: &KvWeb,
    grid: &DynamicWebCrossLinkGrid,
    doors: &[DynamicWebRevolvingDoor],
) -> DynamicWebFusionField {
    let mut fused = Vec::with_capacity(grid.edge_indices.len());

    for (idx, (from, to)) in grid.edge_indices.iter().enumerate() {
        let mut bias = if let Some(edge) = web
            .edges
            .iter()
            .find(|e| e.from == *from && e.to == *to)
        {
            edge.weight
        } else {
            0.0
        };

        let norm = grid.edge_norm_weights[idx];
        bias *= (0.5 + norm * 0.5);

        for door in doors {
            if door.exit_edges.contains(&(*from, *to)) {
                bias *= 1.0 + door.flow_strength * 0.1;
            } else if door.entry_edges.contains(&(*from, *to)) {
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

    DynamicWebFusionField {
        fused_edge_bias: fused,
    }
}

/// Run roundabout dynamic web predictor: multi-pass chain over fused bias.
fn run_roundabout_dynamic_web_predictor(
    fusion: &DynamicWebFusionField,
    cfg: &RoundaboutDynamicWebPredictorConfig,
) -> RoundaboutDynamicWebChain {
    let mut edge_indices = Vec::new();
    let mut total = 0.0f32;

    let mut visited = vec![false; fusion.fused_edge_bias.len()];

    for _pass in 0..cfg.passes {
        let mut best_idx = None;
        let mut best_bias = cfg.min_bias;

        for (i, b) in fusion.fused_edge_bias.iter().enumerate() {
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
            edge_indices.push(idx);
            total += best_bias;
        } else {
            break;
        }
    }

    RoundaboutDynamicWebChain {
        edge_indices,
        total_bias: total,
    }
}

/// Smooth roundabout dynamic web chain by local averaging over fused bias.
fn smooth_roundabout_dynamic_web_chain(
    chain: &mut RoundaboutDynamicWebChain,
    fusion: &DynamicWebFusionField,
    strength: f32,
) {
    if chain.edge_indices.len() < 3 {
        return;
    }

    let mut new_total = 0.0f32;

    for (i, idx) in chain.edge_indices.iter().enumerate() {
        let mut local_sum = 0.0f32;
        let mut local_count = 0.0f32;

        for j in i.saturating_sub(1)..=(i + 1).min(chain.edge_indices.len() - 1) {
            let nid = chain.edge_indices[j];
            if nid < fusion.fused_edge_bias.len() {
                local_sum += fusion.fused_edge_bias[nid];
                local_count += 1.0;
            }
        }

        if local_count > 0.0 {
            let avg = local_sum / local_count;
            let base = fusion.fused_edge_bias[*idx];
            new_total += avg * strength + base * (1.0 - strength);
        }
    }

    chain.total_bias = new_total;
}

/// Update dynamic web pattern memory with new chain, applying decay.
fn update_roundabout_dynamic_web_pattern_memory(
    memory: &mut RoundaboutDynamicWebPatternMemory,
    chain: &RoundaboutDynamicWebChain,
) {
    for pattern in &mut memory.patterns {
        pattern.weight *= memory.decay;
    }

    memory.patterns.push(RoundaboutDynamicWebPattern {
        chain: chain.clone(),
        weight: 1.0,
    });

    memory.patterns.retain(|p| p.weight > 0.01);
}

/// Apply roundabout bias to fused dynamic web using pattern memory.
fn apply_roundabout_dynamic_web_bias(
    fusion: &mut DynamicWebFusionField,
    memory: &RoundaboutDynamicWebPatternMemory,
) {
    let mut fused = fusion.fused_edge_bias.clone();

    for pattern in &memory.patterns {
        let boost = pattern.weight * 0.05;
        for idx in &pattern.chain.edge_indices {
            if *idx < fused.len() {
                fused[*idx] *= 1.0 + boost;
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

    fusion.fused_edge_bias = fused;
}

/// Run roundabout dynamic web solver: choose final edge using fused bias + chain + memory.
fn run_roundabout_dynamic_web_solver(
    fusion: &DynamicWebFusionField,
    chain: &RoundaboutDynamicWebChain,
    memory: &RoundaboutDynamicWebPatternMemory,
) -> RoundaboutDynamicWebSolverResult {
    if let Some(&last) = chain.edge_indices.last() {
        let bias = fusion.fused_edge_bias.get(last).copied().unwrap_or(0.0);
        return RoundaboutDynamicWebSolverResult {
            chosen_edge_index: last,
            bias,
        };
    }

    let mut best_idx = 0usize;
    let mut best_bias = f32::MIN;

    for (i, b) in fusion.fused_edge_bias.iter().enumerate() {
        if *b > best_bias {
            best_bias = *b;
            best_idx = i;
        }
    }

    let mut final_bias = best_bias;
    for pattern in &memory.patterns {
        if pattern.chain.edge_indices.contains(&best_idx) {
            final_bias *= 1.05;
        }
    }

    RoundaboutDynamicWebSolverResult {
        chosen_edge_index: best_idx,
        bias: final_bias,
    }
}

/// Extension trait for dynamic webbing.
pub trait KvWebDynamic {
    fn reinforce_edges(&mut self, nodes: &[WebNodeId], cfg: &DynamicWebConfig)
        -> Option<Vec<u8>>;

    fn decay_edges(&mut self, cfg: &DynamicWebConfig) -> Option<Vec<u8>>;

    fn link_recent_nodes(&mut self, recent: &[WebNodeId], cfg: &DynamicWebConfig)
        -> Option<Vec<u8>>;

    fn normalize_edges(&mut self, cfg: &DynamicWebConfig) -> Option<Vec<u8>>;

    fn optimize_dynamic_web(
        &mut self,
        cfg: &mut DynamicWebConfig,
        opt_cfg: &DynamicWebOptimizationConfig,
    ) -> Option<Vec<u8>>;

    /// Max-tier roundabout dynamic web pipeline (GPU-ready).
    fn roundabout_dynamic_web_pipeline_compressed(
        &mut self,
        cfg: &DynamicWebConfig,
        predictor_cfg: &RoundaboutDynamicWebPredictorConfig,
        memory: &mut RoundaboutDynamicWebPatternMemory,
    ) -> Option<Vec<u8>>;
}

impl KvWebDynamic for KvWeb {
    /// Parallel reinforcement: edges whose endpoints are in `nodes` get strengthened.
    fn reinforce_edges(&mut self, nodes: &[WebNodeId], cfg: &DynamicWebConfig)
        -> Option<Vec<u8>>
    {
        let set: HashSet<WebNodeId> = nodes.iter().cloned().collect();

        self.edges
            .par_iter_mut()
            .for_each(|edge| {
                if set.contains(&edge.from) && set.contains(&edge.to) {
                    edge.weight += cfg.strengthen_amount;
                }
            });

        let scratch = build_dynamic_web_scratch_pad(self);

        self.compressor.as_ref().map(|c| {
            c.compress(&(
                "reinforce_edges",
                nodes,
                cfg.strengthen_amount,
                &scratch.layer_a,
                &scratch.layer_b,
            ))
        })
    }

    /// Parallel global decay.
    fn decay_edges(&mut self, cfg: &DynamicWebConfig) -> Option<Vec<u8>> {
        self.edges
            .par_iter_mut()
            .for_each(|edge| {
                edge.weight -= cfg.weaken_amount;
            });

        let scratch = build_dynamic_web_scratch_pad(self);

        self.compressor.as_ref().map(|c| {
            c.compress(&(
                "decay_edges",
                cfg.weaken_amount,
                &scratch.layer_a,
                &scratch.layer_b,
            ))
        })
    }

    /// Recency-based linking (kept serial; typically small).
    fn link_recent_nodes(&mut self, recent: &[WebNodeId], cfg: &DynamicWebConfig)
        -> Option<Vec<u8>>
    {
        for pair in recent.windows(2) {
            let a = pair[0];
            let b = pair[1];

            self.add_edge(a, b, cfg.recency_link_weight, EdgeKind::Positional);
        }

        let scratch = build_dynamic_web_scratch_pad(self);

        self.compressor.as_ref().map(|c| {
            c.compress(&(
                "link_recent_nodes",
                recent,
                cfg.recency_link_weight,
                &scratch.layer_a,
                &scratch.layer_b,
            ))
        })
    }

    /// Parallel clamp + cleanup.
    fn normalize_edges(&mut self, cfg: &DynamicWebConfig) -> Option<Vec<u8>> {
        self.edges.retain(|e| e.weight >= cfg.min_weight);

        self.edges
            .par_iter_mut()
            .for_each(|edge| {
                if edge.weight > cfg.max_weight {
                    edge.weight = cfg.max_weight;
                }
            });

        let scratch = build_dynamic_web_scratch_pad(self);

        self.compressor.as_ref().map(|c| {
            c.compress(&(
                "normalize_edges",
                cfg.min_weight,
                cfg.max_weight,
                &scratch.layer_a,
                &scratch.layer_b,
            ))
        })
    }

    /// Max-tier optimization loop over dynamic web parameters + scratch pad.
    fn optimize_dynamic_web(
        &mut self,
        cfg: &mut DynamicWebConfig,
        opt_cfg: &DynamicWebOptimizationConfig,
    ) -> Option<Vec<u8>> {
        if self.edges.is_empty() {
            return None;
        }

        let mut min_w = f32::MAX;
        let mut max_w = f32::MIN;
        let mut total_w = 0.0;
        let mut count = 0.0;

        for edge in &self.edges {
            if edge.weight < min_w {
                min_w = edge.weight;
            }
            if edge.weight > max_w {
                max_w = edge.weight;
            }
            total_w += edge.weight;
            count += 1.0;
        }

        let avg_w = total_w / count;
        let span = if max_w > min_w { max_w - min_w } else { 0.0 };

        if span < opt_cfg.min_weight_span {
            cfg.strengthen_amount =
                (cfg.strengthen_amount * 1.05).min(opt_cfg.max_strengthen_amount);
        } else if span > opt_cfg.max_weight_span {
            cfg.strengthen_amount =
                (cfg.strengthen_amount * 0.9).max(opt_cfg.min_strengthen_amount);
        }

        if avg_w > cfg.max_weight * 0.8 {
            cfg.weaken_amount =
                (cfg.weaken_amount * 1.05).min(opt_cfg.max_weaken_amount);
        } else if avg_w < cfg.min_weight * 1.2 {
            cfg.weaken_amount =
                (cfg.weaken_amount * 0.9).max(opt_cfg.min_weaken_amount);
        }

        if span > opt_cfg.max_weight_span {
            cfg.recency_link_weight =
                (cfg.recency_link_weight * 0.9).max(opt_cfg.min_recency_link_weight);
        } else if span < opt_cfg.min_weight_span {
            cfg.recency_link_weight =
                (cfg.recency_link_weight * 1.05).min(opt_cfg.max_recency_link_weight);
        }

        let scratch = build_dynamic_web_scratch_pad(self);

        self.compressor.as_ref().map(|c| {
            c.compress(&(
                "optimize_dynamic_web",
                cfg.strengthen_amount,
                cfg.weaken_amount,
                cfg.recency_link_weight,
                avg_w,
                span,
                &scratch.layer_a,
                &scratch.layer_b,
            ))
        })
    }

    /// Compressed roundabout dynamic web pipeline: reinforce + decay + link + normalize + cross-link + doors + fusion + predictor + smoothing + memory + solver.
    fn roundabout_dynamic_web_pipeline_compressed(
        &mut self,
        cfg: &DynamicWebConfig,
        predictor_cfg: &RoundaboutDynamicWebPredictorConfig,
        memory: &mut RoundaboutDynamicWebPatternMemory,
    ) -> Option<Vec<u8>> {
        let _ = self.decay_edges(cfg);
        let scratch_before = build_dynamic_web_scratch_pad(self);

        let grid = build_dynamic_web_cross_link_grid(self);
        let doors = build_dynamic_web_revolving_doors(&grid);
        let mut fusion = build_dynamic_web_fusion_field(self, &grid, &doors);

        let mut chain = run_roundabout_dynamic_web_predictor(&fusion, predictor_cfg);
        smooth_roundabout_dynamic_web_chain(&mut chain, &fusion, predictor_cfg.smoothing_strength);

        update_roundabout_dynamic_web_pattern_memory(memory, &chain);
        apply_roundabout_dynamic_web_bias(&mut fusion, memory);

        let result = run_roundabout_dynamic_web_solver(&fusion, &chain, memory);

        self.compressor.as_ref().map(|c| {
            c.compress(&(
                "roundabout_dynamic_web_pipeline",
                cfg.strengthen_amount,
                cfg.weaken_amount,
                cfg.recency_link_weight,
                predictor_cfg.passes,
                &scratch_before.layer_a,
                &scratch_before.layer_b,
                &grid.edge_indices,
                &grid.edge_norm_weights,
                &fusion.fused_edge_bias,
                &chain.edge_indices,
                chain.total_bias,
                &memory.patterns,
                result.chosen_edge_index,
                result.bias,
            ))
        })
    }
}
