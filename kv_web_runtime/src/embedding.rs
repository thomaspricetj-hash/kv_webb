//! embedding.rs
//!
//! Embedding-based similarity for KV Web nodes + Polygonal-KV geometry.
//!
//! Adds:
//! - centroid-weighted similarity
//! - face-index semantic bias
//! - radius-based gating
//! - dual-layer scratch pads (embedding + semantic geometry)
//! - parallel embedding similarity (edge computation)
//! - indexing + zoning for similarity clusters
//! - GPU-ready compressed packets
//!
//! Max-tier upgrades:
//! - cross-link grids over embedding similarity
//! - revolving-door routing over similarity zones (entry/exit + flow)
//! - fusion field over embedding similarity (scores + geometry + door flow)
//! - embedded Roundabout logic (predictor + smoothing + memory + solver)
//!
//! All upgrades are backwards-compatible.

use kv_web_core::{KvWeb, WebNodeId};
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Trait for an embedding provider.
pub trait EmbeddingProvider {
    fn embed_node(&self, node_id: WebNodeId, web: &KvWeb) -> Vec<f32>;
}

/// Simple cosine similarity.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }

    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }

    dot / (na.sqrt() * nb.sqrt())
}

/// Polygon-aware similarity modifier.
fn polygon_similarity_bias(
    web: &KvWeb,
    a: WebNodeId,
    b: WebNodeId,
    base_sim: f32,
) -> f32 {
    let node_a = match web.nodes.get(&a) {
        Some(n) => n,
        None => return base_sim,
    };
    let node_b = match web.nodes.get(&b) {
        Some(n) => n,
        None => return base_sim,
    };

    let poly_a = match &node_a.polygon {
        Some(p) => p,
        None => return base_sim,
    };
    let poly_b = match &node_b.polygon {
        Some(p) => p,
        None => return base_sim,
    };

    let face_bonus = if poly_a.face_index == poly_b.face_index { 0.15 } else { 0.0 };

    let mut centroid_dist = 0.0;
    for (ca, cb) in poly_a.centroid.iter().zip(poly_b.centroid.iter()) {
        centroid_dist += (ca - cb).abs();
    }
    let centroid_penalty =
        (centroid_dist / (poly_a.radius + poly_b.radius + 1.0)).min(0.25);

    base_sim + face_bonus - centroid_penalty
}

/// Dual-layer scratch pad for embedding similarity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingScratchPad {
    pub layer_a: Vec<f32>, // raw embedding similarity
    pub layer_b: Vec<f32>, // polygon geometry bias
}

/// Zoning + indexing for embedding similarity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingZoning {
    pub root: WebNodeId,
    pub nodes: Vec<WebNodeId>,
    pub index_map: Vec<WebNodeId>,
    pub zones: Vec<EmbeddingZone>,
    pub scratch: EmbeddingScratchPad,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingZone {
    pub zone_id: usize,
    pub start: usize,
    pub end: usize,
    pub centroid_node: Option<WebNodeId>,
    pub size: usize,
}

/// Cross-link grid over embedding similarity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingCrossLinkGrid {
    pub root: WebNodeId,
    pub nodes: Vec<WebNodeId>,
    pub sims: Vec<f32>,
    pub geom_bias: Vec<f32>,
}

/// Revolving door over embedding zones.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRevolvingDoor {
    pub door_id: usize,
    pub entry_nodes: Vec<WebNodeId>,
    pub exit_nodes: Vec<WebNodeId>,
    pub flow_strength: f32,
}

/// Fusion field over embedding similarity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingFusionField {
    pub fused_node_bias: Vec<f32>,
}

/// Roundabout embedding predictor configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundaboutEmbeddingPredictorConfig {
    pub passes: usize,
    pub min_bias: f32,
    pub max_bias: f32,
    pub smoothing_strength: f32,
}

/// Roundabout embedding chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundaboutEmbeddingChain {
    pub nodes: Vec<WebNodeId>,
    pub total_bias: f32,
}

/// Roundabout embedding pattern memory entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundaboutEmbeddingPattern {
    pub chain: RoundaboutEmbeddingChain,
    pub weight: f32,
}

/// Roundabout embedding pattern memory with decay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundaboutEmbeddingPatternMemory {
    pub patterns: Vec<RoundaboutEmbeddingPattern>,
    pub decay: f32,
}

/// Roundabout embedding solver result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundaboutEmbeddingSolverResult {
    pub chosen_node: WebNodeId,
    pub bias: f32,
}

/// Build dual-layer scratch pad for embedding similarity.
fn build_embedding_scratch_pad(
    web: &KvWeb,
    root: WebNodeId,
    nodes: &[WebNodeId],
    sims: &[f32],
) -> EmbeddingScratchPad {
    let layer_a = sims.to_vec();

    let mut layer_b = Vec::with_capacity(nodes.len());
    for node in nodes {
        let bias = polygon_similarity_bias(web, root, *node, 1.0);
        layer_b.push(bias);
    }

    EmbeddingScratchPad { layer_a, layer_b }
}

/// Build similarity edges between nodes based on embeddings + polygon geometry.
/// Parallelizes similarity computation, then applies edges sequentially.
pub fn build_similarity_edges(
    web: &mut KvWeb,
    provider: &dyn EmbeddingProvider,
    threshold: f32,
    weight_scale: f32,
) {
    let mut embeddings: HashMap<WebNodeId, Vec<f32>> = HashMap::new();

    for id in web.nodes.keys() {
        embeddings.insert(*id, provider.embed_node(*id, web));
    }

    let ids: Vec<WebNodeId> = web.nodes.keys().cloned().collect();

    let candidate_edges: Vec<(WebNodeId, WebNodeId, f32)> = ids
        .par_iter()
        .enumerate()
        .flat_map(|(i, &a)| {
            let mut local_edges = Vec::new();
            for &b in ids.iter().skip(i + 1) {
                let ea = &embeddings[&a];
                let eb = &embeddings[&b];

                let base_sim = cosine_similarity(ea, eb);
                let sim = polygon_similarity_bias(web, a, b, base_sim);

                if sim >= threshold {
                    let w = sim * weight_scale;
                    local_edges.push((a, b, w));
                }
            }
            local_edges
        })
        .collect();

    for (a, b, w) in candidate_edges {
        web.add_edge(a, b, w, kv_web_core::EdgeKind::Semantic);
        web.add_edge(b, a, w, kv_web_core::EdgeKind::Semantic);
    }
}

/// Build zoning + indexing + scratch pad for embedding similarity.
pub fn embedding_index_and_zone(
    web: &KvWeb,
    root: WebNodeId,
    provider: &dyn EmbeddingProvider,
    threshold: f32,
    num_zones: usize,
) -> EmbeddingZoning {
    let root_emb = provider.embed_node(root, web);

    let mut nodes = Vec::new();
    let mut sims = Vec::new();

    for id in web.nodes.keys() {
        let emb = provider.embed_node(*id, web);
        let base = cosine_similarity(&root_emb, &emb);
        let sim = polygon_similarity_bias(web, root, *id, base);

        if sim >= threshold {
            nodes.push(*id);
            sims.push(sim);
        }
    }

    let mut index_map = nodes.clone();
    index_map.sort_by(|a, b| {
        let ia = nodes.iter().position(|x| x == a).unwrap();
        let ib = nodes.iter().position(|x| x == b).unwrap();
        let sa = sims[ia];
        let sb = sims[ib];
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    });

    let num_zones = num_zones.max(1);
    let len = index_map.len();
    let zone_size = (len as f32 / num_zones as f32).ceil() as usize;

    let mut zones = Vec::new();
    let mut zone_id = 0;
    let mut start = 0;

    while start < len {
        let end = (start + zone_size).min(len);
        let slice = &index_map[start..end];

        let centroid_node = slice.get(slice.len() / 2).cloned();

        zones.push(EmbeddingZone {
            zone_id,
            start,
            end,
            centroid_node,
            size: slice.len(),
        });

        zone_id += 1;
        start = end;
    }

    let scratch = build_embedding_scratch_pad(web, root, &nodes, &sims);

    EmbeddingZoning {
        root,
        nodes,
        index_map,
        zones,
        scratch,
    }
}

/// Compressed embedding zoning (GPU-ready).
pub fn embedding_index_and_zone_compressed(
    web: &KvWeb,
    root: WebNodeId,
    provider: &dyn EmbeddingProvider,
    threshold: f32,
    num_zones: usize,
) -> Option<Vec<u8>> {
    let ez = embedding_index_and_zone(web, root, provider, threshold, num_zones);

    web.compressor.as_ref().map(|c| {
        c.compress(&(
            "embedding_index_and_zone",
            root,
            threshold,
            num_zones,
            &ez.nodes,
            &ez.index_map,
            &ez.zones,
            &ez.scratch.layer_a,
            &ez.scratch.layer_b,
        ))
    })
}

/// Build cross-link grid from embedding zoning.
fn build_embedding_cross_link_grid(
    web: &KvWeb,
    ez: &EmbeddingZoning,
) -> EmbeddingCrossLinkGrid {
    let sims = ez.scratch.layer_a.clone();
    let mut geom_bias = Vec::with_capacity(ez.nodes.len());
    for node in &ez.nodes {
        let bias = polygon_similarity_bias(web, ez.root, *node, 1.0);
        geom_bias.push(bias);
    }

    EmbeddingCrossLinkGrid {
        root: ez.root,
        nodes: ez.nodes.clone(),
        sims,
        geom_bias,
    }
}

/// Build revolving doors over embedding zones.
fn build_embedding_revolving_doors(
    ez: &EmbeddingZoning,
    grid: &EmbeddingCrossLinkGrid,
) -> Vec<EmbeddingRevolvingDoor> {
    let mut doors = Vec::new();
    if ez.nodes.is_empty() {
        return doors;
    }

    let mut idxs: Vec<usize> = (0..grid.nodes.len()).collect();
    idxs.sort_by(|&a, &b| {
        grid.sims[b]
            .partial_cmp(&grid.sims[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let zone_count = (idxs.len() as f32 * 0.1).ceil() as usize;
    let entry_count = zone_count;
    let exit_count = zone_count;

    let entry_nodes: Vec<WebNodeId> = idxs
        .iter()
        .rev()
        .take(entry_count)
        .map(|i| grid.nodes[*i])
        .collect();

    let exit_nodes: Vec<WebNodeId> = idxs
        .iter()
        .take(exit_count)
        .map(|i| grid.nodes[*i])
        .collect();

    let mut entry_avg = 0.0f32;
    let mut exit_avg = 0.0f32;

    for n in &entry_nodes {
        if let Some(pos) = grid.nodes.iter().position(|id| id == n) {
            entry_avg += grid.sims[pos];
        }
    }
    for n in &exit_nodes {
        if let Some(pos) = grid.nodes.iter().position(|id| id == n) {
            exit_avg += grid.sims[pos];
        }
    }

    if !entry_nodes.is_empty() {
        entry_avg /= entry_nodes.len() as f32;
    }
    if !exit_nodes.is_empty() {
        exit_avg /= exit_nodes.len() as f32;
    }

    let flow_strength = (exit_avg - entry_avg).abs();

    doors.push(EmbeddingRevolvingDoor {
        door_id: 0,
        entry_nodes,
        exit_nodes,
        flow_strength,
    });

    doors
}

/// Build fusion field over embedding similarity.
fn build_embedding_fusion_field(
    _web: &KvWeb,
    grid: &EmbeddingCrossLinkGrid,
    doors: &[EmbeddingRevolvingDoor],
) -> EmbeddingFusionField {
    let mut fused = Vec::with_capacity(grid.nodes.len());

    for (idx, id) in grid.nodes.iter().enumerate() {
        let mut bias = grid.sims[idx];

        let geom = grid.geom_bias[idx];
        bias *= (0.5 + geom * 0.5);

        for door in doors {
            if door.exit_nodes.contains(id) {
                bias *= 1.0 + door.flow_strength * 0.1;
            } else if door.entry_nodes.contains(id) {
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

    EmbeddingFusionField {
        fused_node_bias: fused,
    }
}

/// Run roundabout embedding predictor.
fn run_roundabout_embedding_predictor(
    grid: &EmbeddingCrossLinkGrid,
    fusion: &EmbeddingFusionField,
    cfg: &RoundaboutEmbeddingPredictorConfig,
) -> RoundaboutEmbeddingChain {
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

    RoundaboutEmbeddingChain {
        nodes,
        total_bias: total,
    }
}

/// Smooth roundabout embedding chain.
fn smooth_roundabout_embedding_chain(
    grid: &EmbeddingCrossLinkGrid,
    chain: &mut RoundaboutEmbeddingChain,
    fusion: &EmbeddingFusionField,
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

/// Update embedding pattern memory.
fn update_roundabout_embedding_pattern_memory(
    memory: &mut RoundaboutEmbeddingPatternMemory,
    chain: &RoundaboutEmbeddingChain,
) {
    for pattern in &mut memory.patterns {
        pattern.weight *= memory.decay;
    }

    memory.patterns.push(RoundaboutEmbeddingPattern {
        chain: chain.clone(),
        weight: 1.0,
    });

    memory.patterns.retain(|p| p.weight > 0.01);
}

/// Apply roundabout bias to fused embedding similarity.
fn apply_roundabout_embedding_bias(
    grid: &EmbeddingCrossLinkGrid,
    fusion: &mut EmbeddingFusionField,
    memory: &RoundaboutEmbeddingPatternMemory,
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

/// Run roundabout embedding solver.
fn run_roundabout_embedding_solver(
    grid: &EmbeddingCrossLinkGrid,
    fusion: &EmbeddingFusionField,
    chain: &RoundaboutEmbeddingChain,
    memory: &RoundaboutEmbeddingPatternMemory,
) -> RoundaboutEmbeddingSolverResult {
    if let Some(&last) = chain.nodes.last() {
        let bias = if let Some(pos) = grid.nodes.iter().position(|x| *x == last) {
            fusion.fused_node_bias.get(pos).copied().unwrap_or(0.0)
        } else {
            0.0
        };
        return RoundaboutEmbeddingSolverResult {
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

    RoundaboutEmbeddingSolverResult {
        chosen_node,
        bias: final_bias,
    }
}

/// Optimization config for embedding-based similarity.
#[derive(Debug, Clone)]
pub struct EmbeddingOptimizationConfig {
    pub min_threshold: f32,
    pub max_threshold: f32,
    pub min_weight_scale: f32,
    pub max_weight_scale: f32,
    pub target_edge_density: f32,
    pub max_edge_density: f32,
}

/// Max-tier optimization loop for embedding similarity.
pub fn optimize_embedding_similarity(
    web: &KvWeb,
    current_threshold: &mut f32,
    current_weight_scale: &mut f32,
    opt_cfg: &EmbeddingOptimizationConfig,
) {
    if web.nodes.is_empty() {
        return;
    }

    let node_count = web.nodes.len() as f32;
    let edge_count = web
        .edges
        .iter()
        .filter(|e| matches!(e.kind, kv_web_core::EdgeKind::Semantic))
        .count() as f32;

    let density = edge_count / node_count;

    if density < opt_cfg.target_edge_density {
        *current_threshold =
            (*current_threshold * 0.95).max(opt_cfg.min_threshold);
        *current_weight_scale =
            (*current_weight_scale * 1.05).min(opt_cfg.max_weight_scale);
    }

    if density > opt_cfg.max_edge_density {
        *current_threshold =
            (*current_threshold * 1.05).min(opt_cfg.max_threshold);
        *current_weight_scale =
            (*current_weight_scale * 0.9).max(opt_cfg.min_weight_scale);
    }
}

/// Packet type for compressed roundabout embedding pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRoundaboutPipelinePacket {
    pub tag: &'static str,
    pub root: WebNodeId,
    pub threshold: f32,
    pub num_zones: usize,
    pub passes: usize,
    pub nodes: Vec<WebNodeId>,
    pub index_map: Vec<WebNodeId>,
    pub zones: Vec<EmbeddingZone>,
    pub scratch_layer_a: Vec<f32>,
    pub scratch_layer_b: Vec<f32>,
    pub grid_nodes: Vec<WebNodeId>,
    pub grid_sims: Vec<f32>,
    pub grid_geom_bias: Vec<f32>,
    pub fused_node_bias: Vec<f32>,
    pub chain_nodes: Vec<WebNodeId>,
    pub chain_total_bias: f32,
    pub patterns: Vec<RoundaboutEmbeddingPattern>,
    pub chosen_node: WebNodeId,
    pub chosen_bias: f32,
}

/// Compressed roundabout embedding pipeline (GPU-ready).
pub fn embedding_roundabout_pipeline_compressed(
    web: &KvWeb,
    root: WebNodeId,
    provider: &dyn EmbeddingProvider,
    threshold: f32,
    num_zones: usize,
    predictor_cfg: &RoundaboutEmbeddingPredictorConfig,
    memory: &mut RoundaboutEmbeddingPatternMemory,
) -> Option<Vec<u8>> {
    let ez = embedding_index_and_zone(web, root, provider, threshold, num_zones);
    let grid = build_embedding_cross_link_grid(web, &ez);
    let doors = build_embedding_revolving_doors(&ez, &grid);
    let mut fusion = build_embedding_fusion_field(web, &grid, &doors);

    let mut chain = run_roundabout_embedding_predictor(&grid, &fusion, predictor_cfg);
    smooth_roundabout_embedding_chain(&grid, &mut chain, &fusion, predictor_cfg.smoothing_strength);

    update_roundabout_embedding_pattern_memory(memory, &chain);
    apply_roundabout_embedding_bias(&grid, &mut fusion, memory);

    let result = run_roundabout_embedding_solver(&grid, &fusion, &chain, memory);

    let packet = EmbeddingRoundaboutPipelinePacket {
        tag: "embedding_roundabout_pipeline",
        root,
        threshold,
        num_zones,
        passes: predictor_cfg.passes,
        nodes: ez.nodes.clone(),
        index_map: ez.index_map.clone(),
        zones: ez.zones.clone(),
        scratch_layer_a: ez.scratch.layer_a.clone(),
        scratch_layer_b: ez.scratch.layer_b.clone(),
        grid_nodes: grid.nodes.clone(),
        grid_sims: grid.sims.clone(),
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


