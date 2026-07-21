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

    // Compute candidate edges in parallel, but don't mutate `web` inside the closure.
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

    // Apply edges sequentially to avoid mutable borrow inside parallel closure.
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


