//! embedding.rs
//!
//! Embedding-based similarity for KV Web nodes + Polygonal-KV geometry.
//!
//! This module assumes you have some external embedding provider.
//! Polygonal-KV adds:
//! - centroid-weighted similarity
//! - face-index semantic bias
//! - radius-based gating
//!
//! All upgrades are backwards-compatible.

use kv_web_core::{KvWeb, WebNodeId};
use std::collections::HashMap;

/// Trait for an embedding provider.
pub trait EmbeddingProvider {
    /// Get an embedding for a node label or content.
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
/// Uses centroid, radius, and face index to bias similarity.
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

    // Face match bonus
    let face_bonus = if poly_a.face_index == poly_b.face_index {
        0.15
    } else {
        0.0
    };

    // Centroid distance penalty
    let mut centroid_dist = 0.0;
    for (ca, cb) in poly_a.centroid.iter().zip(poly_b.centroid.iter()) {
        centroid_dist += (ca - cb).abs();
    }
    let centroid_penalty = (centroid_dist / (poly_a.radius + poly_b.radius + 1.0)).min(0.25);

    // Final polygon-aware similarity
    base_sim + face_bonus - centroid_penalty
}

/// Build similarity edges between nodes based on embeddings + polygon geometry.
pub fn build_similarity_edges(
    web: &mut KvWeb,
    provider: &dyn EmbeddingProvider,
    threshold: f32,
    weight_scale: f32,
) {
    let mut embeddings: HashMap<WebNodeId, Vec<f32>> = HashMap::new();

    for id in web.nodes.keys() {
        let emb = provider.embed_node(*id, web);
        embeddings.insert(*id, emb);
    }

    let ids: Vec<WebNodeId> = web.nodes.keys().cloned().collect();

    for (i, &a) in ids.iter().enumerate() {
        for &b in ids.iter().skip(i + 1) {
            let ea = &embeddings[&a];
            let eb = &embeddings[&b];

            let base_sim = cosine_similarity(ea, eb);

            // Polygon-aware similarity upgrade
            let sim = polygon_similarity_bias(web, a, b, base_sim);

            if sim >= threshold {
                let w = sim * weight_scale;
                web.add_edge(a, b, w, kv_web_core::EdgeKind::Semantic);
                web.add_edge(b, a, w, kv_web_core::EdgeKind::Semantic);
            }
        }
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
/// Adjusts threshold and weight_scale based on resulting semantic edge density.
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

    let density = if node_count > 0.0 {
        edge_count / node_count
    } else {
        0.0
    };

    // If density is too low, lower threshold and increase weight_scale slightly.
    if density < opt_cfg.target_edge_density {
        *current_threshold =
            (*current_threshold * 0.95).max(opt_cfg.min_threshold);
        *current_weight_scale =
            (*current_weight_scale * 1.05).min(opt_cfg.max_weight_scale);
    }

    // If density is too high, raise threshold and decrease weight_scale slightly.
    if density > opt_cfg.max_edge_density {
        *current_threshold =
            (*current_threshold * 1.05).min(opt_cfg.max_threshold);
        *current_weight_scale =
            (*current_weight_scale * 0.9).max(opt_cfg.min_weight_scale);
    }
}

