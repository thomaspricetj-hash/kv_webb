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
