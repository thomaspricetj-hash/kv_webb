//! embedding.rs
//!
//! Embedding-based similarity for KV Web nodes.
//! This module assumes you have some external embedding provider.

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

/// Build similarity edges between nodes based on embeddings.
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

            let sim = cosine_similarity(ea, eb);
            if sim >= threshold {
                let w = sim * weight_scale;
                web.add_edge(a, b, w, kv_web_core::EdgeKind::Semantic);
                web.add_edge(b, a, w, kv_web_core::EdgeKind::Semantic);
            }
        }
    }
}
