//! semantic.rs
//! Semantic clustering utilities for KV Web.
//!
//! This module assumes you already have token embeddings from
//! some external model. It does NOT compute embeddings itself;
//! it just groups tokens into nodes based on similarity.

use kv_web_core::{KvWeb, TokenId, WebNodeId, EdgeKind};
use std::collections::HashMap;

/// Simple embedding type: a slice of f32.
/// Caller provides these; we don't depend on any ML crate here.
pub type Embedding = Vec<f32>;

/// Compute cosine similarity between two embeddings.
fn cosine_similarity(a: &Embedding, b: &Embedding) -> f32 {
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

/// Extension methods for semantic clustering on KvWeb.
pub trait KvWebSemantic {
    /// Cluster tokens into nodes based on similarity threshold.
    ///
    /// `embeddings` maps TokenId -> embedding vector.
    /// `similarity_threshold` controls how tight clusters are (0.0–1.0).
    fn cluster_tokens(
        &mut self,
        embeddings: &HashMap<TokenId, Embedding>,
        similarity_threshold: f32,
    ) -> Vec<WebNodeId>;

    /// Create semantic edges between nodes whose centroids are similar.
    fn link_semantic_nodes(
        &mut self,
        embeddings: &HashMap<TokenId, Embedding>,
        similarity_threshold: f32,
    );
}

impl KvWebSemantic for KvWeb {
    fn cluster_tokens(
        &mut self,
        embeddings: &HashMap<TokenId, Embedding>,
        similarity_threshold: f32,
    ) -> Vec<WebNodeId> {
        let mut clusters: Vec<Vec<TokenId>> = Vec::new();

        // Greedy clustering: assign each token to the first centroid it matches.
        for (token, emb) in embeddings {
            let mut placed = false;

            for cluster in &mut clusters {
                let centroid = centroid_embedding(cluster, embeddings);
                let sim = cosine_similarity(&centroid, emb);

                if sim >= similarity_threshold {
                    cluster.push(*token);
                    placed = true;
                    break;
                }
            }

            if !placed {
                clusters.push(vec![*token]);
            }
        }

        // Convert clusters into nodes
        let mut node_ids = Vec::new();
        for cluster in clusters {
            if cluster.is_empty() {
                continue;
            }

            let label = format!("cluster_{}", cluster.len());
            let score = cluster.len() as f32;

            let id = self.create_node(cluster, Some(label), score);
            node_ids.push(id);
        }

        node_ids
    }

    fn link_semantic_nodes(
        &mut self,
        embeddings: &HashMap<TokenId, Embedding>,
        similarity_threshold: f32,
    ) {
        let node_ids: Vec<WebNodeId> = self.nodes.keys().cloned().collect();

        for (i, a_id) in node_ids.iter().enumerate() {
            for b_id in node_ids.iter().skip(i + 1) {
                let a_node = match self.nodes.get(a_id) {
                    Some(n) => n,
                    None => continue,
                };
                let b_node = match self.nodes.get(b_id) {
                    Some(n) => n,
                    None => continue,
                };

                let a_centroid = centroid_embedding(&a_node.tokens, embeddings);
                let b_centroid = centroid_embedding(&b_node.tokens, embeddings);

                let sim = cosine_similarity(&a_centroid, &b_centroid);
                if sim >= similarity_threshold {
                    // bidirectional semantic edges
                    self.add_edge(*a_id, *b_id, sim, EdgeKind::Semantic);
                    self.add_edge(*b_id, *a_id, sim, EdgeKind::Semantic);
                }
            }
        }
    }
}

/// Compute centroid embedding for a set of tokens.
fn centroid_embedding(
    tokens: &Vec<TokenId>,
    embeddings: &HashMap<TokenId, Embedding>,
) -> Embedding {
    if tokens.is_empty() {
        return Vec::new();
    }

    // Determine embedding dimension
    let mut dim = 0;
    for t in tokens {
        if let Some(e) = embeddings.get(t) {
            dim = e.len();
            break;
        }
    }

    if dim == 0 {
        return Vec::new();
    }

    let mut sum = vec![0.0; dim];
    let mut count = 0.0;

    for t in tokens {
        if let Some(e) = embeddings.get(t) {
            for (i, v) in e.iter().enumerate() {
                sum[i] += v;
            }
            count += 1.0;
        }
    }

    if count == 0.0 {
        return sum;
    }

    for v in &mut sum {
        *v /= count;
    }

    sum
}
