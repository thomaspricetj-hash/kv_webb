//! semantic.rs
//! Semantic clustering utilities for KV Web with polygonal KV geometry.
//!
//! This module assumes you already have token embeddings from
//! some external model. It does NOT compute embeddings itself;
//! it just groups tokens into nodes based on similarity and
//! assigns them to polygonal semantic regions.

use kv_web_core::{KvWeb, TokenId, WebNodeId, EdgeKind};
use std::collections::HashMap;

/// Simple embedding type: a slice of f32.
/// Caller provides these; we don't depend on any ML crate here.
pub type Embedding = Vec<f32>;

/// Polygonal semantic region identifier.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub struct PolygonId(pub u32);

/// A simple polygon in embedding space: centroid + radius + face index.
/// This is intentionally lightweight so it compresses well under BitDrop.
#[derive(Clone, Debug)]
pub struct SemanticPolygon {
    pub id: PolygonId,
    pub centroid: Embedding,
    pub radius: f32,
    pub face_index: u8,
}

/// Optimization config for semantic clustering.
#[derive(Debug, Clone)]
pub struct SemanticOptimizationConfig {
    pub min_similarity_threshold: f32,
    pub max_similarity_threshold: f32,
    pub target_cluster_size: usize,
    pub max_cluster_size: usize,
    pub min_radius: f32,
    pub max_radius: f32,
}

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
    fn cluster_tokens(
        &mut self,
        embeddings: &HashMap<TokenId, Embedding>,
        similarity_threshold: f32,
    ) -> Vec<WebNodeId>;

    fn link_semantic_nodes(
        &mut self,
        embeddings: &HashMap<TokenId, Embedding>,
        similarity_threshold: f32,
    );

    fn build_polygonal_semantic_regions(
        &mut self,
        embeddings: &HashMap<TokenId, Embedding>,
        similarity_threshold: f32,
    ) -> Vec<SemanticPolygon>;

    /// Max-tier optimization loop for semantic clustering.
    fn optimize_semantic(
        &mut self,
        embeddings: &HashMap<TokenId, Embedding>,
        similarity_threshold: &mut f32,
        cfg: &SemanticOptimizationConfig,
    ) -> Option<Vec<u8>>;
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
                    self.add_edge(*a_id, *b_id, sim, EdgeKind::Semantic);
                    self.add_edge(*b_id, *a_id, sim, EdgeKind::Semantic);
                }
            }
        }
    }

    fn build_polygonal_semantic_regions(
        &mut self,
        embeddings: &HashMap<TokenId, Embedding>,
        similarity_threshold: f32,
    ) -> Vec<SemanticPolygon> {
        let node_ids: Vec<WebNodeId> = self.nodes.keys().cloned().collect();
        let mut polygons: Vec<SemanticPolygon> = Vec::new();
        let mut next_id: u32 = 1;

        // Each seed node becomes a polygon centroid; nearby nodes define radius.
        for node_id in &node_ids {
            let node = match self.nodes.get(node_id) {
                Some(n) => n,
                None => continue,
            };

            let centroid = centroid_embedding(&node.tokens, embeddings);
            if centroid.is_empty() {
                continue;
            }

            let mut radius_acc = 0.0;
            let mut radius_count = 0.0;

            for other_id in &node_ids {
                if other_id == node_id {
                    continue;
                }

                let other = match self.nodes.get(other_id) {
                    Some(n) => n,
                    None => continue,
                };

                let other_centroid = centroid_embedding(&other.tokens, embeddings);
                if other_centroid.is_empty() {
                    continue;
                }

                let sim = cosine_similarity(&centroid, &other_centroid);
                if sim >= similarity_threshold {
                    radius_acc += sim;
                    radius_count += 1.0;
                }
            }

            let radius = if radius_count > 0.0 {
                radius_acc / radius_count
            } else {
                similarity_threshold
            };

            let face_index = if radius >= 0.9 {
                3
            } else if radius >= 0.7 {
                2
            } else if radius >= 0.5 {
                1
            } else {
                0
            };

            let polygon = SemanticPolygon {
                id: PolygonId(next_id),
                centroid,
                radius,
                face_index,
            };

            next_id += 1;
            polygons.push(polygon);
        }

        polygons
    }

    fn optimize_semantic(
        &mut self,
        embeddings: &HashMap<TokenId, Embedding>,
        similarity_threshold: &mut f32,
        cfg: &SemanticOptimizationConfig,
    ) -> Option<Vec<u8>> {
        let node_ids: Vec<WebNodeId> = self.nodes.keys().cloned().collect();
        if node_ids.is_empty() {
            return None;
        }

        // Measure cluster sizes
        let mut cluster_sizes = Vec::new();
        for (_, node) in &self.nodes {
            cluster_sizes.push(node.tokens.len());
        }

        let avg_cluster_size = if cluster_sizes.is_empty() {
            0.0
        } else {
            cluster_sizes.iter().sum::<usize>() as f32 / cluster_sizes.len() as f32
        };

        // Adjust similarity threshold based on cluster size
        if avg_cluster_size < cfg.target_cluster_size as f32 {
            *similarity_threshold =
                (*similarity_threshold * 0.95).max(cfg.min_similarity_threshold);
        } else if avg_cluster_size > cfg.max_cluster_size as f32 {
            *similarity_threshold =
                (*similarity_threshold * 1.05).min(cfg.max_similarity_threshold);
        }

        // Measure polygon radius spread
        let mut radii = Vec::new();
        for (_, node) in &self.nodes {
            if let Some(poly) = &node.polygon {
                radii.push(poly.radius);
            }
        }

        let avg_radius = if radii.is_empty() {
            0.0
        } else {
            radii.iter().sum::<f32>() / radii.len() as f32
        };

        // Adjust radius indirectly by nudging similarity threshold
        if avg_radius < cfg.min_radius {
            *similarity_threshold =
                (*similarity_threshold * 0.95).max(cfg.min_similarity_threshold);
        } else if avg_radius > cfg.max_radius {
            *similarity_threshold =
                (*similarity_threshold * 1.05).min(cfg.max_similarity_threshold);
        }

        // Compressed optimization packet
        self.compressor.as_ref().map(|c| {
            c.compress(&(
                "optimize_semantic",
                avg_cluster_size,
                avg_radius,
                *similarity_threshold,
            ))
        })
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

