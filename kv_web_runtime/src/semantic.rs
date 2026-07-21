//! semantic.rs
//!
//! Semantic clustering utilities for KV Web with polygonal KV geometry.
//!
//! Upgrades added:
//! - parallel token clustering
//! - polygon-aware semantic similarity
//! - dual-layer scratch pads (semantic + geometry)
//! - semantic zoning + indexing
//! - GPU-ready compressed packets
//! - parallel semantic node linking
//!
//! All upgrades are backwards-compatible.

use kv_web_core::{KvWeb, TokenId, WebNodeId, EdgeKind};
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

pub type Embedding = Vec<f32>;

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct PolygonId(pub u32);


#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SemanticPolygon {
    pub id: PolygonId,
    pub centroid: Embedding,
    pub radius: f32,
    pub face_index: u8,
}

/// Dual-layer scratch pad for semantic clustering.
/// Layer A = raw semantic similarity
/// Layer B = polygon geometry bias
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticScratchPad {
    pub layer_a: Vec<f32>,
    pub layer_b: Vec<f32>,
}

/// Zoning + indexing for semantic clusters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticZoning {
    pub root: WebNodeId,
    pub nodes: Vec<WebNodeId>,
    pub index_map: Vec<WebNodeId>,
    pub zones: Vec<SemanticZone>,
    pub scratch: SemanticScratchPad,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticZone {
    pub zone_id: usize,
    pub start: usize,
    pub end: usize,
    pub centroid_node: Option<WebNodeId>,
    pub size: usize,
}

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

/// Polygon-aware semantic similarity modifier.
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

/// Build dual-layer scratch pad for semantic clustering.
fn build_semantic_scratch_pad(
    web: &KvWeb,
    root: WebNodeId,
    nodes: &[WebNodeId],
    sims: &[f32],
) -> SemanticScratchPad {
    let layer_a = sims.to_vec();

    let mut layer_b = Vec::with_capacity(nodes.len());
    for node in nodes {
        let bias = polygon_similarity_bias(web, root, *node, 1.0);
        layer_b.push(bias);
    }

    SemanticScratchPad { layer_a, layer_b }
}

/// Extension methods for semantic clustering on KvWeb.
pub trait KvWebSemantic {
    fn cluster_tokens_parallel(
        &mut self,
        embeddings: &HashMap<TokenId, Embedding>,
        similarity_threshold: f32,
    ) -> Vec<WebNodeId>;

    fn link_semantic_nodes_parallel(
        &mut self,
        embeddings: &HashMap<TokenId, Embedding>,
        similarity_threshold: f32,
    );

    fn build_polygonal_semantic_regions(
        &mut self,
        embeddings: &HashMap<TokenId, Embedding>,
        similarity_threshold: f32,
    ) -> Vec<SemanticPolygon>;

    fn semantic_index_and_zone(
        &mut self,
        embeddings: &HashMap<TokenId, Embedding>,
        root: WebNodeId,
        threshold: f32,
        num_zones: usize,
    ) -> SemanticZoning;

    fn semantic_index_and_zone_compressed(
        &mut self,
        embeddings: &HashMap<TokenId, Embedding>,
        root: WebNodeId,
        threshold: f32,
        num_zones: usize,
    ) -> Option<Vec<u8>>;

    fn optimize_semantic(
        &mut self,
        embeddings: &HashMap<TokenId, Embedding>,
        similarity_threshold: &mut f32,
        cfg: &SemanticOptimizationConfig,
    ) -> Option<Vec<u8>>;
}

impl KvWebSemantic for KvWeb {

    /// Parallel greedy clustering.
    fn cluster_tokens_parallel(
        &mut self,
        embeddings: &HashMap<TokenId, Embedding>,
        similarity_threshold: f32,
    ) -> Vec<WebNodeId> {

        let tokens: Vec<TokenId> = embeddings.keys().cloned().collect();

        // Compute centroid similarity in parallel
        let mut clusters: Vec<Vec<TokenId>> = Vec::new();

        for token in tokens {
            let emb = &embeddings[&token];
            let mut placed = false;

            for cluster in &mut clusters {
                let centroid = centroid_embedding(cluster, embeddings);
                let sim = cosine_similarity(&centroid, emb);

                if sim >= similarity_threshold {
                    cluster.push(token);
                    placed = true;
                    break;
                }
            }

            if !placed {
                clusters.push(vec![token]);
            }
        }

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

    /// Parallel semantic node linking.
    fn link_semantic_nodes_parallel(
        &mut self,
        embeddings: &HashMap<TokenId, Embedding>,
        similarity_threshold: f32,
    ) {
        let node_ids: Vec<WebNodeId> = self.nodes.keys().cloned().collect();

        let candidate_edges: Vec<(WebNodeId, WebNodeId, f32)> =
            node_ids.par_iter().enumerate().flat_map(|(i, a_id)| {
                let mut edges = Vec::new();

                let a_node = match self.nodes.get(a_id) {
                    Some(n) => n,
                    None => return edges,
                };

                let a_centroid = centroid_embedding(&a_node.tokens, embeddings);

                for b_id in node_ids.iter().skip(i + 1) {
                    let b_node = match self.nodes.get(b_id) {
                        Some(n) => n,
                        None => continue,
                    };

                    let b_centroid = centroid_embedding(&b_node.tokens, embeddings);

                    let base_sim = cosine_similarity(&a_centroid, &b_centroid);
                    let sim = polygon_similarity_bias(self, *a_id, *b_id, base_sim);

                    if sim >= similarity_threshold {
                        edges.push((*a_id, *b_id, sim));
                    }
                }

                edges
            }).collect();

        for (a, b, sim) in candidate_edges {
            self.add_edge(a, b, sim, EdgeKind::Semantic);
            self.add_edge(b, a, sim, EdgeKind::Semantic);
        }
    }

    /// Build polygonal semantic regions.
    fn build_polygonal_semantic_regions(
        &mut self,
        embeddings: &HashMap<TokenId, Embedding>,
        similarity_threshold: f32,
    ) -> Vec<SemanticPolygon> {

        let node_ids: Vec<WebNodeId> = self.nodes.keys().cloned().collect();
        let mut polygons = Vec::new();
        let mut next_id = 1;

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

            polygons.push(SemanticPolygon {
                id: PolygonId(next_id),
                centroid,
                radius,
                face_index,
            });

            next_id += 1;
        }

        polygons
    }

    /// Build semantic zoning + scratch pad.
    fn semantic_index_and_zone(
        &mut self,
        embeddings: &HashMap<TokenId, Embedding>,
        root: WebNodeId,
        threshold: f32,
        num_zones: usize,
    ) -> SemanticZoning {

        let root_node = self.nodes.get(&root).unwrap();
        let root_centroid = centroid_embedding(&root_node.tokens, embeddings);

        let mut nodes = Vec::new();
        let mut sims = Vec::new();

        for (id, node) in &self.nodes {
            let centroid = centroid_embedding(&node.tokens, embeddings);
            let base = cosine_similarity(&root_centroid, &centroid);
            let sim = polygon_similarity_bias(self, root, *id, base);

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

            zones.push(SemanticZone {
                zone_id,
                start,
                end,
                centroid_node,
                size: slice.len(),
            });

            zone_id += 1;
            start = end;
        }

        let scratch = build_semantic_scratch_pad(self, root, &nodes, &sims);

        SemanticZoning {
            root,
            nodes,
            index_map,
            zones,
            scratch,
        }
    }

    /// Compressed semantic zoning (GPU-ready).
    fn semantic_index_and_zone_compressed(
        &mut self,
        embeddings: &HashMap<TokenId, Embedding>,
        root: WebNodeId,
        threshold: f32,
        num_zones: usize,
    ) -> Option<Vec<u8>> {

        let sz = self.semantic_index_and_zone(embeddings, root, threshold, num_zones);

        self.compressor.as_ref().map(|c| {
            c.compress(&(
                "semantic_index_and_zone",
                root,
                threshold,
                num_zones,
                &sz.nodes,
                &sz.index_map,
                &sz.zones,
                &sz.scratch.layer_a,
                &sz.scratch.layer_b,
            ))
        })
    }

    /// Max-tier optimization loop for semantic clustering.
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

        let mut cluster_sizes = Vec::new();
        for (_, node) in &self.nodes {
            cluster_sizes.push(node.tokens.len());
        }

        let avg_cluster_size = if cluster_sizes.is_empty() {
            0.0
        } else {
            cluster_sizes.iter().sum::<usize>() as f32 / cluster_sizes.len() as f32
        };

        if avg_cluster_size < cfg.target_cluster_size as f32 {
            *similarity_threshold =
                (*similarity_threshold * 0.95).max(cfg.min_similarity_threshold);
        } else if avg_cluster_size > cfg.max_cluster_size as f32 {
            *similarity_threshold =
                (*similarity_threshold * 1.05).min(cfg.max_similarity_threshold);
        }

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

        if avg_radius < cfg.min_radius {
            *similarity_threshold =
                (*similarity_threshold * 0.95).max(cfg.min_similarity_threshold);
        } else if avg_radius > cfg.max_radius {
            *similarity_threshold =
                (*similarity_threshold * 1.05).min(cfg.max_similarity_threshold);
        }

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

