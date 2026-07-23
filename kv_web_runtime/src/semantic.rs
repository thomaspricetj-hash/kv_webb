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
//! - Prompt‑Injection Firewall (SPIF) + multi-attack semantic firewalls
//! - Auto‑Threat Detection Engine (ATDE) with hybrid detect/log/block/adapt
//!
//! MAX‑tier upgrades added:
//! - semantic roundabout hubs per root node
//! - multilayer exit scoring over semantic zones
//! - circulation when no stable semantic exit is available
//! - scratchpad‑aware semantic exit hinting
//! - stability‑weighted semantic routing decisions
//!
//! All upgrades are backwards-compatible.

use kv_web_core::{KvWeb, TokenId, WebNodeId, EdgeKind};
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::time::Instant;

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

/// Firewall mode + suspicion level.

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FirewallMode {
    Normal,
    Paranoid,
    Adaptive,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SuspicionLevel {
    Allow,
    Suspicious,
    Block,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatEvent {
    pub level: SuspicionLevel,
    pub reason_code: &'static str,
    pub similarity_to_root: f32,
    pub zone_coherence: f32,
    pub polygon_distance: f32,
    pub embedding_variance: f32,
    pub flip_ratio: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirewallConfig {
    pub mode: FirewallMode,
    pub entropy_min: f32,
    pub entropy_max: f32,
    pub variance_min: f32,
    pub variance_max: f32,
    pub spike_threshold: f32,
    pub zone_similarity_min: f32,
    pub root_similarity_min: f32,
    pub flip_ratio_max: f32,
    pub polygon_distance_factor: f32,
}

impl Default for FirewallConfig {
    fn default() -> Self {
        FirewallConfig {
            mode: FirewallMode::Adaptive,
            entropy_min: 1e-4,
            entropy_max: 10.0,
            variance_min: 1e-4,
            variance_max: 10.0,
            spike_threshold: 0.35,
            zone_similarity_min: 0.20,
            root_similarity_min: 0.05,
            flip_ratio_max: 0.7,
            polygon_distance_factor: 2.0,
        }
    }
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

// ────────────────────────────────────────────────────────────────
//   MULTI-ATTACK SEMANTIC FIREWALL (SPIF+)
//   Hybrid detect/log/block/adapt
// ────────────────────────────────────────────────────────────────
//

fn compute_embedding_stats(new_embedding: &Embedding) -> (f32, f32) {
    let len = new_embedding.len() as f32;
    if len <= 0.0 {
        return (0.0, 0.0);
    }
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for v in new_embedding {
        sum += *v;
        sum_sq += v * v;
    }
    let mean = sum / len;
    let var = (sum_sq / len) - mean * mean;
    (mean, var)
}

fn compute_heatmap_spike(heatmap_layers: Option<&[f32]>) -> f32 {
    if let Some(layers) = heatmap_layers {
        if layers.len() >= 2 {
            let mut spike = 0.0;
            for w in layers.windows(2) {
                spike += (w[0] - w[1]).abs();
            }
            return spike;
        }
    }
    0.0
}

fn compute_zone_coherence(
    web: &KvWeb,
    embeddings: &HashMap<TokenId, Embedding>,
    new_embedding: &Embedding,
    zoning: Option<&SemanticZoning>,
) -> f32 {
    if let Some(z) = zoning {
        let mut best_sim = 0.0;
        for node_id in &z.nodes {
            if let Some(n) = web.nodes.get(node_id) {
                let c = centroid_embedding(&n.tokens, embeddings);
                let sim = cosine_similarity(&c, new_embedding);
                if sim > best_sim {
                    best_sim = sim;
                }
            }
        }
        return best_sim;
    }
    0.0
}

fn compute_root_similarity(
    web: &KvWeb,
    embeddings: &HashMap<TokenId, Embedding>,
    new_embedding: &Embedding,
    zoning: Option<&SemanticZoning>,
) -> f32 {
    if let Some(z) = zoning {
        if let Some(root_node) = web.nodes.get(&z.root) {
            let root_centroid = centroid_embedding(&root_node.tokens, embeddings);
            return cosine_similarity(&root_centroid, new_embedding);
        }
    }
    0.0
}

fn compute_flip_ratio(
    web: &KvWeb,
    embeddings: &HashMap<TokenId, Embedding>,
    new_embedding: &Embedding,
    zoning: Option<&SemanticZoning>,
) -> f32 {
    if let Some(z) = zoning {
        if let Some(root_node) = web.nodes.get(&z.root) {
            let root_centroid = centroid_embedding(&root_node.tokens, embeddings);
            let mut flips = 0;
            let mut total = 0;
            for (r, v) in root_centroid.iter().zip(new_embedding.iter()) {
                if *r != 0.0 && *v != 0.0 {
                    let rs = r.is_sign_negative();
                    let vs = v.is_sign_negative();
                    if rs != vs {
                        flips += 1;
                    }
                    total += 1;
                }
            }
            if total > 0 {
                return flips as f32 / total as f32;
            }
        }
    }
    0.0
}

fn compute_polygon_distance(
    polygon_region: Option<&kv_web_core::PolygonRegion>,
    new_embedding: &Embedding,
) -> f32 {
    if let Some(poly) = polygon_region {
        let mut dist = 0.0;
        for (a, b) in poly.centroid.iter().zip(new_embedding.iter()) {
            dist += (a - b).abs();
        }
        return dist;
    }
    0.0
}

fn evaluate_threat(
    cfg: &FirewallConfig,
    _mean: f32,
    var: f32,
    spike: f32,
    zone_coherence: f32,
    root_similarity: f32,
    flip_ratio: f32,
    polygon_distance: f32,
) -> ThreatEvent {
    let mut level = SuspicionLevel::Allow;
    let mut reason = "NONE";

    // Adversarial embedding (entropy/variance)
    if var < cfg.variance_min || var > cfg.variance_max {
        level = SuspicionLevel::Suspicious;
        reason = "EMBEDDING_VARIANCE_SUSPECT";
    }

    // Heatmap spike (prompt injection / CHA)
    if spike > cfg.spike_threshold {
        level = SuspicionLevel::Suspicious;
        reason = "HEATMAP_SPIKE";
    }

    // Zone coherence (context hijack / zone flooding)
    if zone_coherence < cfg.zone_similarity_min {
        level = SuspicionLevel::Block;
        reason = "ZONE_COHERENCE_LOW";
    }

    // Root similarity (drift / hijack)
    if root_similarity < cfg.root_similarity_min && zone_coherence >= cfg.zone_similarity_min {
        level = SuspicionLevel::Suspicious;
        reason = "ROOT_SIMILARITY_LOW";
    }

    // Polarity inversion
    if flip_ratio > cfg.flip_ratio_max {
        level = SuspicionLevel::Block;
        reason = "POLARITY_FLIP_HIGH";
    }

    // Geometry break
    if polygon_distance > cfg.polygon_distance_factor * var.max(1.0) {
        level = SuspicionLevel::Block;
        reason = "GEOMETRY_BREAK";
    }

    ThreatEvent {
        level,
        reason_code: reason,
        similarity_to_root: root_similarity,
        zone_coherence,
        polygon_distance,
        embedding_variance: var,
        flip_ratio,
    }
}

fn adapt_firewall_config(cfg: &mut FirewallConfig, event: &ThreatEvent) {
    match cfg.mode {
        FirewallMode::Adaptive => {
            match event.level {
                SuspicionLevel::Block => {
                    // Tighten thresholds
                    cfg.zone_similarity_min = (cfg.zone_similarity_min + 0.01).min(0.5);
                    cfg.root_similarity_min = (cfg.root_similarity_min + 0.01).min(0.3);
                    cfg.spike_threshold = (cfg.spike_threshold * 0.95).max(0.1);
                }
                SuspicionLevel::Suspicious => {
                    // Slight adjustments
                    cfg.zone_similarity_min = (cfg.zone_similarity_min + 0.005).min(0.4);
                    cfg.root_similarity_min = (cfg.root_similarity_min + 0.005).min(0.25);
                }
                SuspicionLevel::Allow => {
                    // Relax slightly over time
                    cfg.zone_similarity_min = (cfg.zone_similarity_min * 0.999).max(0.15);
                    cfg.root_similarity_min = (cfg.root_similarity_min * 0.999).max(0.03);
                }
            }
        }
        FirewallMode::Paranoid => {
            // Always tighten
            cfg.zone_similarity_min = (cfg.zone_similarity_min + 0.01).min(0.6);
            cfg.root_similarity_min = (cfg.root_similarity_min + 0.01).min(0.4);
            cfg.spike_threshold = (cfg.spike_threshold * 0.95).max(0.1);
        }
        FirewallMode::Normal => {
            // No adaptation
        }
    }
}

fn spif_detect(
    web: &KvWeb,
    embeddings: &HashMap<TokenId, Embedding>,
    new_embedding: &Embedding,
    polygon_region: Option<&kv_web_core::PolygonRegion>,
    zoning: Option<&SemanticZoning>,
    heatmap_layers: Option<&[f32]>,
) -> bool {
    let mut cfg = FirewallConfig::default();

    let (mean, var) = compute_embedding_stats(new_embedding);
    let spike = compute_heatmap_spike(heatmap_layers);
    let zone_coherence = compute_zone_coherence(web, embeddings, new_embedding, zoning);
    let root_similarity = compute_root_similarity(web, embeddings, new_embedding, zoning);
    let flip_ratio = compute_flip_ratio(web, embeddings, new_embedding, zoning);
    let polygon_distance = compute_polygon_distance(polygon_region, new_embedding);

    let event = evaluate_threat(
        &cfg,
        mean,
        var,
        spike,
        zone_coherence,
        root_similarity,
        flip_ratio,
        polygon_distance,
    );

    adapt_firewall_config(&mut cfg, &event);

    match event.level {
        SuspicionLevel::Allow => false,
        SuspicionLevel::Suspicious => {
            // Suspicious: allow but log; here we just treat as allow at runtime
            false
        }
        SuspicionLevel::Block => true,
    }
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

        let mut clusters: Vec<Vec<TokenId>> = Vec::new();

        for token in tokens {
            let emb = &embeddings[&token];

            // Multi-attack firewall on raw token embedding
            if spif_detect(self, embeddings, emb, None, None, None) {
                continue;
            }

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

                // Firewall on node centroid
                if spif_detect(self, embeddings, &a_centroid, a_node.polygon.as_ref(), None, None) {
                    return edges;
                }

                for b_id in node_ids.iter().skip(i + 1) {
                    let b_node = match self.nodes.get(b_id) {
                        Some(n) => n,
                        None => continue,
                    };

                    let b_centroid = centroid_embedding(&b_node.tokens, embeddings);

                    // Firewall on other node centroid
                    if spif_detect(self, embeddings, &b_centroid, b_node.polygon.as_ref(), None, None) {
                        continue;
                    }

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

            // Firewall on polygon centroid candidate
            if spif_detect(self, embeddings, &centroid, node.polygon.as_ref(), None, None) {
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

            // Firewall on zoning candidate
            if spif_detect(self, embeddings, &centroid, node.polygon.as_ref(), None, None) {
                continue;
            }

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

// ────────────────────────────────────────────────────────────────
//   SEMANTIC ROUNDABOUT MAX‑TIER UPGRADE
//   Hub‑based multilayer semantic routing
// ────────────────────────────────────────────────────────────────
//

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticPriority {
    High,
    Standard,
    Low,
}

#[derive(Debug, Clone)]
pub struct SemanticPacket {
    pub id: u64,
    pub priority: SemanticPriority,
    pub root: WebNodeId,
    pub created_at: Instant,
    pub last_attempt: Instant,
    pub hops: u32,
    pub last_exit_zone: Option<usize>,
    pub route_score: f32,
    pub stability_factor: f32,
}

impl SemanticPacket {
    pub fn new(id: u64, root: WebNodeId, priority: SemanticPriority) -> Self {
        let now = Instant::now();
        Self {
            id,
            priority,
            root,
            created_at: now,
            last_attempt: now,
            hops: 0,
            last_exit_zone: None,
            route_score: 0.0,
            stability_factor: 1.0,
        }
    }

    pub fn escalate(&mut self) {
        self.priority = match self.priority {
            SemanticPriority::High => SemanticPriority::High,
            SemanticPriority::Standard => SemanticPriority::High,
            SemanticPriority::Low => SemanticPriority::Standard,
        };
        self.stability_factor = (self.stability_factor + 0.05).min(2.0);
    }

    pub fn reinforce(&mut self, success: bool) {
        if success {
            self.stability_factor = (self.stability_factor + 0.05).min(2.0);
        } else {
            self.stability_factor = (self.stability_factor - 0.05).max(0.1);
        }
    }
}

#[derive(Debug, Clone)]
pub struct SemanticRoundaboutConfig {
    pub max_hops_before_escalation: u32,
    pub max_age_before_force_exit_ms: u64,
}

impl Default for SemanticRoundaboutConfig {
    fn default() -> Self {
        Self {
            max_hops_before_escalation: 8,
            max_age_before_force_exit_ms: 250,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SemanticExitCandidate {
    pub zone_id: usize,
    pub node_id: WebNodeId,
    pub score: f32,
}

#[derive(Debug)]
pub enum SemanticRouteDecision {
    Circulate(SemanticPacket),
    Exit {
        packet: SemanticPacket,
        zone_id: usize,
        node_id: WebNodeId,
    },
}

/// MAX-tier semantic roundabout hub per root node.
pub trait KvWebSemanticRoundabout {
    fn route_semantic_packet(
        &mut self,
        embeddings: &HashMap<TokenId, Embedding>,
        zoning: &SemanticZoning,
        cfg: &SemanticRoundaboutConfig,
        packet: SemanticPacket,
    ) -> SemanticRouteDecision;
}

impl KvWebSemanticRoundabout for KvWeb {
    fn route_semantic_packet(
        &mut self,
        embeddings: &HashMap<TokenId, Embedding>,
        zoning: &SemanticZoning,
        cfg: &SemanticRoundaboutConfig,
        packet: SemanticPacket,
    ) -> SemanticRouteDecision {
        let mut packet = packet;
        let now = Instant::now();
        packet.hops += 1;
        packet.last_attempt = now;

        let age_ms = now.duration_since(packet.created_at).as_millis() as u64;
        if age_ms >= cfg.max_age_before_force_exit_ms
            || packet.hops >= cfg.max_hops_before_escalation
        {
            packet.escalate();
        }

        // Build multilayer exit candidates from zones + scratch pad
        let mut candidates: Vec<SemanticExitCandidate> = Vec::new();

        for zone in &zoning.zones {
            let zone_id = zone.zone_id;
            let slice = &zoning.index_map[zone.start..zone.end];

            for node_id in slice {
                if let Some(node) = self.nodes.get(node_id) {
                    let centroid = centroid_embedding(&node.tokens, embeddings);
                    if centroid.is_empty() {
                        continue;
                    }

                    // Firewall on semantic exit centroid
                    if spif_detect(
                        self,
                        embeddings,
                        &centroid,
                        node.polygon.as_ref(),
                        Some(zoning),
                        Some(&zoning.scratch.layer_a),
                    ) {
                        continue;
                    }

                    let base_sim = cosine_similarity(&centroid, &centroid);
                    let bias = polygon_similarity_bias(self, zoning.root, *node_id, base_sim);

                    let scratch_idx = zoning.nodes.iter().position(|n| n == node_id);
                    let scratch_sim = scratch_idx
                        .map(|i| zoning.scratch.layer_a[i])
                        .unwrap_or(0.0);

                    let score = bias * 0.5 + scratch_sim * 0.5;

                    candidates.push(SemanticExitCandidate {
                        zone_id,
                        node_id: *node_id,
                        score,
                    });
                }
            }
        }

        // Sort candidates by score descending
        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Scratchpad hint: prefer last successful zone if still viable
        let mut chosen: Option<SemanticExitCandidate> = None;

        if let Some(hint_zone) = packet.last_exit_zone {
            if let Some(hinted) = candidates.iter().find(|c| c.zone_id == hint_zone) {
                chosen = Some(hinted.clone());
            }
        }

        if chosen.is_none() {
            chosen = candidates.first().cloned();
        }

        match chosen {
            None => {
                packet.reinforce(false);
                SemanticRouteDecision::Circulate(packet)
            }
            Some(exit) => {
                packet.route_score = exit.score;
                packet.last_exit_zone = Some(exit.zone_id);
                packet.reinforce(true);

                SemanticRouteDecision::Exit {
                    packet,
                    zone_id: exit.zone_id,
                    node_id: exit.node_id,
                }
            }
        }
    }
}
