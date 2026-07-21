//! kv_web_runtime
//! Runtime logic for KV‑cache webbing + BitDrop_v2 max‑tier compression
//! + Polygonal‑KV geometry.

pub mod semantic;
pub mod dynamic_web;
pub mod pruning;
pub mod drift;

// Newly wired modules (4–8)
pub mod cluster;
pub mod embedding;
pub mod graph_ops;
pub mod heatmap;

use kv_web_core::{KvWeb, WebNodeId, TokenId, WebNode, KvWebCompressor};
use std::collections::{HashSet, VecDeque};

/// Runtime extensions for KvWeb.
/// This is intentionally separate from the core crate so the core stays pure.
pub trait KvWebRuntime {
    fn neighbors(&self, node: WebNodeId) -> Vec<&WebNode>;
    fn tokens_in_region(&self, root: WebNodeId, depth: usize) -> HashSet<TokenId>;
    fn tokens_in_region_compressed(&self, root: WebNodeId, depth: usize) -> Option<Vec<u8>>;
    fn nodes_in_region(&self, root: WebNodeId, depth: usize) -> HashSet<WebNodeId>;
    fn nodes_in_region_compressed(&self, root: WebNodeId, depth: usize) -> Option<Vec<u8>>;
    fn region_score(&self, root: WebNodeId, depth: usize) -> f32;
    fn prune_low_score(&mut self, min_score: f32);
    fn merge_nodes(&mut self, a: WebNodeId, b: WebNodeId) -> WebNodeId;
    fn merge_nodes_compressed(&mut self, a: WebNodeId, b: WebNodeId) -> Option<Vec<u8>>;

    /// MAX‑TIER unified optimization loop.
    fn optimize_runtime(&mut self);
}

/// Polygon‑aware neighbor bias (same‑face, centroid‑close).
fn polygon_neighbor_bias(web: &KvWeb, a: WebNodeId, b: WebNodeId) -> f32 {
    let na = match web.nodes.get(&a) {
        Some(n) => n,
        None => return 1.0,
    };
    let nb = match web.nodes.get(&b) {
        Some(n) => n,
        None => return 1.0,
    };

    let pa = match &na.polygon {
        Some(p) => p,
        None => return 1.0,
    };
    let pb = match &nb.polygon {
        Some(p) => p,
        None => return 1.0,
    };

    let face_bonus = if pa.face_index == pb.face_index { 1.25 } else { 1.0 };

    let mut dist = 0.0;
    for (ca, cb) in pa.centroid.iter().zip(pb.centroid.iter()) {
        dist += (ca - cb).abs();
    }
    let radius = pa.radius + pb.radius + 1.0;
    let penalty = (dist / radius).min(0.5);

    face_bonus - penalty
}

// ============================================================================
// MAX‑TIER RUNTIME SCHEDULER (local to kv_web_runtime)
// ============================================================================

#[derive(Debug, Clone)]
pub struct RuntimeSchedulerConfig {
    pub default_root: WebNodeId,
    pub default_depth: usize,
}

#[derive(Debug, Clone)]
pub struct RuntimeSchedulerState {
    pub ticks: usize,
}

impl Default for RuntimeSchedulerState {
    fn default() -> Self {
        Self { ticks: 0 }
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeScheduler {
    pub cfg: RuntimeSchedulerConfig,
    pub state: RuntimeSchedulerState,
}

impl RuntimeScheduler {
    pub fn new(cfg: RuntimeSchedulerConfig) -> Self {
        Self {
            cfg,
            state: RuntimeSchedulerState::default(),
        }
    }

    pub fn tick(&mut self, web: &mut KvWeb) {
        self.state.ticks += 1;

        // Simple global sanity pass: prune extremely low scores
        let min_score = 0.0;
        web.prune_low_score(min_score);
    }
}

impl KvWebRuntime for KvWeb {
    fn neighbors(&self, node: WebNodeId) -> Vec<&WebNode> {
        let mut out = Vec::new();
        for edge in &self.edges {
            if edge.from == node {
                if let Some(n) = self.nodes.get(&edge.to) {
                    out.push(n);
                }
            }
        }
        out
    }

    fn tokens_in_region(&self, root: WebNodeId, depth: usize) -> HashSet<TokenId> {
        let mut visited_nodes: HashSet<WebNodeId> = HashSet::new();
        let mut out_tokens: HashSet<TokenId> = HashSet::new();

        let mut queue: VecDeque<(WebNodeId, usize)> = VecDeque::new();
        queue.push_back((root, 0));

        while let Some((current, d)) = queue.pop_front() {
            if d > depth || visited_nodes.contains(&current) {
                continue;
            }
            visited_nodes.insert(current);

            if let Some(node) = self.nodes.get(&current) {
                for t in &node.tokens {
                    out_tokens.insert(*t);
                }
            }

            for edge in &self.edges {
                if edge.from == current {
                    let bias = polygon_neighbor_bias(self, current, edge.to);
                    if bias <= 0.0 {
                        continue;
                    }
                    queue.push_back((edge.to, d + 1));
                }
            }
        }

        out_tokens
    }

    fn tokens_in_region_compressed(&self, root: WebNodeId, depth: usize) -> Option<Vec<u8>> {
        let tokens = self.tokens_in_region(root, depth);
        self.compressor.as_ref().map(|c| c.compress(&tokens))
    }

    fn nodes_in_region(&self, root: WebNodeId, depth: usize) -> HashSet<WebNodeId> {
        let mut visited: HashSet<WebNodeId> = HashSet::new();
        let mut queue: VecDeque<(WebNodeId, usize)> = VecDeque::new();
        queue.push_back((root, 0));

        while let Some((current, d)) = queue.pop_front() {
            if d > depth || visited.contains(&current) {
                continue;
            }
            visited.insert(current);

            for edge in &self.edges {
                if edge.from == current {
                    let bias = polygon_neighbor_bias(self, current, edge.to);
                    if bias <= 0.0 {
                        continue;
                    }
                    queue.push_back((edge.to, d + 1));
                }
            }
        }

        visited
    }

    fn nodes_in_region_compressed(&self, root: WebNodeId, depth: usize) -> Option<Vec<u8>> {
        let nodes = self.nodes_in_region(root, depth);
        self.compressor.as_ref().map(|c| c.compress(&nodes))
    }

    fn region_score(&self, root: WebNodeId, depth: usize) -> f32 {
        let nodes = self.nodes_in_region(root, depth);
        nodes
            .iter()
            .filter_map(|id| self.nodes.get(id))
            .map(|n| n.score)
            .sum()
    }

    fn prune_low_score(&mut self, min_score: f32) {
        self.nodes.retain(|_, node| node.score >= min_score);
        self.edges.retain(|edge| self.nodes.contains_key(&edge.to));

        self.node_index_by_token.clear();
        for (id, node) in &self.nodes {
            for t in &node.tokens {
                self.node_index_by_token.insert(*t, *id);
            }
        }
    }

    fn merge_nodes(&mut self, a: WebNodeId, b: WebNodeId) -> WebNodeId {
        let new_id = WebNodeId(self.nodes.len());

        let mut tokens = Vec::new();
        let mut score = 0.0;
        let mut label = None;
        let mut polygon = None;

        if let Some(na) = self.nodes.get(&a) {
            tokens.extend(na.tokens.clone());
            score += na.score;
            label = na.label.clone();
            polygon = na.polygon.clone();
        }

        if let Some(nb) = self.nodes.get(&b) {
            tokens.extend(nb.tokens.clone());
            score += nb.score;
            if label.is_none() {
                label = nb.label.clone();
            }

            if let (Some(pa), Some(pb)) = (polygon.clone(), nb.polygon.clone()) {
                let mut centroid = pa.centroid.clone();
                if centroid.len() == pb.centroid.len() {
                    for (c, v) in centroid.iter_mut().zip(pb.centroid.iter()) {
                        *c = (*c + *v) / 2.0;
                    }
                }
                let radius = (pa.radius + pb.radius) / 2.0;
                let face_index = pa.face_index.max(pb.face_index);

                polygon = Some(kv_web_core::PolygonRegion {
                    id: pa.id,
                    centroid,
                    radius,
                    face_index,
                });
            }
        }

        self.nodes.remove(&a);
        self.nodes.remove(&b);

        let merged = WebNode {
            id: new_id,
            tokens: tokens.clone(),
            tokens_compressed: None,
            label,
            label_compressed: None,
            score,
            branch_id: None,
            branch_kind: None,
            branch_stability: 0.0,
            branch_drift: 0.0,
            branch_meta_compressed: None,
            polygon,
        };

        self.nodes.insert(new_id, merged);

        for t in tokens {
            self.node_index_by_token.insert(t, new_id);
        }

        self.edges.retain(|e| e.from != a && e.from != b);

        new_id
    }

    fn merge_nodes_compressed(&mut self, a: WebNodeId, b: WebNodeId) -> Option<Vec<u8>> {
        let new_id = self.merge_nodes(a, b);
        let merged = self.nodes.get(&new_id)?;

        self.compressor.as_ref().map(|c| {
            c.compress(&(
                merged.id,
                merged.tokens.clone(),
                merged.label.clone(),
                merged.score,
                merged.branch_id,
                merged.branch_kind,
                merged.branch_stability,
                merged.branch_drift,
                merged.polygon.as_ref()
            ))
        })
    }

    // -------------------------------------------------------------------------
    // ⭐ MAX‑TIER UNIFIED OPTIMIZATION LOOP
    // -------------------------------------------------------------------------
    fn optimize_runtime(&mut self) {
        use crate::{
            cluster::KvWebClusters,
            dynamic_web::{DynamicWebConfig, DynamicWebOptimizationConfig, KvWebDynamic},
            drift::{DriftConfig, DriftOptimizationConfig, KvWebDrift},
            pruning::{PruningConfig, PruningOptimizationConfig, KvWebPruning},
            embedding::EmbeddingOptimizationConfig,
            graph_ops::{GraphOpsOptimizationConfig, optimize_graph_ops},
            heatmap::{HeatmapOptimizationConfig, optimize_heatmap},
            semantic::{SemanticOptimizationConfig, KvWebSemantic},
        };

        // 1) Drift optimization
        let mut drift_cfg = DriftConfig {
            decay_rate: 0.01,
            edge_decay_rate: 0.01,
            mode: drift::DriftMode::Exponential,
            reinforcement_amount: 0.05,
        };
        let drift_opt = DriftOptimizationConfig {
            min_decay_rate: 0.001,
            max_decay_rate: 0.05,
            min_edge_decay_rate: 0.001,
            max_edge_decay_rate: 0.05,
            max_allowed_score_drop: 0.5,
            min_reinforcement_amount: 0.01,
            max_reinforcement_amount: 0.2,
        };
        self.optimize_drift(&mut drift_cfg, &drift_opt);

        // 2) Dynamic web optimization
        let mut dyn_cfg = DynamicWebConfig {
            strengthen_amount: 0.05,
            weaken_amount: 0.01,
            min_weight: 0.01,
            max_weight: 2.0,
            recency_link_weight: 0.05,
        };
        let dyn_opt = DynamicWebOptimizationConfig {
            min_strengthen_amount: 0.01,
            max_strengthen_amount: 0.2,
            min_weaken_amount: 0.001,
            max_weaken_amount: 0.05,
            min_weight_span: 0.1,
            max_weight_span: 2.0,
            min_recency_link_weight: 0.01,
            max_recency_link_weight: 0.2,
        };
        self.optimize_dynamic_web(&mut dyn_cfg, &dyn_opt);

        // 3) Pruning optimization
        let mut prune_cfg = PruningConfig {
            score_decay: 0.01,
            min_node_score: 0.1,
            min_edge_weight: 0.01,
            face_bonus: 0.1,
            centroid_penalty: 0.1,
            radius_factor: 1.0,
        };
        let prune_opt = PruningOptimizationConfig {
            min_score_decay: 0.001,
            max_score_decay: 0.05,
            min_node_score: 0.05,
            max_node_score: 1.0,
            min_edge_weight: 0.001,
            max_edge_weight: 0.2,
            target_pruned_ratio: 0.1,
            max_pruned_ratio: 0.4,
        };
        self.optimize_pruning(&mut prune_cfg, &prune_opt);

        // 4) Semantic optimization
        let mut sim_threshold = 0.7;
        let sem_opt = SemanticOptimizationConfig {
            min_similarity_threshold: 0.4,
            max_similarity_threshold: 0.95,
            target_cluster_size: 5,
            max_cluster_size: 20,
            min_radius: 0.3,
            max_radius: 0.9,
        };
        let embeddings = std::collections::HashMap::new();
        self.optimize_semantic(&embeddings, &mut sim_threshold, &sem_opt);

        // 5) Graph ops optimization
        let mut depth = 3;
        let mut damping = 0.85;
        let graph_opt = GraphOpsOptimizationConfig {
            min_damping: 0.5,
            max_damping: 0.95,
            target_bfs_size: 10,
            max_bfs_size: 50,
            min_depth: 1,
            max_depth: 8,
        };
        optimize_graph_ops(self, WebNodeId(0), &mut depth, &mut damping, &graph_opt);

        // 6) Heatmap optimization
        let mut smoothing_strength = 1.0;
        let heat_opt = HeatmapOptimizationConfig {
            min_smoothing_strength: 0.5,
            max_smoothing_strength: 2.0,
            target_heat_variance: 0.05,
            max_heat_variance: 0.2,
        };
        optimize_heatmap(self, 4096, &mut smoothing_strength, &heat_opt);

        // 7) Cluster optimization
        let cfg = cluster::ClusterConfig {
            min_score: 0.1,
            max_cluster_size: 50,
        };
        let mut clusters = cluster::KvWebClusters::from_web(self, &cfg);
        let cluster_opt = cluster::ClusterOptimizationConfig {
            min_radius: 0.1,
            max_radius: 5.0,
            target_face_index_smoothness: 0.2,
            min_score_reinforce: 0.2,
            max_routing_error: 1.0,
        };
        clusters.optimize(self, &cluster_opt);

        // 8) Runtime scheduler tick (max‑tier global pass)
        let mut scheduler = RuntimeScheduler::new(RuntimeSchedulerConfig {
            default_root: WebNodeId(0),
            default_depth: 3,
        });
        scheduler.tick(self);
    }
}
