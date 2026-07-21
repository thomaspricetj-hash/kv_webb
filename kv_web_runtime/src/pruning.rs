//! pruning.rs
//! Polygon-aware pruning physics for KV Web + BitDrop_v2 max‑tier compression.
//!
//! Tier‑6 upgrades:
//! - parallel bias computation
//! - safe score decay + node pruning
//! - dual-layer pruning scratch pads (scores + weights)
//! - GPU-ready compressed pruning packets
//!
//! All upgrades are backwards-compatible.

use kv_web_core::{KvWeb, WebNodeId};
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use std::collections::{HashSet, HashMap};

/// Configuration for pruning physics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningConfig {
    pub score_decay: f32,
    pub min_node_score: f32,
    pub min_edge_weight: f32,

    pub face_bonus: f32,
    pub centroid_penalty: f32,
    pub radius_factor: f32,
}

/// Optimization configuration for pruning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningOptimizationConfig {
    pub min_score_decay: f32,
    pub max_score_decay: f32,

    pub min_node_score: f32,
    pub max_node_score: f32,

    pub min_edge_weight: f32,
    pub max_edge_weight: f32,

    pub target_pruned_ratio: f32,
    pub max_pruned_ratio: f32,
}

/// Dual-layer scratch pad for pruning.
/// Layer A = node scores or edge weights
/// Layer B = normalized metric (0..1) for GPU routing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningScratchPad {
    pub layer_a: Vec<f32>,
    pub layer_b: Vec<f32>,
}

/// Build node pruning scratch pad.
fn build_node_pruning_scratch_pad(web: &KvWeb) -> PruningScratchPad {
    let mut layer_a = Vec::with_capacity(web.nodes.len());
    let mut min_s = f32::MAX;
    let mut max_s = f32::MIN;

    for (_, node) in &web.nodes {
        layer_a.push(node.score);
        if node.score < min_s {
            min_s = node.score;
        }
        if node.score > max_s {
            max_s = node.score;
        }
    }

    let span = if max_s > min_s { max_s - min_s } else { 1.0 };

    let mut layer_b = Vec::with_capacity(layer_a.len());
    for s in &layer_a {
        let norm = (*s - min_s) / span;
        layer_b.push(norm);
    }

    PruningScratchPad { layer_a, layer_b }
}

/// Build edge pruning scratch pad.
fn build_edge_pruning_scratch_pad(web: &KvWeb) -> PruningScratchPad {
    let mut layer_a = Vec::with_capacity(web.edges.len());
    let mut min_w = f32::MAX;
    let mut max_w = f32::MIN;

    for e in &web.edges {
        layer_a.push(e.weight);
        if e.weight < min_w {
            min_w = e.weight;
        }
        if e.weight > max_w {
            max_w = e.weight;
        }
    }

    let span = if max_w > min_w { max_w - min_w } else { 1.0 };

    let mut layer_b = Vec::with_capacity(layer_a.len());
    for w in &layer_a {
        let norm = (*w - min_w) / span;
        layer_b.push(norm);
    }

    PruningScratchPad { layer_a, layer_b }
}

/// Polygon-aware score modifier.
fn polygon_prune_bias(web: &KvWeb, node_id: WebNodeId, base: f32, cfg: &PruningConfig) -> f32 {
    let node = match web.nodes.get(&node_id) {
        Some(n) => n,
        None => return base,
    };

    let poly = match &node.polygon {
        Some(p) => p,
        None => return base,
    };

    let face_bonus = match poly.face_index {
        3 => cfg.face_bonus,
        2 => cfg.face_bonus * 0.6,
        1 => cfg.face_bonus * 0.3,
        _ => 0.0,
    };

    let mut centroid_mag = 0.0;
    for v in &poly.centroid {
        centroid_mag += v.abs();
    }

    let centroid_penalty =
        f32::min(centroid_mag / (poly.radius + 1.0), 0.25) * cfg.centroid_penalty;

    let radius_mult = 1.0 / (poly.radius + cfg.radius_factor);

    base + face_bonus - centroid_penalty * radius_mult
}

/// Extension trait for pruning physics.
pub trait KvWebPruning {
    fn decay_scores(&mut self, cfg: &PruningConfig);
    fn prune_nodes(&mut self, cfg: &PruningConfig);
    fn prune_edges(&mut self, cfg: &PruningConfig);
    fn cleanup_orphan_tokens(&mut self);

    fn optimize_pruning(
        &mut self,
        cfg: &mut PruningConfig,
        opt_cfg: &PruningOptimizationConfig,
    ) -> Option<Vec<u8>>;
}

impl KvWebPruning for KvWeb {
    /// Safe polygon-aware score decay: bias computed in parallel, applied sequentially.
    fn decay_scores(&mut self, cfg: &PruningConfig) {
        let ids: Vec<WebNodeId> = self.nodes.keys().cloned().collect();

        // Compute biased scores in parallel without mutating self.
        let biased_scores: HashMap<WebNodeId, f32> = ids
            .par_iter()
            .filter_map(|id| {
                let base = match self.nodes.get(id) {
                    Some(n) => n.score - cfg.score_decay,
                    None => return None,
                };

                let biased = polygon_prune_bias(self, *id, base, cfg);
                Some((*id, biased.max(0.0)))
            })
            .collect();

        // Apply results sequentially.
        for (id, score) in biased_scores {
            if let Some(node) = self.nodes.get_mut(&id) {
                node.score = score;
            }
        }

        if let Some(comp) = &self.compressor {
            let scratch = build_node_pruning_scratch_pad(self);
            let _ = comp.compress(&(
                "decay_scores",
                cfg.score_decay,
                &scratch.layer_a,
                &scratch.layer_b,
            ));
        }
    }

    /// Parallel polygon-aware node pruning (bias map computed in parallel, retain sequential).
    fn prune_nodes(&mut self, cfg: &PruningConfig) {
        let ids: Vec<WebNodeId> = self.nodes.keys().cloned().collect();

        let bias_map: HashMap<WebNodeId, f32> = ids
            .par_iter()
            .filter_map(|id| {
                if let Some(node) = self.nodes.get(id) {
                    let biased = polygon_prune_bias(self, *id, node.score, cfg);
                    Some((*id, biased))
                } else {
                    None
                }
            })
            .collect();

        let mut removed = HashSet::new();

        self.nodes.retain(|id, _node| {
            let biased_score = bias_map[id];
            let keep = biased_score >= cfg.min_node_score;
            if !keep {
                removed.insert(*id);
            }
            keep
        });

        self.edges.retain(|e| !removed.contains(&e.to));

        self.node_index_by_token.clear();
        for (id, node) in &self.nodes {
            for t in &node.tokens {
                self.node_index_by_token.insert(*t, *id);
            }
        }

        if let Some(comp) = &self.compressor {
            let scratch = build_node_pruning_scratch_pad(self);
            let _ = comp.compress(&(
                "prune_nodes",
                cfg.min_node_score,
                &scratch.layer_a,
                &scratch.layer_b,
            ));
        }
    }

    /// Parallel edge pruning via marking + retain.
    fn prune_edges(&mut self, cfg: &PruningConfig) {
        self.edges
            .par_iter_mut()
            .for_each(|e| {
                if e.weight < cfg.min_edge_weight {
                    e.weight = -1.0;
                }
            });

        self.edges.retain(|e| e.weight >= cfg.min_edge_weight);

        if let Some(comp) = &self.compressor {
            let scratch = build_edge_pruning_scratch_pad(self);
            let _ = comp.compress(&(
                "prune_edges",
                cfg.min_edge_weight,
                &scratch.layer_a,
                &scratch.layer_b,
            ));
        }
    }

    fn cleanup_orphan_tokens(&mut self) {
        self.node_index_by_token
            .retain(|_, node_id| self.nodes.contains_key(node_id));

        if let Some(comp) = &self.compressor {
            let _ = comp.compress(&("cleanup_orphan_tokens", self.node_index_by_token.len()));
        }
    }

    fn optimize_pruning(
        &mut self,
        cfg: &mut PruningConfig,
        opt_cfg: &PruningOptimizationConfig,
    ) -> Option<Vec<u8>> {
        let total_nodes = self.nodes.len() as f32;
        if total_nodes == 0.0 {
            return None;
        }

        let mut below_threshold = 0.0;
        for (_, node) in &self.nodes {
            if node.score < cfg.min_node_score {
                below_threshold += 1.0;
            }
        }

        let pruned_ratio = below_threshold / total_nodes;

        if pruned_ratio < opt_cfg.target_pruned_ratio {
            cfg.score_decay =
                (cfg.score_decay * 1.05).min(opt_cfg.max_score_decay);
        } else if pruned_ratio > opt_cfg.max_pruned_ratio {
            cfg.score_decay =
                (cfg.score_decay * 0.9).max(opt_cfg.min_score_decay);
        }

        if pruned_ratio > opt_cfg.max_pruned_ratio {
            cfg.min_node_score =
                (cfg.min_node_score * 0.9).max(opt_cfg.min_node_score);
        } else if pruned_ratio < opt_cfg.target_pruned_ratio {
            cfg.min_node_score =
                (cfg.min_node_score * 1.05).min(opt_cfg.max_node_score);
        }

        let mut low_edges = 0.0;
        for e in &self.edges {
            if e.weight < cfg.min_edge_weight {
                low_edges += 1.0;
            }
        }

        let edge_ratio = if self.edges.is_empty() {
            0.0
        } else {
            low_edges / self.edges.len() as f32
        };

        if edge_ratio > 0.5 {
            cfg.min_edge_weight =
                (cfg.min_edge_weight * 0.9).max(opt_cfg.min_edge_weight);
        } else {
            cfg.min_edge_weight =
                (cfg.min_edge_weight * 1.05).min(opt_cfg.max_edge_weight);
        }

        let scratch_nodes = build_node_pruning_scratch_pad(self);
        let scratch_edges = build_edge_pruning_scratch_pad(self);

        self.compressor.as_ref().map(|c| {
            c.compress(&(
                "optimize_pruning",
                pruned_ratio,
                edge_ratio,
                cfg.score_decay,
                cfg.min_node_score,
                cfg.min_edge_weight,
                &scratch_nodes.layer_a,
                &scratch_nodes.layer_b,
                &scratch_edges.layer_a,
                &scratch_edges.layer_b,
            ))
        })
    }
}

