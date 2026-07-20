//! pruning.rs
//! Polygon-aware pruning physics for KV Web.

use kv_web_core::{KvWeb, WebNodeId};
use std::collections::{HashSet, HashMap};

/// Configuration for pruning physics.
#[derive(Debug, Clone)]
pub struct PruningConfig {
    pub score_decay: f32,
    pub min_node_score: f32,
    pub min_edge_weight: f32,

    pub face_bonus: f32,
    pub centroid_penalty: f32,
    pub radius_factor: f32,
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
}

impl KvWebPruning for KvWeb {
    fn decay_scores(&mut self, cfg: &PruningConfig) {
        // FIX: compute bias BEFORE mutable borrow
        let ids: Vec<WebNodeId> = self.nodes.keys().cloned().collect();

        for id in ids {
            let base = match self.nodes.get(&id) {
                Some(n) => n.score - cfg.score_decay,
                None => continue,
            };

            let biased = polygon_prune_bias(self, id, base, cfg);

            if let Some(node) = self.nodes.get_mut(&id) {
                node.score = biased.max(0.0);
            }
        }
    }

    fn prune_nodes(&mut self, cfg: &PruningConfig) {
        let mut removed = HashSet::new();

        // FIX: compute bias BEFORE retain
        let ids: Vec<WebNodeId> = self.nodes.keys().cloned().collect();
        let mut bias_map = HashMap::new();

        for id in &ids {
            if let Some(node) = self.nodes.get(id) {
                let biased = polygon_prune_bias(self, *id, node.score, cfg);
                bias_map.insert(*id, biased);
            }
        }

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
    }

    fn prune_edges(&mut self, cfg: &PruningConfig) {
        self.edges.retain(|e| e.weight >= cfg.min_edge_weight);
    }

    fn cleanup_orphan_tokens(&mut self) {
        self.node_index_by_token
            .retain(|_, node_id| self.nodes.contains_key(node_id));
    }
}

