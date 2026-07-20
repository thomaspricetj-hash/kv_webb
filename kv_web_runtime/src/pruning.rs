//! pruning.rs
//! Pruning physics for KV Web.
//!
//! This module handles:
//! - score decay over time
//! - pruning nodes below thresholds
//! - pruning edges with low weight
//! - cleanup of orphaned tokens
//!
//! This keeps the KV Web lean and relevant.

use kv_web_core::{KvWeb, WebNodeId};
use std::collections::HashSet;

/// Configuration for pruning physics.
#[derive(Debug, Clone)]
pub struct PruningConfig {
    pub score_decay: f32,     // subtract from node score each cycle
    pub min_node_score: f32,  // nodes below this are removed
    pub min_edge_weight: f32, // edges below this are removed
}

/// Extension trait for pruning physics.
pub trait KvWebPruning {
    /// Apply score decay to all nodes.
    fn decay_scores(&mut self, cfg: &PruningConfig);

    /// Remove nodes whose score is too low.
    fn prune_nodes(&mut self, cfg: &PruningConfig);

    /// Remove edges whose weight is too low.
    fn prune_edges(&mut self, cfg: &PruningConfig);

    /// Cleanup orphaned tokens (tokens pointing to deleted nodes).
    fn cleanup_orphan_tokens(&mut self);
}

impl KvWebPruning for KvWeb {
    fn decay_scores(&mut self, cfg: &PruningConfig) {
        for (_, node) in &mut self.nodes {
            node.score -= cfg.score_decay;

            // Prevent runaway negative scores
            if node.score < 0.0 {
                node.score = 0.0;
            }
        }
    }

    fn prune_nodes(&mut self, cfg: &PruningConfig) {
        let mut removed: HashSet<WebNodeId> = HashSet::new();

        // Remove nodes below threshold
        self.nodes.retain(|id, node| {
            let keep = node.score >= cfg.min_node_score;
            if !keep {
                removed.insert(*id);
            }
            keep
        });

        // Remove edges pointing to removed nodes
        self.edges.retain(|e| !removed.contains(&e.to));

        // Rebuild token index
        self.node_index_by_token.clear();
        for (id, node) in &self.nodes {
            for t in &node.tokens {
                self.node_index_by_token.insert(*t, *id);
            }
        }
    }

    fn prune_edges(&mut self, cfg: &PruningConfig) {
        // Remove edges below threshold
        self.edges.retain(|e| e.weight >= cfg.min_edge_weight);
    }

    fn cleanup_orphan_tokens(&mut self) {
        // Remove tokens pointing to deleted nodes
        self.node_index_by_token
            .retain(|_, node_id| self.nodes.contains_key(node_id));
    }
}
