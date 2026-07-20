//! dynamic_web.rs
//! Automatic edge generation and dynamic webbing logic.
//!
//! This module makes the KV Web *adaptive*:
//! - edges strengthen when nodes co‑occur
//! - edges weaken when unused
//! - new edges form based on recency or semantic similarity
//!
//! This creates a living, self‑adjusting graph.

use kv_web_core::{KvWeb, WebNodeId, EdgeKind};

/// Dynamic webbing configuration.
#[derive(Debug, Clone)]
pub struct DynamicWebConfig {
    pub strengthen_amount: f32,   // how much to increase weight when nodes co‑occur
    pub weaken_amount: f32,       // how much to decrease weight when unused
    pub min_weight: f32,          // edges below this are removed
    pub max_weight: f32,          // clamp edge weight
    pub recency_link_weight: f32, // weight for auto‑linking recent nodes
}

/// Extension trait for dynamic webbing.
pub trait KvWebDynamic {
    /// Strengthen edges between nodes that appear together.
    fn reinforce_edges(&mut self, nodes: &[WebNodeId], cfg: &DynamicWebConfig);

    /// Weaken all edges slightly (global decay).
    fn decay_edges(&mut self, cfg: &DynamicWebConfig);

    /// Auto‑link nodes based on recency.
    fn link_recent_nodes(&mut self, recent: &[WebNodeId], cfg: &DynamicWebConfig);

    /// Normalize edge weights (clamp + cleanup).
    fn normalize_edges(&mut self, cfg: &DynamicWebConfig);
}

impl KvWebDynamic for KvWeb {
    fn reinforce_edges(&mut self, nodes: &[WebNodeId], cfg: &DynamicWebConfig) {
        // Strengthen edges between all pairs of nodes.
        for (i, a) in nodes.iter().enumerate() {
            for b in nodes.iter().skip(i + 1) {
                for edge in &mut self.edges {
                    if edge.from == *a && edge.to == *b {
                        edge.weight += cfg.strengthen_amount;
                    }
                    if edge.from == *b && edge.to == *a {
                        edge.weight += cfg.strengthen_amount;
                    }
                }
            }
        }
    }

    fn decay_edges(&mut self, cfg: &DynamicWebConfig) {
        // Apply global decay.
        for edge in &mut self.edges {
            edge.weight -= cfg.weaken_amount;
        }
    }

    fn link_recent_nodes(&mut self, recent: &[WebNodeId], cfg: &DynamicWebConfig) {
        // Link each node to the next one in the recency list.
        for pair in recent.windows(2) {
            let a = pair[0];
            let b = pair[1];

            // Add a recency edge.
            self.add_edge(a, b, cfg.recency_link_weight, EdgeKind::Positional);
        }
    }

    fn normalize_edges(&mut self, cfg: &DynamicWebConfig) {
        // Remove edges below min_weight.
        self.edges.retain(|e| e.weight >= cfg.min_weight);

        // Clamp edges to max_weight.
        for edge in &mut self.edges {
            if edge.weight > cfg.max_weight {
                edge.weight = cfg.max_weight;
            }
        }
    }
}
