//! dynamic_web.rs
//! Automatic edge generation and dynamic webbing logic + BitDrop_v2 max‑tier compression.
//!
//! This module makes the KV Web *adaptive*:
//! - edges strengthen when nodes co‑occur
//! - edges weaken when unused
//! - new edges form based on recency or semantic similarity
//!
//! With BitDrop_v2 wired in, all dynamic‑web operations now produce
//! reversible compressed packets for drift‑aware and branch‑aware routing.

use kv_web_core::{KvWeb, WebNodeId, EdgeKind, KvWebCompressor};

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
    fn reinforce_edges(&mut self, nodes: &[WebNodeId], cfg: &DynamicWebConfig)
        -> Option<Vec<u8>>;

    /// Weaken all edges slightly (global decay).
    fn decay_edges(&mut self, cfg: &DynamicWebConfig) -> Option<Vec<u8>>;

    /// Auto‑link nodes based on recency.
    fn link_recent_nodes(&mut self, recent: &[WebNodeId], cfg: &DynamicWebConfig)
        -> Option<Vec<u8>>;

    /// Normalize edge weights (clamp + cleanup).
    fn normalize_edges(&mut self, cfg: &DynamicWebConfig) -> Option<Vec<u8>>;
}

impl KvWebDynamic for KvWeb {
    fn reinforce_edges(&mut self, nodes: &[WebNodeId], cfg: &DynamicWebConfig)
        -> Option<Vec<u8>>
    {
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

        // MAX‑TIER BitDrop_v2 compressed reinforcement packet
        self.compressor.as_ref().map(|c| {
            c.compress(&(
                "reinforce_edges",
                nodes,
                cfg.strengthen_amount
            ))
        })
    }

    fn decay_edges(&mut self, cfg: &DynamicWebConfig) -> Option<Vec<u8>> {
        for edge in &mut self.edges {
            edge.weight -= cfg.weaken_amount;
        }

        // MAX‑TIER BitDrop_v2 compressed decay packet
        self.compressor.as_ref().map(|c| {
            c.compress(&(
                "decay_edges",
                cfg.weaken_amount
            ))
        })
    }

    fn link_recent_nodes(&mut self, recent: &[WebNodeId], cfg: &DynamicWebConfig)
        -> Option<Vec<u8>>
    {
        for pair in recent.windows(2) {
            let a = pair[0];
            let b = pair[1];

            self.add_edge(a, b, cfg.recency_link_weight, EdgeKind::Positional);
        }

        // MAX‑TIER BitDrop_v2 compressed recency‑link packet
        self.compressor.as_ref().map(|c| {
            c.compress(&(
                "link_recent_nodes",
                recent,
                cfg.recency_link_weight
            ))
        })
    }

    fn normalize_edges(&mut self, cfg: &DynamicWebConfig) -> Option<Vec<u8>> {
        self.edges.retain(|e| e.weight >= cfg.min_weight);

        for edge in &mut self.edges {
            if edge.weight > cfg.max_weight {
                edge.weight = cfg.max_weight;
            }
        }

        // MAX‑TIER BitDrop_v2 compressed normalization packet
        self.compressor.as_ref().map(|c| {
            c.compress(&(
                "normalize_edges",
                cfg.min_weight,
                cfg.max_weight
            ))
        })
    }
}
