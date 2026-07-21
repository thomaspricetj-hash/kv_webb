//! dynamic_web.rs
//! Automatic edge generation and dynamic webbing logic + BitDrop_v2 max‑tier compression.
//!
//! This module makes the KV Web *adaptive*:
//! - edges strengthen when nodes co‑occur
//! - edges weaken when unused
//! - new edges form based on recency or semantic similarity
//!
//! Tier‑6 upgrades:
//! - parallel edge reinforcement + decay
//! - dual-layer scratch pads (weights + normalized geometry)
//! - GPU-ready compressed packets
//!
//! All upgrades are backwards-compatible.

use kv_web_core::{KvWeb, WebNodeId, EdgeKind, KvWebCompressor};
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use std::collections::HashSet;

/// Dynamic webbing configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicWebConfig {
    pub strengthen_amount: f32,   // how much to increase weight when nodes co‑occur
    pub weaken_amount: f32,       // how much to decrease weight when unused
    pub min_weight: f32,          // edges below this are removed
    pub max_weight: f32,          // clamp edge weight
    pub recency_link_weight: f32, // weight for auto‑linking recent nodes
}

/// Dynamic web optimization configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicWebOptimizationConfig {
    pub min_strengthen_amount: f32,
    pub max_strengthen_amount: f32,
    pub min_weaken_amount: f32,
    pub max_weaken_amount: f32,
    pub min_weight_span: f32,
    pub max_weight_span: f32,
    pub min_recency_link_weight: f32,
    pub max_recency_link_weight: f32,
}

/// Dual-layer scratch pad for dynamic web.
/// Layer A = raw edge weights
/// Layer B = normalized weights (0..1) for GPU routing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicWebScratchPad {
    pub layer_a: Vec<f32>,
    pub layer_b: Vec<f32>,
}

/// Build dual-layer scratch pad from current edges.
fn build_dynamic_web_scratch_pad(web: &KvWeb) -> DynamicWebScratchPad {
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

    DynamicWebScratchPad { layer_a, layer_b }
}

/// Extension trait for dynamic webbing.
pub trait KvWebDynamic {
    fn reinforce_edges(&mut self, nodes: &[WebNodeId], cfg: &DynamicWebConfig)
        -> Option<Vec<u8>>;

    fn decay_edges(&mut self, cfg: &DynamicWebConfig) -> Option<Vec<u8>>;

    fn link_recent_nodes(&mut self, recent: &[WebNodeId], cfg: &DynamicWebConfig)
        -> Option<Vec<u8>>;

    fn normalize_edges(&mut self, cfg: &DynamicWebConfig) -> Option<Vec<u8>>;

    fn optimize_dynamic_web(
        &mut self,
        cfg: &mut DynamicWebConfig,
        opt_cfg: &DynamicWebOptimizationConfig,
    ) -> Option<Vec<u8>>;
}

impl KvWebDynamic for KvWeb {
    /// Parallel reinforcement: edges whose endpoints are in `nodes` get strengthened.
    fn reinforce_edges(&mut self, nodes: &[WebNodeId], cfg: &DynamicWebConfig)
        -> Option<Vec<u8>>
    {
        let set: HashSet<WebNodeId> = nodes.iter().cloned().collect();

        self.edges
            .par_iter_mut()
            .for_each(|edge| {
                if set.contains(&edge.from) && set.contains(&edge.to) {
                    edge.weight += cfg.strengthen_amount;
                }
            });

        let scratch = build_dynamic_web_scratch_pad(self);

        self.compressor.as_ref().map(|c| {
            c.compress(&(
                "reinforce_edges",
                nodes,
                cfg.strengthen_amount,
                &scratch.layer_a,
                &scratch.layer_b,
            ))
        })
    }

    /// Parallel global decay.
    fn decay_edges(&mut self, cfg: &DynamicWebConfig) -> Option<Vec<u8>> {
        self.edges
            .par_iter_mut()
            .for_each(|edge| {
                edge.weight -= cfg.weaken_amount;
            });

        let scratch = build_dynamic_web_scratch_pad(self);

        self.compressor.as_ref().map(|c| {
            c.compress(&(
                "decay_edges",
                cfg.weaken_amount,
                &scratch.layer_a,
                &scratch.layer_b,
            ))
        })
    }

    /// Recency-based linking (kept serial; typically small).
    fn link_recent_nodes(&mut self, recent: &[WebNodeId], cfg: &DynamicWebConfig)
        -> Option<Vec<u8>>
    {
        for pair in recent.windows(2) {
            let a = pair[0];
            let b = pair[1];

            self.add_edge(a, b, cfg.recency_link_weight, EdgeKind::Positional);
        }

        let scratch = build_dynamic_web_scratch_pad(self);

        self.compressor.as_ref().map(|c| {
            c.compress(&(
                "link_recent_nodes",
                recent,
                cfg.recency_link_weight,
                &scratch.layer_a,
                &scratch.layer_b,
            ))
        })
    }

    /// Parallel clamp + cleanup.
    fn normalize_edges(&mut self, cfg: &DynamicWebConfig) -> Option<Vec<u8>> {
        self.edges.retain(|e| e.weight >= cfg.min_weight);

        self.edges
            .par_iter_mut()
            .for_each(|edge| {
                if edge.weight > cfg.max_weight {
                    edge.weight = cfg.max_weight;
                }
            });

        let scratch = build_dynamic_web_scratch_pad(self);

        self.compressor.as_ref().map(|c| {
            c.compress(&(
                "normalize_edges",
                cfg.min_weight,
                cfg.max_weight,
                &scratch.layer_a,
                &scratch.layer_b,
            ))
        })
    }

    /// Max-tier optimization loop over dynamic web parameters + scratch pad.
    fn optimize_dynamic_web(
        &mut self,
        cfg: &mut DynamicWebConfig,
        opt_cfg: &DynamicWebOptimizationConfig,
    ) -> Option<Vec<u8>> {
        if self.edges.is_empty() {
            return None;
        }

        let mut min_w = f32::MAX;
        let mut max_w = f32::MIN;
        let mut total_w = 0.0;
        let mut count = 0.0;

        for edge in &self.edges {
            if edge.weight < min_w {
                min_w = edge.weight;
            }
            if edge.weight > max_w {
                max_w = edge.weight;
            }
            total_w += edge.weight;
            count += 1.0;
        }

        let avg_w = total_w / count;
        let span = if max_w > min_w { max_w - min_w } else { 0.0 };

        if span < opt_cfg.min_weight_span {
            cfg.strengthen_amount =
                (cfg.strengthen_amount * 1.05).min(opt_cfg.max_strengthen_amount);
        } else if span > opt_cfg.max_weight_span {
            cfg.strengthen_amount =
                (cfg.strengthen_amount * 0.9).max(opt_cfg.min_strengthen_amount);
        }

        if avg_w > cfg.max_weight * 0.8 {
            cfg.weaken_amount =
                (cfg.weaken_amount * 1.05).min(opt_cfg.max_weaken_amount);
        } else if avg_w < cfg.min_weight * 1.2 {
            cfg.weaken_amount =
                (cfg.weaken_amount * 0.9).max(opt_cfg.min_weaken_amount);
        }

        if span > opt_cfg.max_weight_span {
            cfg.recency_link_weight =
                (cfg.recency_link_weight * 0.9).max(opt_cfg.min_recency_link_weight);
        } else if span < opt_cfg.min_weight_span {
            cfg.recency_link_weight =
                (cfg.recency_link_weight * 1.05).min(opt_cfg.max_recency_link_weight);
        }

        let scratch = build_dynamic_web_scratch_pad(self);

        self.compressor.as_ref().map(|c| {
            c.compress(&(
                "optimize_dynamic_web",
                cfg.strengthen_amount,
                cfg.weaken_amount,
                cfg.recency_link_weight,
                avg_w,
                span,
                &scratch.layer_a,
                &scratch.layer_b,
            ))
        })
    }
}
