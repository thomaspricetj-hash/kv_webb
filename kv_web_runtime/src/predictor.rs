//! predictor.rs
//!
//! Global predictor layer for KV Web + BitDrop_v2 + Polygonal-KV geometry.
//!
//! Adds:
//! - node activity prediction
//! - edge activation prediction
//! - region (subgraph) routing prediction
//! - dual-layer predictor scratch pads
//! - roundabout-style predictor memory
//! - GPU-ready compressed predictor packets
//!
//! All upgrades are backwards-compatible.

use kv_web_core::{KvWeb, WebNodeId, KvWebCompressor};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Predictor configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvWebPredictorConfig {
    pub min_activity: f32,
    pub max_activity: f32,
    pub drift_weight: f32,
    pub edge_weight: f32,
    pub heat_weight: f32,
    pub prune_weight: f32,
}

/// Predictor scratch pad.
/// Layer A = raw predictor scores
/// Layer B = normalized predictor scores (0..1)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvWebPredictorScratchPad {
    pub layer_a: Vec<f32>,
    pub layer_b: Vec<f32>,
}

/// Roundabout predictor memory entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvWebPredictorPattern {
    pub nodes: Vec<WebNodeId>,
    pub weight: f32,
}

/// Roundabout predictor memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvWebPredictorMemory {
    pub patterns: Vec<KvWebPredictorPattern>,
    pub decay: f32,
}

/// Predictor result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvWebPredictorResult {
    pub top_nodes: Vec<WebNodeId>,
    pub scores: Vec<f32>,
}

/// Compressed predictor packet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvWebPredictorPacket {
    pub tag: &'static str,
    pub top_nodes: Vec<WebNodeId>,
    pub scores: Vec<f32>,
    pub scratch_a: Vec<f32>,
    pub scratch_b: Vec<f32>,
    pub patterns: Vec<KvWebPredictorPattern>,
}

/// Build predictor scratch pad.
fn build_predictor_scratch_pad(scores: &[f32]) -> KvWebPredictorScratchPad {
    let mut layer_a = scores.to_vec();
    let mut max = 0.0f32;
    for v in &layer_a {
        if *v > max {
            max = *v;
        }
    }

    let mut layer_b = Vec::with_capacity(layer_a.len());
    if max > 0.0 {
        for v in &layer_a {
            layer_b.push(*v / max);
        }
    } else {
        layer_b.resize(layer_a.len(), 0.0);
    }

    KvWebPredictorScratchPad { layer_a, layer_b }
}

/// Extension trait for predictor.
pub trait KvWebPredictor {
    fn predict_activity(
        &self,
        cfg: &KvWebPredictorConfig,
        memory: &mut KvWebPredictorMemory,
        top_k: usize,
    ) -> KvWebPredictorResult;

    fn predict_activity_compressed(
        &self,
        cfg: &KvWebPredictorConfig,
        memory: &mut KvWebPredictorMemory,
        top_k: usize,
    ) -> Option<Vec<u8>>;
}

impl KvWebPredictor for KvWeb {
    /// Core predictor: compute node activity scores from multiple signals.
    fn predict_activity(
        &self,
        cfg: &KvWebPredictorConfig,
        memory: &mut KvWebPredictorMemory,
        top_k: usize,
    ) -> KvWebPredictorResult {
        if self.nodes.is_empty() {
            return KvWebPredictorResult {
                top_nodes: Vec::new(),
                scores: Vec::new(),
            };
        }

        let mut scores = Vec::with_capacity(self.nodes.len());
        let mut ids = Vec::with_capacity(self.nodes.len());

        // Base scores from node.score + edge degree.
        let mut degree_map: HashMap<WebNodeId, f32> = HashMap::new();
        for e in &self.edges {
            *degree_map.entry(e.from).or_insert(0.0) += 1.0;
        }

        for (id, node) in &self.nodes {
            let degree = degree_map.get(id).copied().unwrap_or(0.0);
            let mut s = 0.0f32;

            // Core node score.
            s += node.score * cfg.heat_weight;

            // Edge degree.
            s += degree * cfg.edge_weight;

            // Polygon radius / face index as a stability signal.
            if let Some(poly) = &node.polygon {
                let radius_term = (1.0 / (poly.radius + 1.0)).min(1.0);
                let face_term = match poly.face_index {
                    3 => 1.0,
                    2 => 0.7,
                    1 => 0.4,
                    _ => 0.2,
                };
                s += radius_term * face_term * cfg.prune_weight;
            }

            scores.push(s);
            ids.push(*id);
        }

        // Apply memory bias.
        for pattern in &memory.patterns {
            let boost = pattern.weight * 0.05;
            for nid in &pattern.nodes {
                if let Some(pos) = ids.iter().position(|x| x == nid) {
                    scores[pos] *= 1.0 + boost;
                }
            }
        }

        // Decay memory.
        for pattern in &mut memory.patterns {
            pattern.weight *= memory.decay;
        }
        memory.patterns.retain(|p| p.weight > 0.01);

        // Sort by score.
        let mut idxs: Vec<usize> = (0..ids.len()).collect();
        idxs.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let k = top_k.min(idxs.len());
        let mut top_nodes = Vec::with_capacity(k);
        let mut top_scores = Vec::with_capacity(k);

        for i in 0..k {
            let idx = idxs[i];
            top_nodes.push(ids[idx]);
            top_scores.push(scores[idx]);
        }

        // Update memory with new pattern.
        memory.patterns.push(KvWebPredictorPattern {
            nodes: top_nodes.clone(),
            weight: 1.0,
        });

        KvWebPredictorResult {
            top_nodes,
            scores: top_scores,
        }
    }

    /// Compressed predictor packet (GPU-ready).
    fn predict_activity_compressed(
        &self,
        cfg: &KvWebPredictorConfig,
        memory: &mut KvWebPredictorMemory,
        top_k: usize,
    ) -> Option<Vec<u8>> {
        let result = self.predict_activity(cfg, memory, top_k);

        let scratch = build_predictor_scratch_pad(&result.scores);

        let packet = KvWebPredictorPacket {
            tag: "kv_web_predictor",
            top_nodes: result.top_nodes.clone(),
            scores: result.scores.clone(),
            scratch_a: scratch.layer_a.clone(),
            scratch_b: scratch.layer_b.clone(),
            patterns: memory.patterns.clone(),
        };

        self.compressor.as_ref().map(|c| c.compress(&packet))
    }
}
