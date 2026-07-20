//! cluster.rs
//!
//! Semantic clustering over KV Web nodes + BitDrop_v2 max‑tier compression.
//! Groups nodes into clusters based on labels, scores, and simple heuristics.

use kv_web_core::{KvWeb, WebNodeId, KvWebCompressor};
use std::collections::HashMap;

/// A simple cluster: just a set of node IDs.
#[derive(Debug, Clone)]
pub struct Cluster {
    pub id: usize,
    pub nodes: Vec<WebNodeId>,
    pub label: Option<String>,
    pub score: f32,

    // MAX‑TIER BitDrop_v2 compressed payload
    pub compressed: Option<Vec<u8>>,
}

/// Clustering configuration.
#[derive(Debug, Clone)]
pub struct ClusterConfig {
    pub min_score: f32,
    pub max_cluster_size: usize,
}

pub struct KvWebClusters {
    pub clusters: Vec<Cluster>,

    // Optional compressor for cluster snapshots
    pub compressor: Option<KvWebCompressor>,
}

impl KvWebClusters {
    pub fn from_web(web: &KvWeb, cfg: &ClusterConfig) -> Self {
        let mut clusters = Vec::new();
        let mut current_id = 0;

        let compressor = web.compressor.clone();

        // naive clustering: group by label
        let mut by_label: HashMap<String, Vec<WebNodeId>> = HashMap::new();

        for (id, node) in &web.nodes {
            if node.score < cfg.min_score {
                continue;
            }

            let key = node
                .label
                .clone()
                .unwrap_or_else(|| format!("unlabeled_{}", id.0));

            by_label.entry(key).or_default().push(*id);
        }

        for (label, nodes) in by_label {
            let mut chunk = Vec::new();
            for n in nodes {
                chunk.push(n);
                if chunk.len() >= cfg.max_cluster_size {
                    let score = avg_score(web, &chunk);

                    let compressed = compressor.as_ref().map(|c| {
                        c.compress(&(
                            current_id,
                            &chunk,
                            &label,
                            score
                        ))
                    });

                    clusters.push(Cluster {
                        id: current_id,
                        nodes: chunk.clone(),
                        label: Some(label.clone()),
                        score,
                        compressed,
                    });

                    current_id += 1;
                    chunk.clear();
                }
            }

            if !chunk.is_empty() {
                let score = avg_score(web, &chunk);

                let compressed = compressor.as_ref().map(|c| {
                    c.compress(&(
                        current_id,
                        &chunk,
                        &label,
                        score
                    ))
                });

                clusters.push(Cluster {
                    id: current_id,
                    nodes: chunk.clone(),
                    label: Some(label.clone()),
                    score,
                    compressed,
                });

                current_id += 1;
            }
        }

        Self {
            clusters,
            compressor,
        }
    }
}

fn avg_score(web: &KvWeb, nodes: &[WebNodeId]) -> f32 {
    if nodes.is_empty() {
        return 0.0;
    }

    let mut sum = 0.0;
    let mut count = 0;

    for id in nodes {
        if let Some(node) = web.nodes.get(id) {
            sum += node.score;
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        sum / count as f32
    }
}

