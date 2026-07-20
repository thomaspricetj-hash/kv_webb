//! cluster.rs
//!
//! Semantic clustering over KV Web nodes + BitDrop_v2 max‑tier compression
//! + Polygonal‑KV geometry upgrade.
//!
//! Each cluster now forms a polygonal semantic region with:
//! - centroid
//! - radius
//! - face index
//! - polygon id
//!
//! These polygonal regions compress extremely well under BitDrop_v2 and
//! allow geometric KV routing (multi‑facet semantic faces).

use kv_web_core::{KvWeb, WebNodeId, KvWebCompressor, PolygonRegion};
use std::collections::HashMap;

/// A cluster with polygonal KV metadata.
#[derive(Debug, Clone)]
pub struct Cluster {
    pub id: usize,
    pub nodes: Vec<WebNodeId>,
    pub label: Option<String>,
    pub score: f32,

    // Polygonal KV metadata
    pub polygon: Option<PolygonRegion>,

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
        let mut polygon_id_counter: u32 = 1;

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

                    let polygon = build_polygon_region(web, &chunk, polygon_id_counter);
                    polygon_id_counter += 1;

                    let compressed = compressor.as_ref().map(|c| {
                        c.compress(&(
                            current_id,
                            &chunk,
                            &label,
                            score,
                            polygon.as_ref()
                        ))
                    });

                    clusters.push(Cluster {
                        id: current_id,
                        nodes: chunk.clone(),
                        label: Some(label.clone()),
                        score,
                        polygon,
                        compressed,
                    });

                    current_id += 1;
                    chunk.clear();
                }
            }

            if !chunk.is_empty() {
                let score = avg_score(web, &chunk);

                let polygon = build_polygon_region(web, &chunk, polygon_id_counter);
                polygon_id_counter += 1;

                let compressed = compressor.as_ref().map(|c| {
                    c.compress(&(
                        current_id,
                        &chunk,
                        &label,
                        score,
                        polygon.as_ref()
                    ))
                });

                clusters.push(Cluster {
                    id: current_id,
                    nodes: chunk.clone(),
                    label: Some(label.clone()),
                    score,
                    polygon,
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

/// Build polygonal KV region metadata for a cluster.
fn build_polygon_region(web: &KvWeb, nodes: &[WebNodeId], polygon_id: u32) -> Option<PolygonRegion> {
    if nodes.is_empty() {
        return None;
    }

    // Compute centroid from node scores, token count, and connectivity.
    let mut centroid = vec![0.0; 3]; // [score, token_density, connectivity]
    let mut count = 0.0;

    for id in nodes {
        if let Some(node) = web.nodes.get(id) {
            centroid[0] += node.score;                 // semantic magnitude
            centroid[1] += node.tokens.len() as f32;   // density

            // connectivity: number of outgoing edges from this node
            let conn = web.edges.iter().filter(|e| e.from == node.id).count() as f32;
            centroid[2] += conn;

            count += 1.0;
        }
    }

    if count > 0.0 {
        for v in &mut centroid {
            *v /= count;
        }
    }

    // Radius: average deviation from centroid
    let mut radius_acc = 0.0;
    let mut radius_count = 0.0;

    for id in nodes {
        if let Some(node) = web.nodes.get(id) {
            let conn = web.edges.iter().filter(|e| e.from == node.id).count() as f32;

            let d = (node.score - centroid[0]).abs()
                + ((node.tokens.len() as f32) - centroid[1]).abs()
                + (conn - centroid[2]).abs();

            radius_acc += d;
            radius_count += 1.0;
        }
    }

    let radius = if radius_count > 0.0 {
        radius_acc / radius_count
    } else {
        1.0
    };

    // Face index: simple bucket based on radius
    let face_index = if radius < 1.0 {
        3
    } else if radius < 3.0 {
        2
    } else if radius < 6.0 {
        1
    } else {
        0
    };

    Some(PolygonRegion {
        id: polygon_id,
        centroid,
        radius,
        face_index,
    })
}

/// Compute average score for a cluster.
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

