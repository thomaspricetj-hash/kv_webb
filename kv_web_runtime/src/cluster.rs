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

    // Optimization metadata
    pub routing_error: f32,
    pub radius_error: f32,
    pub face_index_confidence: f32,
}

/// Clustering configuration.
#[derive(Debug, Clone)]
pub struct ClusterConfig {
    pub min_score: f32,
    pub max_cluster_size: usize,
}

/// Optimization configuration for clusters.
#[derive(Debug, Clone)]
pub struct ClusterOptimizationConfig {
    pub min_radius: f32,
    pub max_radius: f32,
    pub target_face_index_smoothness: f32,
    pub min_score_reinforce: f32,
    pub max_routing_error: f32,
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

                    let (routing_error, radius_error, face_index_confidence) =
                        evaluate_cluster_geometry(web, &chunk, polygon.as_ref());

                    clusters.push(Cluster {
                        id: current_id,
                        nodes: chunk.clone(),
                        label: Some(label.clone()),
                        score,
                        polygon,
                        compressed,
                        routing_error,
                        radius_error,
                        face_index_confidence,
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

                let (routing_error, radius_error, face_index_confidence) =
                    evaluate_cluster_geometry(web, &chunk, polygon.as_ref());

                clusters.push(Cluster {
                    id: current_id,
                    nodes: chunk.clone(),
                    label: Some(label.clone()),
                    score,
                    polygon,
                    compressed,
                    routing_error,
                    radius_error,
                    face_index_confidence,
                });

                current_id += 1;
            }
        }

        Self {
            clusters,
            compressor,
        }
    }

    /// Max‑tier optimization loop over polygonal clusters.
    ///
    /// This loop:
    /// - evaluates geometry quality (radius, centroid fit, face index confidence)
    /// - adjusts polygon radius within bounds
    /// - nudges centroid toward high‑score / high‑connectivity nodes
    /// - reinforces high‑score clusters and de‑emphasizes weak ones
    /// - keeps compressed payloads in sync with updated metadata
    pub fn optimize(
        &mut self,
        web: &KvWeb,
        opt_cfg: &ClusterOptimizationConfig,
    ) {
        for cluster in &mut self.clusters {
            // Skip clusters with no polygon metadata.
            let polygon = match cluster.polygon.as_mut() {
                Some(p) => p,
                None => continue,
            };

            // Re‑evaluate geometry metrics.
            let (routing_error, radius_error, face_index_confidence) =
                evaluate_cluster_geometry(web, &cluster.nodes, Some(polygon));

            cluster.routing_error = routing_error;
            cluster.radius_error = radius_error;
            cluster.face_index_confidence = face_index_confidence;

            // 1) Radius optimization: clamp and smooth radius into configured bounds.
            if polygon.radius < opt_cfg.min_radius {
                polygon.radius = (polygon.radius + opt_cfg.min_radius) * 0.5;
            } else if polygon.radius > opt_cfg.max_radius {
                polygon.radius = (polygon.radius + opt_cfg.max_radius) * 0.5;
            }

            // 2) Centroid optimization: bias centroid toward high‑score / high‑connectivity nodes.
            optimize_centroid(web, &cluster.nodes, polygon);

            // 3) Face index optimization: smooth face index toward target confidence.
            optimize_face_index(polygon, face_index_confidence, opt_cfg.target_face_index_smoothness);

            // 4) Cluster score reinforcement: boost strong clusters, damp weak ones.
            if cluster.score >= opt_cfg.min_score_reinforce && routing_error <= opt_cfg.max_routing_error {
                cluster.score = (cluster.score * 1.05).min(1.0);
            } else {
                cluster.score *= 0.97;
            }

            // 5) Keep compressed payload in sync with updated metadata.
            if let Some(compressor) = &self.compressor {
                if let Some(label) = &cluster.label {
                    cluster.compressed = Some(
                        compressor.compress(&(
                            cluster.id,
                            &cluster.nodes,
                            label,
                            cluster.score,
                            cluster.polygon.as_ref(),
                        ))
                    );
                }
            }
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

/// Evaluate geometry quality for a cluster: routing error, radius error, face index confidence.
fn evaluate_cluster_geometry(
    web: &KvWeb,
    nodes: &[WebNodeId],
    polygon: Option<&PolygonRegion>,
) -> (f32, f32, f32) {
    if nodes.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let mut routing_error = 0.0;
    let mut radius_error = 0.0;
    let mut count = 0.0;

    if let Some(poly) = polygon {
        for id in nodes {
            if let Some(node) = web.nodes.get(id) {
                let conn = web.edges.iter().filter(|e| e.from == node.id).count() as f32;

                let d = (node.score - poly.centroid[0]).abs()
                    + ((node.tokens.len() as f32) - poly.centroid[1]).abs()
                    + (conn - poly.centroid[2]).abs();

                routing_error += d;

                let r_err = (d - poly.radius).abs();
                radius_error += r_err;

                count += 1.0;
            }
        }
    }

    if count > 0.0 {
        routing_error /= count;
        radius_error /= count;
    }

    // Face index confidence: higher when radius is within a reasonable band.
    let face_index_confidence = if let Some(poly) = polygon {
        let r = poly.radius;
        if r < 1.0 {
            0.95
        } else if r < 3.0 {
            0.85
        } else if r < 6.0 {
            0.7
        } else {
            0.5
        }
    } else {
        0.0
    };

    (routing_error, radius_error, face_index_confidence)
}

/// Optimize centroid toward high‑score / high‑connectivity nodes.
fn optimize_centroid(web: &KvWeb, nodes: &[WebNodeId], polygon: &mut PolygonRegion) {
    if nodes.is_empty() {
        return;
    }

    let mut best_score = f32::MIN;
    let mut best_centroid = polygon.centroid.clone();

    for id in nodes {
        if let Some(node) = web.nodes.get(id) {
            let conn = web.edges.iter().filter(|e| e.from == node.id).count() as f32;
            let score = node.score + (conn * 0.01);

            if score > best_score {
                best_score = score;
                best_centroid[0] = node.score;
                best_centroid[1] = node.tokens.len() as f32;
                best_centroid[2] = conn;
            }
        }
    }

    // Smoothly move centroid toward best candidate.
    for i in 0..polygon.centroid.len() {
        polygon.centroid[i] = (polygon.centroid[i] * 0.7) + (best_centroid[i] * 0.3);
    }
}

/// Optimize face index based on confidence and target smoothness.
fn optimize_face_index(
    polygon: &mut PolygonRegion,
    confidence: f32,
    target_smoothness: f32,
) {
    // Higher confidence → keep face index stable, lower confidence → nudge toward mid‑range.
    let mut target_face = polygon.face_index as f32;

    if confidence < 0.6 {
        // Nudge toward a more neutral face index.
        target_face = 1.5;
    }

    let current = polygon.face_index as f32;
    let blended = current * (1.0 - target_smoothness) + target_face * target_smoothness;

    polygon.face_index = blended.round() as u8;
}

