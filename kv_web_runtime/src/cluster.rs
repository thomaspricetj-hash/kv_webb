//! cluster.rs
//!
//! Semantic clustering over KV Web nodes + BitDrop_v2 max‑tier compression
//! + Polygon‑KV geometry upgrade.
//!
//! Tier‑6 upgrades:
//! - parallel geometry evaluation
//! - cluster indexing + zoning
//! - dual-layer scratch pads (score + geometry)
//! - GPU-ready compressed cluster packets
//!
//! Max-tier upgrade:
//! - Cross-link grids (clusters/zones → nodes)
//! - Revolving-door routing over clusters (entry/exit + flow)
//! - Fusion field over clusters (score + geometry + door flow)
//! - Embedded Roundabout logic (heatmaps + predictor + smoothing + memory + solver)
//!
//! All upgrades are backwards-compatible.

use kv_web_core::{KvWeb, WebNodeId, KvWebCompressor, PolygonRegion};
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// A cluster with polygonal KV metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cluster {
    pub id: usize,
    pub nodes: Vec<WebNodeId>,
    pub label: Option<String>,
    pub score: f32,

    pub polygon: Option<PolygonRegion>,
    pub compressed: Option<Vec<u8>>,

    pub routing_error: f32,
    pub radius_error: f32,
    pub face_index_confidence: f32,
}

/// Clustering configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    pub min_score: f32,
    pub max_cluster_size: usize,
}

/// Optimization configuration for clusters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterOptimizationConfig {
    pub min_radius: f32,
    pub max_radius: f32,
    pub target_face_index_smoothness: f32,
    pub min_score_reinforce: f32,
    pub max_routing_error: f32,
}

/// Dual-layer scratch pad for clusters.
/// Layer A = cluster scores
/// Layer B = geometry metric (radius or routing error)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterScratchPad {
    pub layer_a: Vec<f32>,
    pub layer_b: Vec<f32>,
}

/// Zoning + indexing for clusters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterZone {
    pub zone_id: usize,
    pub start: usize,
    pub end: usize,
    pub centroid_cluster: Option<usize>,
    pub size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterZoning {
    pub indices: Vec<usize>,
    pub zones: Vec<ClusterZone>,
    pub scratch: ClusterScratchPad,
}

/// Cross-link grid over clusters.
/// Links zones and clusters to node sets for routing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterCrossLinkGrid {
    /// Cluster ID → nodes
    pub cluster_to_nodes: Vec<Vec<WebNodeId>>,
    /// Zone ID → cluster indices
    pub zone_to_clusters: Vec<Vec<usize>>,
}

/// Revolving door over clusters.
/// Entry/exit clusters + flow scalar.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterRevolvingDoor {
    pub door_id: usize,
    pub entry_cluster_id: usize,
    pub exit_cluster_id: usize,
    pub flow_strength: f32,
}

/// Fusion field over clusters.
/// Combines score, geometry, and door flow into a single bias.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterFusionField {
    pub fused_score: Vec<f32>,
}

/// Roundabout cluster predictor configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundaboutClusterPredictorConfig {
    pub passes: usize,
    pub min_bias: f32,
    pub max_bias: f32,
    pub smoothing_strength: f32,
}

/// Roundabout cluster chain (a routed path through cluster indices).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundaboutClusterChain {
    pub cluster_ids: Vec<usize>,
    pub total_bias: f32,
}

/// Roundabout cluster pattern memory entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundaboutClusterPattern {
    pub chain: RoundaboutClusterChain,
    pub weight: f32,
}

/// Roundabout cluster pattern memory with decay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundaboutClusterPatternMemory {
    pub patterns: Vec<RoundaboutClusterPattern>,
    pub decay: f32,
}

/// Roundabout solver result for clusters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundaboutClusterSolverResult {
    pub chosen_cluster_id: usize,
    pub bias: f32,
}

pub struct KvWebClusters {
    pub clusters: Vec<Cluster>,
    pub compressor: Option<KvWebCompressor>,
}

impl KvWebClusters {
    pub fn from_web(web: &KvWeb, cfg: &ClusterConfig) -> Self {
        let mut clusters = Vec::new();
        let mut current_id = 0;
        let mut polygon_id_counter: u32 = 1;

        let compressor = web.compressor.clone();

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

    /// Build dual-layer scratch pad for current clusters.
    fn build_scratch_pad(&self) -> ClusterScratchPad {
        let mut layer_a = Vec::with_capacity(self.clusters.len());
        let mut layer_b = Vec::with_capacity(self.clusters.len());

        for c in &self.clusters {
            layer_a.push(c.score);
            let geom = if c.radius_error > 0.0 {
                c.radius_error
            } else {
                c.routing_error
            };
            layer_b.push(geom);
        }

        ClusterScratchPad { layer_a, layer_b }
    }

    /// Build cluster indexing + zoning.
    pub fn build_zoning(&self, num_zones: usize) -> ClusterZoning {
        let mut indices: Vec<usize> = (0..self.clusters.len()).collect();

        indices.sort_by(|&a, &b| {
            let sa = self.clusters[a].score;
            let sb = self.clusters[b].score;
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });

        let num_zones = num_zones.max(1);
        let len = indices.len();
        let zone_size = (len as f32 / num_zones as f32).ceil() as usize;

        let mut zones = Vec::new();
        let mut zone_id = 0;
        let mut start = 0;

        while start < len {
            let end = (start + zone_size).min(len);
            let slice = &indices[start..end];

            let centroid_cluster = slice.get(slice.len() / 2).cloned();

            zones.push(ClusterZone {
                zone_id,
                start,
                end,
                centroid_cluster,
                size: slice.len(),
            });

            zone_id += 1;
            start = end;
        }

        let scratch = self.build_scratch_pad();

        ClusterZoning {
            indices,
            zones,
            scratch,
        }
    }

    /// Compressed zoning (GPU-ready).
    pub fn build_zoning_compressed(&self, num_zones: usize) -> Option<Vec<u8>> {
        let zoning = self.build_zoning(num_zones);
        self.compressor.as_ref().map(|c| {
            c.compress(&(
                "cluster_zoning",
                &zoning.indices,
                &zoning.zones,
                &zoning.scratch.layer_a,
                &zoning.scratch.layer_b,
            ))
        })
    }

    /// Max‑tier optimization loop over polygonal clusters.
    pub fn optimize(
        &mut self,
        web: &KvWeb,
        opt_cfg: &ClusterOptimizationConfig,
    ) {
        self.clusters
            .par_iter_mut()
            .for_each(|cluster| {
                let polygon = match cluster.polygon.as_mut() {
                    Some(p) => p,
                    None => return,
                };

                let (routing_error, radius_error, face_index_confidence) =
                    evaluate_cluster_geometry(web, &cluster.nodes, Some(polygon));

                cluster.routing_error = routing_error;
                cluster.radius_error = radius_error;
                cluster.face_index_confidence = face_index_confidence;

                if polygon.radius < opt_cfg.min_radius {
                    polygon.radius = (polygon.radius + opt_cfg.min_radius) * 0.5;
                } else if polygon.radius > opt_cfg.max_radius {
                    polygon.radius = (polygon.radius + opt_cfg.max_radius) * 0.5;
                }

                optimize_centroid(web, &cluster.nodes, polygon);
                optimize_face_index(polygon, face_index_confidence, opt_cfg.target_face_index_smoothness);

                if cluster.score >= opt_cfg.min_score_reinforce && routing_error <= opt_cfg.max_routing_error {
                    cluster.score = (cluster.score * 1.05).min(1.0);
                } else {
                    cluster.score *= 0.97;
                }

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
            });

        if let Some(compressor) = &self.compressor {
            let scratch = self.build_scratch_pad();
            let _ = compressor.compress(&(
                "optimize_clusters_scratch",
                &scratch.layer_a,
                &scratch.layer_b,
            ));
        }
    }

    /// Build cross-link grid from current clusters + zoning.
    pub fn build_cross_link_grid(&self, num_zones: usize) -> ClusterCrossLinkGrid {
        let zoning = self.build_zoning(num_zones);

        let mut cluster_to_nodes = Vec::with_capacity(self.clusters.len());
        for c in &self.clusters {
            cluster_to_nodes.push(c.nodes.clone());
        }

        let mut zone_to_clusters = Vec::new();
        for zone in &zoning.zones {
            let mut indices = Vec::new();
            for i in zone.start..zone.end {
                indices.push(zoning.indices[i]);
            }
            zone_to_clusters.push(indices);
        }

        ClusterCrossLinkGrid {
            cluster_to_nodes,
            zone_to_clusters,
        }
    }

    /// Build revolving doors over clusters using zoning.
    pub fn build_revolving_doors(
        &self,
        num_zones: usize,
    ) -> Vec<ClusterRevolvingDoor> {
        let zoning = self.build_zoning(num_zones);
        let mut doors = Vec::new();
        let mut door_id = 0usize;

        if zoning.zones.len() < 2 {
            return doors;
        }

        // Simple heuristic: first zone = entry, last zone = exit.
        let entry_zone = &zoning.zones[0];
        let exit_zone = &zoning.zones[zoning.zones.len() - 1];

        let entry_cluster_id = entry_zone
            .centroid_cluster
            .unwrap_or(zoning.indices[entry_zone.start]);
        let exit_cluster_id = exit_zone
            .centroid_cluster
            .unwrap_or(zoning.indices[exit_zone.start]);

        let entry_score = self.clusters[entry_cluster_id].score;
        let exit_score = self.clusters[exit_cluster_id].score;
        let flow_strength = (exit_score - entry_score).abs();

        doors.push(ClusterRevolvingDoor {
            door_id,
            entry_cluster_id,
            exit_cluster_id,
            flow_strength,
        });

        doors
    }

    /// Build fusion field over clusters from scores + geometry + doors.
    pub fn build_fusion_field(
        &self,
        doors: &[ClusterRevolvingDoor],
    ) -> ClusterFusionField {
        let mut fused = Vec::with_capacity(self.clusters.len());

        for (idx, c) in self.clusters.iter().enumerate() {
            let mut bias = c.score;

            // Geometry contribution: lower routing_error and radius_error → higher bias.
            let geom_penalty = (c.routing_error + c.radius_error) * 0.1;
            bias *= (1.0 - geom_penalty).max(0.0);

            // Door flow contribution: if cluster is entry/exit, boost.
            for door in doors {
                if door.entry_cluster_id == idx || door.exit_cluster_id == idx {
                    bias *= 1.0 + door.flow_strength * 0.05;
                }
            }

            fused.push(bias);
        }

        ClusterFusionField { fused_score: fused }
    }

    /// Run roundabout cluster predictor: multi-pass chain over fused scores.
    pub fn run_roundabout_predictor(
        &self,
        fusion: &ClusterFusionField,
        cfg: &RoundaboutClusterPredictorConfig,
    ) -> RoundaboutClusterChain {
        let mut cluster_ids = Vec::new();
        let mut total = 0.0f32;

        let mut visited = vec![false; fusion.fused_score.len()];

        for _pass in 0..cfg.passes {
            let mut best_id = None;
            let mut best_bias = cfg.min_bias;

            for (i, b) in fusion.fused_score.iter().enumerate() {
                if visited[i] {
                    continue;
                }
                if *b > best_bias && *b <= cfg.max_bias {
                    best_bias = *b;
                    best_id = Some(i);
                }
            }

            if let Some(id) = best_id {
                visited[id] = true;
                cluster_ids.push(id);
                total += best_bias;
            } else {
                break;
            }
        }

        RoundaboutClusterChain {
            cluster_ids,
            total_bias: total,
        }
    }

    /// Smooth roundabout cluster chain by local averaging over fused scores.
    pub fn smooth_roundabout_chain(
        &self,
        chain: &mut RoundaboutClusterChain,
        fusion: &ClusterFusionField,
        strength: f32,
    ) {
        if chain.cluster_ids.len() < 3 {
            return;
        }

        let mut new_total = 0.0f32;

        for (i, id) in chain.cluster_ids.iter().enumerate() {
            let mut local_sum = 0.0f32;
            let mut local_count = 0.0f32;

            for j in i.saturating_sub(1)..=(i + 1).min(chain.cluster_ids.len() - 1) {
                let nid = chain.cluster_ids[j];
                if nid < fusion.fused_score.len() {
                    local_sum += fusion.fused_score[nid];
                    local_count += 1.0;
                }
            }

            if local_count > 0.0 {
                let avg = local_sum / local_count;
                let base = if *id < fusion.fused_score.len() {
                    fusion.fused_score[*id]
                } else {
                    0.0
                };
                new_total += avg * strength + base * (1.0 - strength);
            }
        }

        chain.total_bias = new_total;
    }

    /// Update cluster pattern memory with new chain, applying decay.
    pub fn update_pattern_memory(
        &self,
        memory: &mut RoundaboutClusterPatternMemory,
        chain: &RoundaboutClusterChain,
    ) {
        for pattern in &mut memory.patterns {
            pattern.weight *= memory.decay;
        }

        memory.patterns.push(RoundaboutClusterPattern {
            chain: chain.clone(),
            weight: 1.0,
        });

        memory.patterns.retain(|p| p.weight > 0.01);
    }

    /// Apply roundabout bias to fused scores using pattern memory.
    pub fn apply_roundabout_bias(
        &self,
        fusion: &mut ClusterFusionField,
        memory: &RoundaboutClusterPatternMemory,
    ) {
        let mut fused = fusion.fused_score.clone();

        for pattern in &memory.patterns {
            let boost = pattern.weight * 0.05;
            for id in &pattern.chain.cluster_ids {
                if *id < fused.len() {
                    fused[*id] *= 1.0 + boost;
                }
            }
        }

        // Normalize fused scores.
        let mut max = 0.0f32;
        for v in &fused {
            if *v > max {
                max = *v;
            }
        }
        if max > 0.0 {
            for v in &mut fused {
                *v /= max;
            }
        }

        fusion.fused_score = fused;
    }

    /// Run roundabout cluster solver: choose final cluster using fused scores + chain + memory.
    pub fn run_roundabout_solver(
        &self,
        fusion: &ClusterFusionField,
        chain: &RoundaboutClusterChain,
        memory: &RoundaboutClusterPatternMemory,
    ) -> RoundaboutClusterSolverResult {
        // Prefer last cluster in chain if available.
        if let Some(&last) = chain.cluster_ids.last() {
            let bias = fusion.fused_score.get(last).copied().unwrap_or(0.0);
            return RoundaboutClusterSolverResult {
                chosen_cluster_id: last,
                bias,
            };
        }

        // Fallback: choose max fused score cluster.
        let mut best_id = 0usize;
        let mut best_bias = f32::MIN;

        for (i, b) in fusion.fused_score.iter().enumerate() {
            if *b > best_bias {
                best_bias = *b;
                best_id = i;
            }
        }

        // Light bias from memory: if any pattern contains best_id, boost bias slightly.
        let mut final_bias = best_bias;
        for pattern in &memory.patterns {
            if pattern.chain.cluster_ids.contains(&best_id) {
                final_bias *= 1.05;
            }
        }

        RoundaboutClusterSolverResult {
            chosen_cluster_id: best_id,
            bias: final_bias,
        }
    }

    /// Compressed roundabout cluster pipeline: cross-link + doors + fusion + predictor + smoothing + memory + solver.
    pub fn roundabout_cluster_pipeline_compressed(
        &self,
        web: &KvWeb,
        num_zones: usize,
        predictor_cfg: &RoundaboutClusterPredictorConfig,
        memory: &mut RoundaboutClusterPatternMemory,
    ) -> Option<Vec<u8>> {
        let grid = self.build_cross_link_grid(num_zones);
        let doors = self.build_revolving_doors(num_zones);
        let mut fusion = self.build_fusion_field(&doors);

        let mut chain = self.run_roundabout_predictor(&fusion, predictor_cfg);
        self.smooth_roundabout_chain(&mut chain, &fusion, predictor_cfg.smoothing_strength);

        self.update_pattern_memory(memory, &chain);
        self.apply_roundabout_bias(&mut fusion, memory);

        let result = self.run_roundabout_solver(&fusion, &chain, memory);

        self.compressor.as_ref().map(|c| {
            c.compress(&(
                "roundabout_cluster_pipeline",
                web.nodes.len(),
                num_zones,
                predictor_cfg.passes,
                &grid.cluster_to_nodes,
                &grid.zone_to_clusters,
                &fusion.fused_score,
                &chain.cluster_ids,
                chain.total_bias,
                &memory.patterns,
                result.chosen_cluster_id,
                result.bias,
            ))
        })
    }
}

/// Build polygonal KV region metadata for a cluster.
fn build_polygon_region(web: &KvWeb, nodes: &[WebNodeId], polygon_id: u32) -> Option<PolygonRegion> {
    if nodes.is_empty() {
        return None;
    }

    let mut centroid = vec![0.0; 3];
    let mut count = 0.0;

    for id in nodes {
        if let Some(node) = web.nodes.get(id) {
            centroid[0] += node.score;
            centroid[1] += node.tokens.len() as f32;

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

    for i in 0..polygon.centroid.len() {
        polygon.centroid[i] = (polygon.centroid[i] * 0.7) + (best_centroid[i] * 0.3);
    }
}

fn optimize_face_index(
    polygon: &mut PolygonRegion,
    confidence: f32,
    target_smoothness: f32,
) {
    let mut target_face = polygon.face_index as f32;

    if confidence < 0.6 {
        target_face = 1.5;
    }

    let current = polygon.face_index as f32;
    let blended = current * (1.0 - target_smoothness) + target_face * target_smoothness;

    polygon.face_index = blended.round() as u8;
}


