//! graph_ops.rs
//!
//! General graph operations over KvWeb + BitDrop_v2 max‑tier compression
//! + Polygonal-KV geometry upgrade.
//!
//! Adds:
//! - polygon-aware BFS expansion
//! - polygon-biased PageRank
//! - face-index semantic weighting
//! - dual-layer scratch pads (graph + semantic geometry)
//! - per-region indexing + zoning
//! - parallel BFS + PageRank
//! - GPU-ready compressed graph packets
//!
//! Max-tier upgrade:
//! - Cross-link grids (cluster/tag/door → nodes)
//! - Revolving-door routing (entry/exit + flow vectors)
//! - Fusion field (load + semantic + geometry + door flow)
//! - Routing solver (bias-aware, GPU-ready packets)
//! - Embedded Roundabout logic (heatmaps, predictor, smoothing, memory, bias, solver)
//!
//! All upgrades are backwards-compatible.

use kv_web_core::{KvWeb, WebNodeId};
use crate::KvWebRuntime; // needed for web.neighbors(...)
use rayon::prelude::*;
use serde::{Serialize, Deserialize};

/// Graph ops optimization configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphOpsOptimizationConfig {
    pub min_damping: f32,
    pub max_damping: f32,
    pub target_bfs_size: usize,
    pub max_bfs_size: usize,
    pub min_depth: usize,
    pub max_depth: usize,
}

/// Dual-layer scratch pad for graph ops.
/// Layer A = raw graph metrics (depth, span, size)
/// Layer B = semantic geometry metrics (polygon bias, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphScratchPad {
    pub layer_a: Vec<f32>,
    pub layer_b: Vec<f32>,
}

/// A single zone inside a BFS region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Zone {
    pub zone_id: usize,
    pub start: usize,
    pub end: usize,
    pub centroid_node: Option<WebNodeId>,
    pub size: usize,
}

/// Zoning + indexing + scratch pad for a BFS region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionZoning {
    pub root: WebNodeId,
    pub nodes: Vec<WebNodeId>,      // BFS nodes
    pub index_map: Vec<WebNodeId>,  // nodes sorted by polygon bias / rank
    pub zones: Vec<Zone>,           // semantic zones
    pub scratch: GraphScratchPad,   // dual-layer scratch pad
}

/// Polygon-aware neighbor weighting.
/// Nodes in the same polygon face get priority.
/// Nodes closer in centroid space get boosted.
fn polygon_neighbor_bias(web: &KvWeb, a: WebNodeId, b: WebNodeId) -> f32 {
    let node_a = match web.nodes.get(&a) {
        Some(n) => n,
        None => return 1.0,
    };
    let node_b = match web.nodes.get(&b) {
        Some(n) => n,
        None => return 1.0,
    };

    let poly_a = match &node_a.polygon {
        Some(p) => p,
        None => return 1.0,
    };
    let poly_b = match &node_b.polygon {
        Some(p) => p,
        None => return 1.0,
    };

    // Face match bonus
    let face_bonus = if poly_a.face_index == poly_b.face_index {
        1.25
    } else {
        1.0
    };

    // Centroid distance penalty
    let mut dist = 0.0;
    for (ca, cb) in poly_a.centroid.iter().zip(poly_b.centroid.iter()) {
        dist += (ca - cb).abs();
    }

    let radius = poly_a.radius + poly_b.radius + 1.0;
    let centroid_penalty = (dist / radius).min(0.5);

    face_bonus - centroid_penalty
}

/// BFS region expansion with polygonal bias.
pub fn bfs_region(web: &KvWeb, root: WebNodeId, depth: usize) -> Vec<WebNodeId> {
    use std::collections::{HashSet, VecDeque};

    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    let mut result = Vec::new();

    visited.insert(root);
    queue.push_back((root, 0));

    while let Some((current, d)) = queue.pop_front() {
        result.push(current);

        if d >= depth {
            continue;
        }

        for neighbor in web.neighbors(current) {
            let nid = neighbor.id;

            if visited.contains(&nid) {
                continue;
            }

            // Polygon-aware gating
            let bias = polygon_neighbor_bias(web, current, nid);
            if bias <= 0.0 {
                continue;
            }

            visited.insert(nid);
            queue.push_back((nid, d + 1));
        }
    }

    result
}

/// Build dual-layer scratch pad for a BFS region.
fn build_dual_layer_scratch_pad_for_bfs(
    web: &KvWeb,
    root: WebNodeId,
    bfs_nodes: &[WebNodeId],
    depth: usize,
) -> GraphScratchPad {
    // Layer A: raw BFS depth encoded per node
    let layer_a = vec![depth as f32; bfs_nodes.len()];

    // Layer B: semantic geometry (polygon bias) per node
    let mut layer_b = Vec::with_capacity(bfs_nodes.len());
    for node in bfs_nodes {
        layer_b.push(polygon_neighbor_bias(web, root, *node));
    }

    GraphScratchPad { layer_a, layer_b }
}

/// Build indexing + zoning + dual-layer scratch pad for a BFS region.
pub fn bfs_region_index_and_zone(
    web: &KvWeb,
    root: WebNodeId,
    depth: usize,
    num_zones: usize,
) -> RegionZoning {
    let nodes = bfs_region(web, root, depth);

    // Index map: sort nodes by polygon bias relative to root (descending).
    let mut index_map = nodes.clone();
    index_map.sort_by(|a, b| {
        let ba = polygon_neighbor_bias(web, root, *a);
        let bb = polygon_neighbor_bias(web, root, *b);
        bb.partial_cmp(&ba).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Zoning: split index_map into contiguous zones.
    let num_zones = num_zones.max(1);
    let len = index_map.len();
    let zone_size = (len as f32 / num_zones as f32).ceil() as usize;

    let mut zones = Vec::new();
    let mut zone_id = 0;
    let mut start = 0;

    while start < len {
        let end = (start + zone_size).min(len);
        let slice = &index_map[start..end];

        let centroid_node = slice.get(slice.len() / 2).cloned();
        zones.push(Zone {
            zone_id,
            start,
            end,
            centroid_node,
            size: slice.len(),
        });

        zone_id += 1;
        start = end;
    }

    // Dual-layer scratch pad for this region
    let scratch = build_dual_layer_scratch_pad_for_bfs(web, root, &nodes, depth);

    RegionZoning {
        root,
        nodes,
        index_map,
        zones,
        scratch,
    }
}

/// Compressed BFS region expansion.
pub fn bfs_region_compressed(web: &KvWeb, root: WebNodeId, depth: usize) -> Option<Vec<u8>> {
    let nodes = bfs_region(web, root, depth);
    web.compressor.as_ref().map(|c| {
        c.compress(&("bfs_region", root, depth, &nodes))
    })
}

/// Compressed BFS region with indexing + zoning + scratch pad (GPU-ready).
pub fn bfs_region_index_and_zone_compressed(
    web: &KvWeb,
    root: WebNodeId,
    depth: usize,
    num_zones: usize,
) -> Option<Vec<u8>> {
    let rz = bfs_region_index_and_zone(web, root, depth, num_zones);
    web.compressor.as_ref().map(|c| {
        c.compress(&(
            "bfs_region_index_and_zone",
            root,
            depth,
            num_zones,
            &rz.nodes,
            &rz.index_map,
            &rz.zones,
            &rz.scratch.layer_a,
            &rz.scratch.layer_b,
        ))
    })
}

/// Parallel BFS over multiple roots.
pub fn bfs_region_parallel(
    web: &KvWeb,
    roots: &[WebNodeId],
    depth: usize,
) -> Vec<(WebNodeId, Vec<WebNodeId>)> {
    roots
        .par_iter()
        .map(|root| (*root, bfs_region(web, *root, depth)))
        .collect()
}

/// Parallel BFS with indexing + zoning + scratch pad.
pub fn bfs_region_parallel_index_and_zone(
    web: &KvWeb,
    roots: &[WebNodeId],
    depth: usize,
    num_zones: usize,
) -> Vec<RegionZoning> {
    roots
        .par_iter()
        .map(|root| bfs_region_index_and_zone(web, *root, depth, num_zones))
        .collect()
}

/// Compressed parallel BFS with indexing + zoning + scratch pad (GPU-ready).
pub fn bfs_region_parallel_index_and_zone_compressed(
    web: &KvWeb,
    roots: &[WebNodeId],
    depth: usize,
    num_zones: usize,
) -> Option<Vec<u8>> {
    let regions = bfs_region_parallel_index_and_zone(web, roots, depth, num_zones);
    web.compressor.as_ref().map(|c| {
        c.compress(&(
            "bfs_region_parallel_index_and_zone",
            depth,
            num_zones,
            &regions,
        ))
    })
}

/// Polygon-biased PageRank.
/// Nodes in the same polygon face reinforce each other.
/// Centroid distance reduces rank flow.
pub fn pagerank(web: &KvWeb, iterations: usize, damping: f32) -> Vec<(WebNodeId, f32)> {
    use std::collections::HashMap;

    let n = web.nodes.len();
    if n == 0 {
        return Vec::new();
    }

    let ids: Vec<WebNodeId> = web.nodes.keys().cloned().collect();

    let mut rank: HashMap<WebNodeId, f32> =
        ids.iter().map(|id| (*id, 1.0 / n as f32)).collect();

    for _ in 0..iterations {
        let mut new_rank: HashMap<WebNodeId, f32> =
            ids.iter().map(|id| (*id, (1.0 - damping) / n as f32)).collect();

        for edge in &web.edges {
            let out_rank = rank[&edge.from];

            // Polygon-aware rank flow
            let bias = polygon_neighbor_bias(web, edge.from, edge.to);

            *new_rank.entry(edge.to).or_insert(0.0) += damping * out_rank * bias;
        }

        rank = new_rank;
    }

    let mut result: Vec<(WebNodeId, f32)> = rank.into_iter().collect();
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    result
}

/// Build dual-layer scratch pad for PageRank (global graph view).
fn build_dual_layer_scratch_pad_for_pagerank(
    web: &KvWeb,
    pr: &[(WebNodeId, f32)],
    _damping: f32,
) -> GraphScratchPad {
    // Layer A: PageRank scores
    let mut layer_a = Vec::with_capacity(pr.len());
    // Layer B: semantic geometry (polygon bias vs top-ranked node)
    let mut layer_b = Vec::with_capacity(pr.len());

    let top_node = pr.first().map(|(id, _)| *id);

    for (id, score) in pr {
        layer_a.push(*score);
        let bias = if let Some(top) = top_node {
            polygon_neighbor_bias(web, top, *id)
        } else {
            1.0
        };
        layer_b.push(bias);
    }

    GraphScratchPad { layer_a, layer_b }
}

/// Compressed PageRank.
pub fn pagerank_compressed(web: &KvWeb, iterations: usize, damping: f32) -> Option<Vec<u8>> {
    let pr = pagerank(web, iterations, damping);
    web.compressor.as_ref().map(|c| {
        c.compress(&("pagerank", iterations, damping, &pr))
    })
}

/// PageRank + dual-layer scratch pad (GPU-ready).
pub fn pagerank_with_scratch_compressed(
    web: &KvWeb,
    iterations: usize,
    damping: f32,
) -> Option<Vec<u8>> {
    let pr = pagerank(web, iterations, damping);
    let scratch = build_dual_layer_scratch_pad_for_pagerank(web, &pr, damping);

    web.compressor.as_ref().map(|c| {
        c.compress(&(
            "pagerank_with_scratch",
            iterations,
            damping,
            &pr,
            &scratch.layer_a,
            &scratch.layer_b,
        ))
    })
}

/// Parallel PageRank over multiple damping factors.
pub fn pagerank_parallel(
    web: &KvWeb,
    iterations: usize,
    dampings: &[f32],
) -> Vec<(f32, Vec<(WebNodeId, f32)>)> {
    dampings
        .par_iter()
        .map(|d| (*d, pagerank(web, iterations, *d)))
        .collect()
}

/// Compressed parallel PageRank (GPU-ready).
pub fn pagerank_parallel_compressed(
    web: &KvWeb,
    iterations: usize,
    dampings: &[f32],
) -> Option<Vec<u8>> {
    let prs = pagerank_parallel(web, iterations, dampings);
    web.compressor.as_ref().map(|c| {
        c.compress(&("pagerank_parallel", iterations, &prs))
    })
}

/// Max-tier optimization loop for BFS depth and PageRank damping.
pub fn optimize_graph_ops(
    web: &KvWeb,
    root: WebNodeId,
    depth: &mut usize,
    damping: &mut f32,
    cfg: &GraphOpsOptimizationConfig,
) -> Option<Vec<u8>> {
    // Measure BFS region size.
    let bfs_nodes = bfs_region(web, root, *depth);
    let bfs_size = bfs_nodes.len();

    // Measure PageRank spread (simple span between max and min rank).
    let pr = pagerank(web, 16, *damping);
    let mut min_r = f32::MAX;
    let mut max_r = f32::MIN;

    for (_, r) in &pr {
        if *r < min_r {
            min_r = *r;
        }
        if *r > max_r {
            max_r = *r;
        }
    }

    let pr_span = if max_r > min_r { max_r - min_r } else { 0.0 };

    // Adjust depth based on BFS size.
    if bfs_size < cfg.target_bfs_size && *depth < cfg.max_depth {
        *depth += 1;
    } else if bfs_size > cfg.max_bfs_size && *depth > cfg.min_depth {
        *depth -= 1;
    }

    // Adjust damping based on PageRank span.
    if pr_span < 0.01 {
        *damping = (*damping * 1.05).min(cfg.max_damping);
    } else if pr_span > 0.2 {
        *damping = (*damping * 0.95).max(cfg.min_damping);
    }

    // Optional: build scratch pad for optimizer GPU packets
    let scratch = build_dual_layer_scratch_pad_for_bfs(web, root, &bfs_nodes, *depth);

    web.compressor.as_ref().map(|c| {
        c.compress(&(
            "optimize_graph_ops",
            root,
            bfs_size,
            pr_span,
            *depth,
            *damping,
            &scratch.layer_a,
            &scratch.layer_b,
        ))
    })
}

//
// ──────────────────────────────────────────────────────────────
//   MAX‑TIER UPGRADE: CROSS‑LINK GRID + REVOLVING DOORS + FUSION FIELD
// ──────────────────────────────────────────────────────────────
//

/// Cross-link grid over the KvWeb graph.
/// Links clusters, tags, and doors to node sets for routing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCrossLinkGrid {
    /// Cluster ID → nodes
    pub cluster_to_nodes: Vec<Vec<WebNodeId>>,
    /// Tag ID → nodes
    pub tag_to_nodes: Vec<Vec<WebNodeId>>,
    /// Door ID → nodes
    pub door_to_nodes: Vec<Vec<WebNodeId>>,
}

/// Revolving door for KV routing.
/// Entry/exit nodes + flow vector + bounding box.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvRevolvingDoor {
    pub door_id: usize,
    pub entry_nodes: Vec<WebNodeId>,
    pub exit_nodes: Vec<WebNodeId>,

    /// Cached centroid for routing bias (entry + exit)
    pub centroid: (f32, f32),

    /// Cached flow vector (exit centroid − entry centroid)
    pub flow_vec: (f32, f32),

    /// Cached bounding box for fast rejection
    pub bbox_min: (f32, f32),
    pub bbox_max: (f32, f32),
}

/// Fusion field over the graph.
/// Combines load, semantic, geometry, and door flow into a single routing bias.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvFusionField {
    /// Per-node fused bias (load + semantic + geometry + door flow)
    pub fused_bias: Vec<f32>,
}

/// Routing decision packet (GPU-ready).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvRoutingDecision {
    pub root: WebNodeId,
    pub chosen_zone_id: usize,
    pub chosen_node: WebNodeId,
    pub fused_bias: f32,
}

/// Build a simple cross-link grid from region zoning.
/// In a full system, this would be fed by cluster/semantic modules.
pub fn build_cross_link_grid_from_regions(
    regions: &[RegionZoning],
) -> KvCrossLinkGrid {
    let mut cluster_to_nodes = Vec::new();
    let mut tag_to_nodes = Vec::new();
    let mut door_to_nodes = Vec::new();

    // For now, treat each region as a "cluster",
    // each zone as a "tag", and each root as a "door anchor".
    for (_cluster_id, region) in regions.iter().enumerate() {
        // Cluster → all nodes in region
        cluster_to_nodes.push(region.nodes.clone());

        // Tags → each zone slice
        for zone in &region.zones {
            let slice = &region.index_map[zone.start..zone.end];
            tag_to_nodes.push(slice.to_vec());
        }

        // Doors → root + high-bias nodes (top of index_map)
        let mut door_nodes = Vec::new();
        door_nodes.push(region.root);
        let top_k = region.index_map.iter().take(8).cloned().collect::<Vec<_>>();
        door_nodes.extend(top_k);
        door_to_nodes.push(door_nodes);
    }

    KvCrossLinkGrid {
        cluster_to_nodes,
        tag_to_nodes,
        door_to_nodes,
    }
}

/// Compute centroid of a set of nodes in polygon space.
fn compute_polygon_centroid(web: &KvWeb, nodes: &[WebNodeId]) -> (f32, f32) {
    if nodes.is_empty() {
        return (0.0, 0.0);
    }

    let mut sx = 0.0f32;
    let mut sy = 0.0f32;
    let mut count = 0.0f32;

    for id in nodes {
        if let Some(node) = web.nodes.get(id) {
            if let Some(poly) = &node.polygon {
                if poly.centroid.len() >= 2 {
                    sx += poly.centroid[0];
                    sy += poly.centroid[1];
                    count += 1.0;
                }
            }
        }
    }

    if count == 0.0 {
        (0.0, 0.0)
    } else {
        (sx / count, sy / count)
    }
}

/// Compute bounding box in polygon centroid space.
fn compute_polygon_bbox(web: &KvWeb, nodes: &[WebNodeId]) -> ((f32, f32), (f32, f32)) {
    if nodes.is_empty() {
        return ((0.0, 0.0), (0.0, 0.0));
    }

    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;

    for id in nodes {
        if let Some(node) = web.nodes.get(id) {
            if let Some(poly) = &node.polygon {
                if poly.centroid.len() >= 2 {
                    let x = poly.centroid[0];
                    let y = poly.centroid[1];

                    if x < min_x { min_x = x; }
                    if y < min_y { min_y = y; }
                    if x > max_x { max_x = x; }
                    if y > max_y { max_y = y; }
                }
            }
        }
    }

    ((min_x, min_y), (max_x, max_y))
}

/// Build revolving doors from cross-link grid.
/// Each door uses entry_nodes and exit_nodes from door_to_nodes slices.
pub fn build_revolving_doors(
    web: &KvWeb,
    grid: &KvCrossLinkGrid,
) -> Vec<KvRevolvingDoor> {
    let mut doors = Vec::new();

    for (door_id, nodes) in grid.door_to_nodes.iter().enumerate() {
        if nodes.is_empty() {
            continue;
        }

        // Simple split: first half = entry, second half = exit.
        let mid = nodes.len() / 2;
        let entry_nodes = nodes[..mid.max(1)].to_vec();
        let exit_nodes = nodes[mid.max(1)..].to_vec();

        let entry_centroid = compute_polygon_centroid(web, &entry_nodes);
        let exit_centroid = compute_polygon_centroid(web, &exit_nodes);

        let flow_vec = (exit_centroid.0 - entry_centroid.0,
                        exit_centroid.1 - entry_centroid.1);

        let (bbox_min, bbox_max) = compute_polygon_bbox(web, nodes);

        doors.push(KvRevolvingDoor {
            door_id,
            entry_nodes,
            exit_nodes,
            centroid: (
                (entry_centroid.0 + exit_centroid.0) * 0.5,
                (entry_centroid.1 + exit_centroid.1) * 0.5,
            ),
            flow_vec,
            bbox_min,
            bbox_max,
        });
    }

    doors
}

/// Build fusion field from PageRank + polygon bias + door flow.
pub fn build_fusion_field(
    web: &KvWeb,
    pr: &[(WebNodeId, f32)],
    doors: &[KvRevolvingDoor],
) -> KvFusionField {
    let mut fused_bias = Vec::with_capacity(pr.len());

    for (id, score) in pr {
        // Base: PageRank score
        let mut bias = *score;

        // Semantic geometry: polygon bias vs top-ranked node
        let top_node = pr.first().map(|(tid, _)| *tid);
        if let Some(top) = top_node {
            bias *= polygon_neighbor_bias(web, top, *id);
        }

        // Door flow contribution: if node lies inside any door bbox, boost by flow magnitude
        for door in doors {
            if let Some(node) = web.nodes.get(id) {
                if let Some(poly) = &node.polygon {
                    if poly.centroid.len() >= 2 {
                        let x = poly.centroid[0];
                        let y = poly.centroid[1];

                        if x >= door.bbox_min.0 && x <= door.bbox_max.0 &&
                           y >= door.bbox_min.1 && y <= door.bbox_max.1 {
                            let flow_mag = (door.flow_vec.0 * door.flow_vec.0
                                + door.flow_vec.1 * door.flow_vec.1).sqrt();
                            bias *= 1.0 + (flow_mag * 0.05);
                        }
                    }
                }
            }
        }

        fused_bias.push(bias);
    }

    KvFusionField { fused_bias }
}

/// Routing solver: choose a node inside a region using fusion field + zoning.
/// Returns GPU-ready routing decision packets.
pub fn solve_routing_with_fusion(
    web: &KvWeb,
    regions: &[RegionZoning],
    fusion: &KvFusionField,
    pr: &[(WebNodeId, f32)],
) -> Vec<KvRoutingDecision> {
    use std::collections::HashMap;

    // Map node → fused bias for quick lookup
    let mut bias_map = HashMap::new();
    for ((id, _score), fb) in pr.iter().zip(fusion.fused_bias.iter()) {
        bias_map.insert(*id, *fb);
    }

    let mut decisions = Vec::new();

    for region in regions {
        // Choose zone with highest average fused bias
        let mut best_zone_id = 0usize;
        let mut best_zone_bias = f32::MIN;

        for zone in &region.zones {
            let slice = &region.index_map[zone.start..zone.end];
            if slice.is_empty() {
                continue;
            }

            let mut sum = 0.0f32;
            let mut count = 0.0f32;

            for id in slice {
                if let Some(b) = bias_map.get(id) {
                    sum += *b;
                    count += 1.0;
                }
            }

            if count > 0.0 {
                let avg = sum / count;
                if avg > best_zone_bias {
                    best_zone_bias = avg;
                    best_zone_id = zone.zone_id;
                }
            }
        }

        // Inside chosen zone, pick node with max fused bias
        let chosen_zone = region.zones.iter().find(|z| z.zone_id == best_zone_id);
        if let Some(zone) = chosen_zone {
            let slice = &region.index_map[zone.start..zone.end];
            let mut best_node = region.root;
            let mut best_bias = f32::MIN;

            for id in slice {
                if let Some(b) = bias_map.get(id) {
                    if *b > best_bias {
                        best_bias = *b;
                        best_node = *id;
                    }
                }
            }

            decisions.push(KvRoutingDecision {
                root: region.root,
                chosen_zone_id: best_zone_id,
                chosen_node: best_node,
                fused_bias: best_bias,
            });
        }
    }

    decisions
}

/// Compressed routing decisions (GPU-ready).
pub fn solve_routing_with_fusion_compressed(
    web: &KvWeb,
    regions: &[RegionZoning],
    pr_iterations: usize,
    pr_damping: f32,
) -> Option<Vec<u8>> {
    let pr = pagerank(web, pr_iterations, pr_damping);
    let doors_grid = build_cross_link_grid_from_regions(regions);
    let doors = build_revolving_doors(web, &doors_grid);
    let fusion = build_fusion_field(web, &pr, &doors);
    let decisions = solve_routing_with_fusion(web, regions, &fusion, &pr);

    web.compressor.as_ref().map(|c| {
        c.compress(&(
            "solve_routing_with_fusion",
            pr_iterations,
            pr_damping,
            &decisions,
        ))
    })
}

//
// ──────────────────────────────────────────────────────────────
//   EMBEDDED ROUNDABOUT LOGIC (HEATMAPS + PREDICTOR + SMOOTHING + MEMORY + SOLVER)
// ──────────────────────────────────────────────────────────────
//

/// Roundabout heatmaps: dual-layer view over regions.
/// Layer A = structural (BFS / PageRank)
/// Layer B = semantic / geometric (polygon + fusion bias)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundaboutHeatmaps {
    pub layer_a: Vec<f32>,
    pub layer_b: Vec<f32>,
}

/// Roundabout predictor configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundaboutPredictorConfig {
    pub passes: usize,
    pub min_bias: f32,
    pub max_bias: f32,
    pub smoothing_strength: f32,
}

/// Roundabout chain element (a routed path through nodes).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundaboutChain {
    pub root: WebNodeId,
    pub nodes: Vec<WebNodeId>,
    pub total_bias: f32,
}

/// Roundabout pattern memory entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundaboutPattern {
    pub root: WebNodeId,
    pub chain: RoundaboutChain,
    pub weight: f32,
}

/// Roundabout pattern memory with decay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundaboutPatternMemory {
    pub patterns: Vec<RoundaboutPattern>,
    pub decay: f32,
}

/// Roundabout solver result (final corrected / chosen node).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundaboutSolverResult {
    pub root: WebNodeId,
    pub chosen_node: WebNodeId,
    pub bias: f32,
}

/// Build roundabout heatmaps from regions + fusion field.
pub fn build_roundabout_heatmaps(
    regions: &[RegionZoning],
    fusion: &KvFusionField,
    pr: &[(WebNodeId, f32)],
) -> RoundaboutHeatmaps {
    use std::collections::HashMap;

    let mut bias_map = HashMap::new();
    for ((id, _score), fb) in pr.iter().zip(fusion.fused_bias.iter()) {
        bias_map.insert(*id, *fb);
    }

    let mut layer_a = Vec::new();
    let mut layer_b = Vec::new();

    for region in regions {
        // Structural: BFS depth encoded via scratch.layer_a
        for v in &region.scratch.layer_a {
            layer_a.push(*v);
        }

        // Semantic/geometric: fused bias per node
        for id in &region.nodes {
            let b = bias_map.get(id).copied().unwrap_or(0.0);
            layer_b.push(b);
        }
    }

    RoundaboutHeatmaps { layer_a, layer_b }
}

/// Run roundabout predictor: multi-pass selection of chains using fusion + heatmaps.
pub fn run_roundabout_predictor(
    web: &KvWeb,
    regions: &[RegionZoning],
    fusion: &KvFusionField,
    pr: &[(WebNodeId, f32)],
    cfg: &RoundaboutPredictorConfig,
) -> Vec<RoundaboutChain> {
    use std::collections::HashMap;

    let mut bias_map = HashMap::new();
    for ((id, _score), fb) in pr.iter().zip(fusion.fused_bias.iter()) {
        bias_map.insert(*id, *fb);
    }

    let mut chains = Vec::new();

    for region in regions {
        // Start from root, walk through highest-bias nodes in index_map.
        let mut current = region.root;
        let mut chain_nodes = Vec::new();
        let mut total_bias = 0.0f32;

        chain_nodes.push(current);

        for _pass in 0..cfg.passes {
            // Choose next node in region with highest bias that is not yet in chain.
            let mut best_node = None;
            let mut best_bias = cfg.min_bias;

            for id in &region.index_map {
                if chain_nodes.contains(id) {
                    continue;
                }
                if let Some(b) = bias_map.get(id) {
                    if *b > best_bias && *b <= cfg.max_bias {
                        best_bias = *b;
                        best_node = Some(*id);
                    }
                }
            }

            if let Some(next) = best_node {
                chain_nodes.push(next);
                total_bias += best_bias;
                current = next;
            } else {
                break;
            }
        }

        chains.push(RoundaboutChain {
            root: region.root,
            nodes: chain_nodes,
            total_bias,
        });
    }

    chains
}

/// Smooth roundabout chains: reduce sharp jumps in bias by local averaging.
pub fn smooth_roundabout_chains(
    chains: &mut [RoundaboutChain],
    fusion: &KvFusionField,
    pr: &[(WebNodeId, f32)],
    strength: f32,
) {
    use std::collections::HashMap;

    let mut bias_map = HashMap::new();
    for ((id, _score), fb) in pr.iter().zip(fusion.fused_bias.iter()) {
        bias_map.insert(*id, *fb);
    }

    for chain in chains {
        if chain.nodes.len() < 3 {
            continue;
        }

        let mut new_total = 0.0f32;

        for (i, id) in chain.nodes.iter().enumerate() {
            let mut local_sum = 0.0f32;
            let mut local_count = 0.0f32;

            for j in i.saturating_sub(1)..=(i + 1).min(chain.nodes.len() - 1) {
                let nid = chain.nodes[j];
                if let Some(b) = bias_map.get(&nid) {
                    local_sum += *b;
                    local_count += 1.0;
                }
            }

            if local_count > 0.0 {
                let avg = local_sum / local_count;
                new_total += avg * strength + bias_map.get(id).copied().unwrap_or(0.0) * (1.0 - strength);
            }
        }

        chain.total_bias = new_total;
    }
}

/// Update pattern memory with new chains, applying decay.
pub fn update_roundabout_pattern_memory(
    memory: &mut RoundaboutPatternMemory,
    chains: &[RoundaboutChain],
) {
    // Apply decay to existing patterns
    for pattern in &mut memory.patterns {
        pattern.weight *= memory.decay;
    }

    // Add new chains as fresh patterns
    for chain in chains {
        memory.patterns.push(RoundaboutPattern {
            root: chain.root,
            chain: chain.clone(),
            weight: 1.0,
        });
    }

    // Optional: prune very low-weight patterns
    memory.patterns.retain(|p| p.weight > 0.01);
}

/// Roundabout bias: adjust fused bias using pattern memory.
pub fn apply_roundabout_bias(
    fusion: &mut KvFusionField,
    pr: &[(WebNodeId, f32)],
    memory: &RoundaboutPatternMemory,
) {
    use std::collections::HashMap;

    let mut bias_map = HashMap::new();
    for ((id, _score), fb) in pr.iter().zip(fusion.fused_bias.iter()) {
        bias_map.insert(*id, *fb);
    }

    // For each pattern, boost bias along its chain.
    for pattern in &memory.patterns {
        let boost = pattern.weight * 0.05;
        for id in &pattern.chain.nodes {
            if let Some(b) = bias_map.get_mut(id) {
                *b *= 1.0 + boost;
            }
        }
    }

    // Write back to fusion field
    fusion.fused_bias.clear();
    for (id, _score) in pr {
        let b = bias_map.get(id).copied().unwrap_or(0.0);
        fusion.fused_bias.push(b);
    }
}

/// Hybrid roundabout solver: final selection of node per root using
/// fusion field, chains, and pattern memory.
pub fn run_roundabout_solver(
    regions: &[RegionZoning],
    fusion: &KvFusionField,
    pr: &[(WebNodeId, f32)],
    chains: &[RoundaboutChain],
    memory: &RoundaboutPatternMemory,
) -> Vec<RoundaboutSolverResult> {
    use std::collections::HashMap;

    let mut bias_map = HashMap::new();
    for ((id, _score), fb) in pr.iter().zip(fusion.fused_bias.iter()) {
        bias_map.insert(*id, *fb);
    }

    // Build quick lookup: root → best chain
    let mut best_chain_for_root: HashMap<WebNodeId, &RoundaboutChain> = HashMap::new();
    for chain in chains {
        best_chain_for_root
            .entry(chain.root)
            .and_modify(|existing| {
                if chain.total_bias > existing.total_bias {
                    *existing = chain;
                }
            })
            .or_insert(chain);
    }

    let mut results = Vec::new();

    for region in regions {
        // Prefer chain-based choice if available
        if let Some(chain) = best_chain_for_root.get(&region.root) {
            // Choose last node in chain as final routing target
            if let Some(&last) = chain.nodes.last() {
                let b = bias_map.get(&last).copied().unwrap_or(0.0);
                results.push(RoundaboutSolverResult {
                    root: region.root,
                    chosen_node: last,
                    bias: b,
                });
                continue;
            }
        }

        // Fallback: use fusion + zoning (same logic as solve_routing_with_fusion)
        let mut best_node = region.root;
        let mut best_bias = f32::MIN;

        for id in &region.index_map {
            if let Some(b) = bias_map.get(id) {
                if *b > best_bias {
                    best_bias = *b;
                    best_node = *id;
                }
            }
        }

        results.push(RoundaboutSolverResult {
            root: region.root,
            chosen_node: best_node,
            bias: best_bias,
        });
    }

    results
}

/// Compressed roundabout pipeline: heatmaps + predictor + smoothing + memory + solver.
pub fn roundabout_pipeline_compressed(
    web: &KvWeb,
    regions: &[RegionZoning],
    pr_iterations: usize,
    pr_damping: f32,
    predictor_cfg: &RoundaboutPredictorConfig,
    memory: &mut RoundaboutPatternMemory,
) -> Option<Vec<u8>> {
    // Base PageRank + doors + fusion
    let pr = pagerank(web, pr_iterations, pr_damping);
    let grid = build_cross_link_grid_from_regions(regions);
    let doors = build_revolving_doors(web, &grid);
    let mut fusion = build_fusion_field(web, &pr, &doors);

    // Heatmaps
    let heatmaps = build_roundabout_heatmaps(regions, &fusion, &pr);

    // Predictor
    let mut chains = run_roundabout_predictor(web, regions, &fusion, &pr, predictor_cfg);

    // Smoothing
    smooth_roundabout_chains(&mut chains, &fusion, &pr, predictor_cfg.smoothing_strength);

    // Memory update + bias
    update_roundabout_pattern_memory(memory, &chains);
    apply_roundabout_bias(&mut fusion, &pr, memory);

    // Solver
    let results = run_roundabout_solver(regions, &fusion, &pr, &chains, memory);

    web.compressor.as_ref().map(|c| {
        c.compress(&(
            "roundabout_pipeline",
            pr_iterations,
            pr_damping,
            predictor_cfg.passes,
            &heatmaps.layer_a,
            &heatmaps.layer_b,
            &chains,
            &memory.patterns,
            &results,
        ))
    })
}

