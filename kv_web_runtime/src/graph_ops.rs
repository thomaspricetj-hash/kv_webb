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

