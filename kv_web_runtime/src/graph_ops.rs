//! graph_ops.rs
//!
//! General graph operations over KvWeb + BitDrop_v2 max‑tier compression
//! + Polygonal-KV geometry upgrade.
//!
//! Adds:
//! - polygon-aware BFS expansion
//! - polygon-biased PageRank
//! - face-index semantic weighting
//!
//! All upgrades are backwards-compatible.

use kv_web_core::{KvWeb, WebNodeId, KvWebCompressor};
use crate::KvWebRuntime;

/// Graph ops optimization configuration.
#[derive(Debug, Clone)]
pub struct GraphOpsOptimizationConfig {
    pub min_damping: f32,
    pub max_damping: f32,
    pub target_bfs_size: usize,
    pub max_bfs_size: usize,
    pub min_depth: usize,
    pub max_depth: usize,
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

/// Compressed BFS region expansion.
pub fn bfs_region_compressed(web: &KvWeb, root: WebNodeId, depth: usize) -> Option<Vec<u8>> {
    let nodes = bfs_region(web, root, depth);
    web.compressor.as_ref().map(|c| {
        c.compress(&(
            "bfs_region",
            root,
            depth,
            &nodes
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

/// Compressed PageRank.
pub fn pagerank_compressed(web: &KvWeb, iterations: usize, damping: f32) -> Option<Vec<u8>> {
    let pr = pagerank(web, iterations, damping);
    web.compressor.as_ref().map(|c| {
        c.compress(&(
            "pagerank",
            iterations,
            damping,
            &pr
        ))
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

    web.compressor.as_ref().map(|c| {
        c.compress(&(
            "optimize_graph_ops",
            root,
            bfs_size,
            pr_span,
            *depth,
            *damping,
        ))
    })
}
