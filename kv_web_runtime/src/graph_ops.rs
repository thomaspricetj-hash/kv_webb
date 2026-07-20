//! graph_ops.rs
//!
//! General graph operations over KvWeb + BitDrop_v2 max‑tier compression:
//! - BFS region expansion
//! - PageRank-like scoring
//! - neighbor collection utilities
//!
//! All graph operations now produce reversible compressed packets.

use kv_web_core::{KvWeb, WebNodeId, KvWebCompressor};
use crate::KvWebRuntime;   // ★ REQUIRED so web.neighbors() works

/// BFS region expansion from a root node.
/// Returns all nodes reachable within `depth` hops.
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
            if !visited.contains(&nid) {
                visited.insert(nid);
                queue.push_back((nid, d + 1));
            }
        }
    }

    result
}

/// Compressed BFS region expansion.
/// Returns a BitDrop_v2 compressed packet containing the BFS result.
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

/// Simple PageRank-like scoring over nodes.
/// This is intentionally lightweight — good enough for relevance ranking.
pub fn pagerank(web: &KvWeb, iterations: usize, damping: f32) -> Vec<(WebNodeId, f32)> {
    use std::collections::HashMap;

    let n = web.nodes.len();
    if n == 0 {
        return Vec::new();
    }

    let ids: Vec<WebNodeId> = web.nodes.keys().cloned().collect();

    // initialize rank uniformly
    let mut rank: HashMap<WebNodeId, f32> =
        ids.iter().map(|id| (*id, 1.0 / n as f32)).collect();

    for _ in 0..iterations {
        let mut new_rank: HashMap<WebNodeId, f32> =
            ids.iter().map(|id| (*id, (1.0 - damping) / n as f32)).collect();

        // distribute rank along outgoing edges
        for edge in &web.edges {
            let out_rank = rank[&edge.from];
            *new_rank.entry(edge.to).or_insert(0.0) += damping * out_rank;
        }

        rank = new_rank;
    }

    let mut result: Vec<(WebNodeId, f32)> = rank.into_iter().collect();
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    result
}

/// Compressed PageRank.
/// Returns a BitDrop_v2 compressed packet containing the PageRank vector.
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
