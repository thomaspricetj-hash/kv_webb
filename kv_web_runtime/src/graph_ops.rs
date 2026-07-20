//! graph_ops.rs
//!
//! General graph operations over KvWeb:
//! - BFS region expansion
//! - simple PageRank-like scoring
//! - neighbor collection utilities

use kv_web_core::{KvWeb, WebNodeId};
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

        // neighbors() comes from KvWebRuntime — now works
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
            // NOTE: this is a simplified PageRank — no out-degree normalization
            *new_rank.entry(edge.to).or_insert(0.0) += damping * out_rank;
        }

        rank = new_rank;
    }

    let mut result: Vec<(WebNodeId, f32)> = rank.into_iter().collect();
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    result
}
