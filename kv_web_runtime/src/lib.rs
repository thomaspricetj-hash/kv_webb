//! kv_web_runtime
//! Runtime logic for KV‑cache webbing + BitDrop_v2 max‑tier compression
//! + Polygonal‑KV geometry.

pub mod semantic;
pub mod dynamic_web;
pub mod pruning;
pub mod drift;

// Newly wired modules (4–8)
pub mod cluster;
pub mod embedding;
pub mod graph_ops;
pub mod heatmap;

use kv_web_core::{KvWeb, WebNodeId, TokenId, WebNode, KvWebCompressor};
use std::collections::{HashSet, VecDeque};

/// Runtime extensions for KvWeb.
/// This is intentionally separate from the core crate so the core stays pure.
pub trait KvWebRuntime {
    /// Get direct neighbor nodes (outgoing edges only).
    fn neighbors(&self, node: WebNodeId) -> Vec<&WebNode>;

    /// Depth‑limited region query.
    /// Returns all tokens reachable from `root` within `depth` hops.
    fn tokens_in_region(&self, root: WebNodeId, depth: usize) -> HashSet<TokenId>;

    /// Same as above, but returns a compressed BitDrop snapshot.
    fn tokens_in_region_compressed(&self, root: WebNodeId, depth: usize) -> Option<Vec<u8>>;

    /// Get all nodes reachable within `depth` hops.
    fn nodes_in_region(&self, root: WebNodeId, depth: usize) -> HashSet<WebNodeId>;

    /// Compressed region snapshot.
    fn nodes_in_region_compressed(&self, root: WebNodeId, depth: usize) -> Option<Vec<u8>>;

    /// Compute a simple region score (sum of node scores).
    fn region_score(&self, root: WebNodeId, depth: usize) -> f32;

    /// Prune nodes below a score threshold.
    fn prune_low_score(&mut self, min_score: f32);

    /// Merge two nodes into one (simple union).
    fn merge_nodes(&mut self, a: WebNodeId, b: WebNodeId) -> WebNodeId;

    /// Merge nodes + compressed metadata.
    fn merge_nodes_compressed(&mut self, a: WebNodeId, b: WebNodeId) -> Option<Vec<u8>>;
}

/// Polygon‑aware neighbor bias (same‑face, centroid‑close).
fn polygon_neighbor_bias(web: &KvWeb, a: WebNodeId, b: WebNodeId) -> f32 {
    let na = match web.nodes.get(&a) {
        Some(n) => n,
        None => return 1.0,
    };
    let nb = match web.nodes.get(&b) {
        Some(n) => n,
        None => return 1.0,
    };

    let pa = match &na.polygon {
        Some(p) => p,
        None => return 1.0,
    };
    let pb = match &nb.polygon {
        Some(p) => p,
        None => return 1.0,
    };

    let face_bonus = if pa.face_index == pb.face_index { 1.25 } else { 1.0 };

    let mut dist = 0.0;
    for (ca, cb) in pa.centroid.iter().zip(pb.centroid.iter()) {
        dist += (ca - cb).abs();
    }
    let radius = pa.radius + pb.radius + 1.0;
    let penalty = (dist / radius).min(0.5);

    face_bonus - penalty
}

impl KvWebRuntime for KvWeb {
    fn neighbors(&self, node: WebNodeId) -> Vec<&WebNode> {
        let mut out = Vec::new();
        for edge in &self.edges {
            if edge.from == node {
                if let Some(n) = self.nodes.get(&edge.to) {
                    out.push(n);
                }
            }
        }
        out
    }

    fn tokens_in_region(&self, root: WebNodeId, depth: usize) -> HashSet<TokenId> {
        let mut visited_nodes: HashSet<WebNodeId> = HashSet::new();
        let mut out_tokens: HashSet<TokenId> = HashSet::new();

        let mut queue: VecDeque<(WebNodeId, usize)> = VecDeque::new();
        queue.push_back((root, 0));

        while let Some((current, d)) = queue.pop_front() {
            if d > depth || visited_nodes.contains(&current) {
                continue;
            }
            visited_nodes.insert(current);

            if let Some(node) = self.nodes.get(&current) {
                for t in &node.tokens {
                    out_tokens.insert(*t);
                }
            }

            for edge in &self.edges {
                if edge.from == current {
                    let bias = polygon_neighbor_bias(self, current, edge.to);
                    if bias <= 0.0 {
                        continue;
                    }
                    queue.push_back((edge.to, d + 1));
                }
            }
        }

        out_tokens
    }

    fn tokens_in_region_compressed(&self, root: WebNodeId, depth: usize) -> Option<Vec<u8>> {
        let tokens = self.tokens_in_region(root, depth);
        self.compressor.as_ref().map(|c| c.compress(&tokens))
    }

    fn nodes_in_region(&self, root: WebNodeId, depth: usize) -> HashSet<WebNodeId> {
        let mut visited: HashSet<WebNodeId> = HashSet::new();
        let mut queue: VecDeque<(WebNodeId, usize)> = VecDeque::new();
        queue.push_back((root, 0));

        while let Some((current, d)) = queue.pop_front() {
            if d > depth || visited.contains(&current) {
                continue;
            }
            visited.insert(current);

            for edge in &self.edges {
                if edge.from == current {
                    let bias = polygon_neighbor_bias(self, current, edge.to);
                    if bias <= 0.0 {
                        continue;
                    }
                    queue.push_back((edge.to, d + 1));
                }
            }
        }

        visited
    }

    fn nodes_in_region_compressed(&self, root: WebNodeId, depth: usize) -> Option<Vec<u8>> {
        let nodes = self.nodes_in_region(root, depth);
        self.compressor.as_ref().map(|c| c.compress(&nodes))
    }

    fn region_score(&self, root: WebNodeId, depth: usize) -> f32 {
        let nodes = self.nodes_in_region(root, depth);
        nodes
            .iter()
            .filter_map(|id| self.nodes.get(id))
            .map(|n| n.score)
            .sum()
    }

    fn prune_low_score(&mut self, min_score: f32) {
        self.nodes.retain(|_, node| node.score >= min_score);
        self.edges.retain(|edge| self.nodes.contains_key(&edge.to));

        self.node_index_by_token.clear();
        for (id, node) in &self.nodes {
            for t in &node.tokens {
                self.node_index_by_token.insert(*t, *id);
            }
        }
    }

    fn merge_nodes(&mut self, a: WebNodeId, b: WebNodeId) -> WebNodeId {
        let new_id = WebNodeId(self.nodes.len());

        let mut tokens = Vec::new();
        let mut score = 0.0;
        let mut label = None;
        let mut polygon = None;

        if let Some(na) = self.nodes.get(&a) {
            tokens.extend(na.tokens.clone());
            score += na.score;
            label = na.label.clone();
            polygon = na.polygon.clone();
        }

        if let Some(nb) = self.nodes.get(&b) {
            tokens.extend(nb.tokens.clone());
            score += nb.score;
            if label.is_none() {
                label = nb.label.clone();
            }

            // merge polygon metadata if both have it
            if let (Some(pa), Some(pb)) = (polygon.clone(), nb.polygon.clone()) {
                let mut centroid = pa.centroid.clone();
                if centroid.len() == pb.centroid.len() {
                    for (c, v) in centroid.iter_mut().zip(pb.centroid.iter()) {
                        *c = (*c + *v) / 2.0;
                    }
                }
                let radius = (pa.radius + pb.radius) / 2.0;
                let face_index = if pa.face_index == pb.face_index {
                    pa.face_index
                } else {
                    pa.face_index.max(pb.face_index)
                };

                polygon = Some(kv_web_core::PolygonRegion {
                    id: pa.id,
                    centroid,
                    radius,
                    face_index,
                });
            }
        }

        self.nodes.remove(&a);
        self.nodes.remove(&b);

        let merged = WebNode {
            id: new_id,
            tokens: tokens.clone(),
            tokens_compressed: None,
            label,
            label_compressed: None,
            score,
            branch_id: None,
            branch_kind: None,
            branch_stability: 0.0,
            branch_drift: 0.0,
            branch_meta_compressed: None,
            polygon,
        };

        self.nodes.insert(new_id, merged);

        for t in tokens {
            self.node_index_by_token.insert(t, new_id);
        }

        self.edges.retain(|e| e.from != a && e.from != b);

        new_id
    }

    fn merge_nodes_compressed(&mut self, a: WebNodeId, b: WebNodeId) -> Option<Vec<u8>> {
        let new_id = self.merge_nodes(a, b);
        let merged = self.nodes.get(&new_id)?;

        self.compressor.as_ref().map(|c| {
            c.compress(&(
                merged.id,
                merged.tokens.clone(),
                merged.label.clone(),
                merged.score,
                merged.branch_id,
                merged.branch_kind,
                merged.branch_stability,
                merged.branch_drift,
                merged.polygon.as_ref()
            ))
        })
    }
}


