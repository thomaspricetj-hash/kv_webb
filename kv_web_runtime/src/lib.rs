//! kv_web_runtime
//! Runtime logic for KV‑cache webbing.
//!
//! This crate provides the operational layer on top of kv_web_core:
//! - neighbor traversal
//! - depth‑limited region queries
//! - scoring utilities
//! - pruning and merging helpers
//! - semantic clustering
//! - embedding‑based similarity
//! - graph algorithms (BFS, PageRank)
//! - heatmap generation
//!
//! The transformer still sees a flat KV cache, but this runtime lets
//! you query *structured regions* of that cache using the web graph.

// Core runtime modules
pub mod semantic;
pub mod dynamic_web;
pub mod pruning;
pub mod drift;

// Newly wired modules (4–8)
pub mod cluster;
pub mod embedding;
pub mod graph_ops;
pub mod heatmap;

use kv_web_core::{KvWeb, WebNodeId, TokenId, WebNode};
use std::collections::{HashSet, VecDeque};

/// Runtime extensions for KvWeb.
/// This is intentionally separate from the core crate so the core stays pure.
pub trait KvWebRuntime {
    /// Get direct neighbor nodes (outgoing edges only).
    fn neighbors(&self, node: WebNodeId) -> Vec<&WebNode>;

    /// Depth‑limited region query.
    /// Returns all tokens reachable from `root` within `depth` hops.
    fn tokens_in_region(&self, root: WebNodeId, depth: usize) -> HashSet<TokenId>;

    /// Get all nodes reachable within `depth` hops.
    fn nodes_in_region(&self, root: WebNodeId, depth: usize) -> HashSet<WebNodeId>;

    /// Compute a simple region score (sum of node scores).
    fn region_score(&self, root: WebNodeId, depth: usize) -> f32;

    /// Prune nodes below a score threshold.
    fn prune_low_score(&mut self, min_score: f32);

    /// Merge two nodes into one (simple union).
    fn merge_nodes(&mut self, a: WebNodeId, b: WebNodeId) -> WebNodeId;
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
                    queue.push_back((edge.to, d + 1));
                }
            }
        }

        out_tokens
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
                    queue.push_back((edge.to, d + 1));
                }
            }
        }

        visited
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
        // Remove nodes below threshold
        self.nodes.retain(|_, node| node.score >= min_score);

        // Remove edges pointing to deleted nodes
        self.edges.retain(|edge| self.nodes.contains_key(&edge.to));

        // Rebuild token index
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

        if let Some(na) = self.nodes.get(&a) {
            tokens.extend(na.tokens.clone());
            score += na.score;
            label = na.label.clone();
        }

        if let Some(nb) = self.nodes.get(&b) {
            tokens.extend(nb.tokens.clone());
            score += nb.score;
            if label.is_none() {
                label = nb.label.clone();
            }
        }

        // Remove old nodes
        self.nodes.remove(&a);
        self.nodes.remove(&b);

        // Insert merged node
        let merged = WebNode {
            id: new_id,
            tokens: tokens.clone(),
            label,
            score,
        };

        self.nodes.insert(new_id, merged);

        // Update token index
        for t in tokens {
            self.node_index_by_token.insert(t, new_id);
        }

        // Remove edges from old nodes
        self.edges.retain(|e| e.from != a && e.from != b);

        new_id
    }
}

