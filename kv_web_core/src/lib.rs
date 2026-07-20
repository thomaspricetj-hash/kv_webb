//! kv_web_core
//! Core data structures for KV‑cache webbing.

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::time::Instant;

/// A single token position in the KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TokenId(pub usize);

/// Unique identifier for a node in the KV web.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WebNodeId(pub usize);

/// A semantic or structural node over a span of tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebNode {
    pub id: WebNodeId,
    pub tokens: Vec<TokenId>,
    pub label: Option<String>,
    pub score: f32,
}

/// Types of edges between nodes.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EdgeKind {
    Semantic,
    Positional,
    ConversationTurn,
    Custom(u8),
}

/// A directed edge between two nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebEdge {
    pub from: WebNodeId,
    pub to: WebNodeId,
    pub weight: f32,
    pub kind: EdgeKind,
}

/// Drift state for each node (moved here to avoid crate cycles).
#[derive(Debug, Clone)]
pub struct NodeDriftState {
    pub last_access: Instant,
}

impl NodeDriftState {
    pub fn new() -> Self {
        Self {
            last_access: Instant::now(),
        }
    }
}

/// The KV‑cache web structure.
#[derive(Debug, Serialize, Deserialize)]
pub struct KvWeb {
    pub nodes: HashMap<WebNodeId, WebNode>,
    pub edges: Vec<WebEdge>,
    pub node_index_by_token: HashMap<TokenId, WebNodeId>,
    next_node_id: usize,

    #[serde(skip)]
    pub drift_state: Option<HashMap<WebNodeId, NodeDriftState>>,
}

impl Default for KvWeb {
    fn default() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            node_index_by_token: HashMap::new(),
            next_node_id: 0,
            drift_state: None,
        }
    }
}

impl KvWeb {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn create_node<S: Into<String>>(
        &mut self,
        tokens: Vec<TokenId>,
        label: Option<S>,
        score: f32,
    ) -> WebNodeId {
        let id = WebNodeId(self.next_node_id);
        self.next_node_id += 1;

        let node = WebNode {
            id,
            tokens: tokens.clone(),
            label: label.map(Into::into),
            score,
        };

        for t in &tokens {
            self.node_index_by_token.insert(*t, id);
        }

        self.nodes.insert(id, node);
        id
    }

    pub fn add_edge(
        &mut self,
        from: WebNodeId,
        to: WebNodeId,
        weight: f32,
        kind: EdgeKind,
    ) {
        self.edges.push(WebEdge { from, to, weight, kind });
    }

    pub fn node_for_token(&self, token: TokenId) -> Option<&WebNode> {
        self.node_index_by_token
            .get(&token)
            .and_then(|id| self.nodes.get(id))
    }
}
