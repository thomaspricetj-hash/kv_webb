//! kv_web_core
//! Core data structures for KV‑cache webbing + diverging‑memory metadata + BitDrop_v2 compression
//! + Polygonal‑KV geometry.

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::time::Instant;

// ============================================================
// BitDrop_v2 COMPRESSION ADAPTER (FUNCTION-BASED)
// ============================================================

#[derive(Debug, Clone)]
/// Thin wrapper so KvWeb never touches BitDrop internals directly.
pub struct KvWebCompressor;

impl KvWebCompressor {
    pub fn new() -> Self {
        Self
    }

    /// Compress any serializable payload into a reversible BitDrop block.
    pub fn compress<T: Serialize>(&self, payload: &T) -> Vec<u8> {
        let bytes = bincode::serialize(payload).unwrap_or_default();
        bitdrop_v2::compress_adaptive(&bytes)
    }

    /// Decompress a BitDrop block back into a Rust type.
    pub fn decompress<T: for<'de> Deserialize<'de>>(&self, data: &[u8]) -> Option<T> {
        let raw = bitdrop_v2::decompress_adaptive(data);
        bincode::deserialize(&raw).ok()
    }
}

// ============================================================
// Core KV‑Web Types
// ============================================================

/// A single token position in the KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TokenId(pub usize);

/// Unique identifier for a node in the KV web.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WebNodeId(pub usize);

/// Polygonal semantic region for polygonal‑KV geometry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolygonRegion {
    pub id: u32,
    pub centroid: Vec<f32>,
    pub radius: f32,
    pub face_index: u8,
}

/// A semantic or structural node over a span of tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebNode {
    pub id: WebNodeId,

    // RAW (uncompressed) token list — stored only if compressor is disabled.
    pub tokens: Vec<TokenId>,

    // COMPRESSED token list — BitDrop_v2 max‑tier
    pub tokens_compressed: Option<Vec<u8>>,

    pub label: Option<String>,
    pub label_compressed: Option<Vec<u8>>,

    pub score: f32,

    // Polygonal‑KV geometry (optional per node)
    pub polygon: Option<PolygonRegion>,

    // Diverging‑memory upgrade: branch metadata
    pub branch_id: Option<u32>,
    pub branch_kind: Option<u8>,      // e.g. 0=semantic,1=context,2=motion,3=color,...
    pub branch_stability: f32,        // higher = more stable branch
    pub branch_drift: f32,            // higher = more drift from canonical meaning

    // COMPRESSED branch metadata
    pub branch_meta_compressed: Option<Vec<u8>>,
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

    // Diverging‑memory upgrade: drift + reinforcement tracking
    pub drift_score: f32,
    pub reinforcement_score: f32,

    // COMPRESSED drift packet
    pub drift_packet_compressed: Option<Vec<u8>>,
}

impl NodeDriftState {
    pub fn new() -> Self {
        Self {
            last_access: Instant::now(),
            drift_score: 0.0,
            reinforcement_score: 0.0,
            drift_packet_compressed: None,
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

    #[serde(skip)]
    pub compressor: Option<KvWebCompressor>,
}

impl Default for KvWeb {
    fn default() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            node_index_by_token: HashMap::new(),
            next_node_id: 0,
            drift_state: None,
            compressor: Some(KvWebCompressor::new()),
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

        // ============================================================
        // MAX‑TIER BITDROP COMPRESSION
        // ============================================================

        let tokens_compressed = self
            .compressor
            .as_ref()
            .map(|c| c.compress(&tokens));

        // avoid generic inference issues by materializing the label first
        let label_string: Option<String> = label.map(Into::into);

        let label_compressed = label_string
            .as_ref()
            .and_then(|l| self.compressor.as_ref().map(|c| c.compress(l)));

        let branch_meta_compressed = self
            .compressor
            .as_ref()
            .map(|c| c.compress(&(
                None::<u32>,
                None::<u8>,
                0.0f32,
                0.0f32,
            )));

        let node = WebNode {
            id,
            tokens: tokens.clone(),
            tokens_compressed,
            label: label_string,
            label_compressed,
            score,

            // Polygonal‑KV defaults
            polygon: None,

            // Diverging‑memory defaults
            branch_id: None,
            branch_kind: None,
            branch_stability: 0.0,
            branch_drift: 0.0,

            branch_meta_compressed,
        };

        for t in &tokens {
            self.node_index_by_token.insert(*t, id);
        }

        self.nodes.insert(id, node);

        // Lazily initialize drift_state map and entry for this node
        if let Some(drift) = &mut self.drift_state {
            let mut ds = NodeDriftState::new();

            // Compress drift packet immediately
            if let Some(comp) = &self.compressor {
                let packet = comp.compress(&(ds.drift_score, ds.reinforcement_score));
                ds.drift_packet_compressed = Some(packet);
            }

            drift.insert(id, ds);
        }

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

// ============================================================================
// MAX‑TIER CORE WEB OPTIMIZATION LOOP (added, no logic removed)
// ============================================================================

/// Optimization config for core KvWeb behavior.
#[derive(Debug, Clone)]
pub struct KvWebOptimizationConfig {
    pub min_node_capacity: usize,
    pub max_node_capacity: usize,
    pub min_edge_capacity: usize,
    pub max_edge_capacity: usize,
    pub min_compression_threshold: usize,
    pub max_compression_threshold: usize,
}

/// Optimization state for core KvWeb.
#[derive(Debug, Clone)]
pub struct KvWebOptimizationState {
    pub node_capacity: usize,
    pub edge_capacity: usize,
    pub compression_threshold: usize,
}

impl Default for KvWebOptimizationState {
    fn default() -> Self {
        Self {
            node_capacity: 1024,
            edge_capacity: 4096,
            compression_threshold: 256,
        }
    }
}

/// Max‑tier optimization loop for KvWeb.
/// Adjusts capacities and compression threshold based on current web size.
pub fn optimize_kv_web(
    web: &mut KvWeb,
    state: &mut KvWebOptimizationState,
    cfg: &KvWebOptimizationConfig,
) {
    let node_count = web.nodes.len();
    let edge_count = web.edges.len();

    // 1) Node capacity tuning
    if node_count > state.node_capacity {
        state.node_capacity = (state.node_capacity * 2).min(cfg.max_node_capacity);
    } else if node_count < state.node_capacity / 2 {
        state.node_capacity = (state.node_capacity / 2).max(cfg.min_node_capacity);
    }

    // 2) Edge capacity tuning
    if edge_count > state.edge_capacity {
        state.edge_capacity = (state.edge_capacity * 2).min(cfg.max_edge_capacity);
    } else if edge_count < state.edge_capacity / 2 {
        state.edge_capacity = (state.edge_capacity / 2).max(cfg.min_edge_capacity);
    }

    // 3) Compression threshold tuning (fixed: cast to f32, then back to usize)
    if node_count < cfg.min_compression_threshold {
        state.compression_threshold =
            ((state.compression_threshold as f32 * 0.9) as usize).max(cfg.min_compression_threshold);
    } else if node_count > cfg.max_compression_threshold {
        state.compression_threshold =
            ((state.compression_threshold as f32 * 1.1) as usize).min(cfg.max_compression_threshold);
    }

    // Optional: enable/disable compressor based on threshold
    if node_count >= state.compression_threshold {
        if web.compressor.is_none() {
            web.compressor = Some(KvWebCompressor::new());
        }
    } else {
        // keep compressor but do not remove it to stay backwards‑compatible
    }

    // No BitDrop packets here — this is structural optimization only.
}
