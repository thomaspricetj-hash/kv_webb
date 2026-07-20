//! kv_web_integration
//!
//! Integration layer between KV‑cache webbing and transformer KV‑cache tensors.
//!
//! This crate does NOT implement a transformer. Instead, it provides:
//! - mapping between TokenId → KV tensor index
//! - selecting subsets of KV entries based on web regions
//! - building attention masks from web queries
//! - optional GPU acceleration for mask building
//! - visualization utilities
//! - export utilities (JSON, KV subsets)
//!
//! The transformer still sees a flat KV cache, but this layer lets you
//! apply *web‑aware selection* to restrict or weight attention.

pub mod gpu;
pub mod transformer;
pub mod visualize;
pub mod export;   // ★ NEW: export utilities wired in

use kv_web_core::{KvWeb, TokenId, WebNodeId};
use kv_web_runtime::KvWebRuntime;
use crate::gpu::{GpuContext, build_attention_mask_gpu};

/// A simple KV‑cache representation for integration.
/// In a real system, this would be the transformer's KV tensors.
#[derive(Debug)]
pub struct KvCache {
    pub keys: Vec<Vec<f32>>,   // placeholder for K vectors
    pub values: Vec<Vec<f32>>, // placeholder for V vectors
}

impl KvCache {
    pub fn len(&self) -> usize {
        self.keys.len()
    }
}

/// Integration utilities for KV‑cache webbing.
pub struct KvWebIntegration<'a> {
    pub web: &'a KvWeb,
    pub cache: &'a KvCache,
    pub gpu: Option<&'a GpuContext>,   // optional GPU context
}

impl<'a> KvWebIntegration<'a> {
    pub fn new(web: &'a KvWeb, cache: &'a KvCache) -> Self {
        Self { web, cache, gpu: None }
    }

    /// Enable GPU acceleration.
    pub fn with_gpu(mut self, gpu: &'a GpuContext) -> Self {
        self.gpu = Some(gpu);
        self
    }

    /// Map a set of TokenIds to KV‑cache indices.
    pub fn tokens_to_indices(&self, tokens: &Vec<TokenId>) -> Vec<usize> {
        tokens.iter().map(|t| t.0).collect()
    }

    /// Select a subset of KV‑cache entries based on a web region.
    pub fn select_region(&self, root: WebNodeId, depth: usize) -> Vec<usize> {
        let tokens = self.web.tokens_in_region(root, depth);
        tokens.into_iter().map(|t| t.0).collect()
    }

    /// Build an attention mask where:
    /// - tokens in the region get weight 1.0
    /// - tokens outside get weight 0.0
    ///
    /// GPU‑accelerated if available.
    pub fn attention_mask(&self, root: WebNodeId, depth: usize) -> Vec<f32> {
        build_attention_mask_gpu(
            self.web,
            root,
            depth,
            self.cache.len(),
            self.gpu,
        )
    }

    /// Build a *soft* attention mask where:
    /// - region tokens get weight 1.0
    /// - neighbors get weight 0.5
    /// - everything else gets 0.0
    ///
    /// (CPU only — GPU version coming later)
    pub fn soft_attention_mask(&self, root: WebNodeId, depth: usize) -> Vec<f32> {
        let mut mask = vec![0.0; self.cache.len()];

        let region_nodes = self.web.nodes_in_region(root, depth);
        let mut region_tokens = Vec::new();

        for node_id in &region_nodes {
            if let Some(node) = self.web.nodes.get(node_id) {
                region_tokens.extend(node.tokens.clone());
            }
        }

        // region tokens → strong weight
        for t in &region_tokens {
            if t.0 < mask.len() {
                mask[t.0] = 1.0;
            }
        }

        // neighbor tokens → medium weight
        for neighbor in self.web.neighbors(root) {
            for t in &neighbor.tokens {
                if t.0 < mask.len() && mask[t.0] == 0.0 {
                    mask[t.0] = 0.5;
                }
            }
        }

        mask
    }

    /// Extract KV entries for a region.
    pub fn kv_subset(&self, root: WebNodeId, depth: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let indices = self.select_region(root, depth);

        let keys: Vec<Vec<f32>> = indices.iter().map(|i| self.cache.keys[*i].clone()).collect();
        let values: Vec<Vec<f32>> = indices.iter().map(|i| self.cache.values[*i].clone()).collect();

        (keys, values)
    }
}
