//! transformer.rs
//!
//! Web‑aware transformer KV‑cache integration.
//!
//! This module does NOT implement a transformer. Instead, it provides:
//! - region‑aware KV selection
//! - web‑aware attention mask building
//! - optional GPU acceleration
//! - hooks for plugging into any transformer backend
//!
//! The transformer still sees a flat KV cache, but this layer lets you
//! apply *web‑aware selection* to restrict or weight attention.

use crate::gpu::{GpuContext, build_attention_mask_gpu};
use crate::KvCache;
use kv_web_core::{KvWeb, WebNodeId};
use kv_web_runtime::KvWebRuntime;

/// A simple wrapper for transformer KV‑cache operations.
pub struct TransformerKV<'a> {
    pub web: &'a KvWeb,
    pub cache: &'a KvCache,
    pub gpu: Option<&'a GpuContext>,
}

impl<'a> TransformerKV<'a> {
    pub fn new(web: &'a KvWeb, cache: &'a KvCache) -> Self {
        Self { web, cache, gpu: None }
    }

    pub fn with_gpu(mut self, gpu: &'a GpuContext) -> Self {
        self.gpu = Some(gpu);
        self
    }

    /// Build a hard attention mask:
    /// - region tokens → 1.0
    /// - outside region → 0.0
    pub fn hard_mask(&self, root: WebNodeId, depth: usize) -> Vec<f32> {
        build_attention_mask_gpu(
            self.web,
            root,
            depth,
            self.cache.len(),
            self.gpu,
        )
    }

    /// Build a soft attention mask:
    /// - region tokens → 1.0
    /// - neighbors → 0.5
    /// - outside → 0.0
    pub fn soft_mask(&self, root: WebNodeId, depth: usize) -> Vec<f32> {
        let mut mask = vec![0.0; self.cache.len()];

        let region_nodes = self.web.nodes_in_region(root, depth);
        let mut region_tokens = Vec::new();

        for node_id in &region_nodes {
            if let Some(node) = self.web.nodes.get(node_id) {
                region_tokens.extend(node.tokens.clone());
            }
        }

        // region → strong
        for t in &region_tokens {
            if t.0 < mask.len() {
                mask[t.0] = 1.0;
            }
        }

        // neighbors → medium
        for neighbor in self.web.neighbors(root) {
            for t in &neighbor.tokens {
                if t.0 < mask.len() && mask[t.0] == 0.0 {
                    mask[t.0] = 0.5;
                }
            }
        }

        mask
    }

    /// Extract KV subset for a region.
    pub fn kv_subset(&self, root: WebNodeId, depth: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let tokens = self.web.tokens_in_region(root, depth);
        let mut keys = Vec::new();
        let mut values = Vec::new();

        for t in tokens {
            if t.0 < self.cache.keys.len() {
                keys.push(self.cache.keys[t.0].clone());
                values.push(self.cache.values[t.0].clone());
            }
        }

        (keys, values)
    }

    /// Apply a mask to a transformer attention matrix.
    /// This is backend‑agnostic: you pass in your attention matrix.
    pub fn apply_mask(&self, attn: &mut [f32], mask: &[f32]) {
        for (a, m) in attn.iter_mut().zip(mask.iter()) {
            *a *= *m;
        }
    }

    /// Full pipeline:
    /// - build mask
    /// - apply mask to attention
    /// - return masked attention
    pub fn masked_attention(
        &self,
        root: WebNodeId,
        depth: usize,
        attn: &mut [f32],
    ) {
        let mask = self.hard_mask(root, depth);
        self.apply_mask(attn, &mask);
    }
}
