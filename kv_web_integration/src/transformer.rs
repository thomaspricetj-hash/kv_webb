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

// ============================================================================
// ⭐ MAX‑TIER TRANSFORMER OPTIMIZATION LOOP (added, no logic removed)
// ============================================================================

/// Optimization config for transformer KV selection.
#[derive(Debug, Clone)]
pub struct TransformerOptimizationConfig {
    pub min_depth: usize,
    pub max_depth: usize,
    pub target_mask_density: f32,
    pub max_mask_density: f32,
    pub min_gpu_threshold: usize,
    pub max_gpu_threshold: usize,
}

/// Optimization state for transformer KV routing.
#[derive(Debug, Clone)]
pub struct TransformerOptimizationState {
    pub depth: usize,
    pub gpu_threshold: usize,
}

impl Default for TransformerOptimizationState {
    fn default() -> Self {
        Self {
            depth: 3,
            gpu_threshold: 512,
        }
    }
}

/// Max‑tier optimization loop for transformer KV routing.
/// Adjusts depth, mask density, and GPU/CPU crossover.
pub fn optimize_transformer_kv(
    transformer: &TransformerKV,
    root: WebNodeId,
    state: &mut TransformerOptimizationState,
    cfg: &TransformerOptimizationConfig,
) {
    let kv_len = transformer.cache.len();
    let mask = transformer.hard_mask(root, state.depth);

    // Compute mask density
    let mut active = 0.0;
    for v in &mask {
        if *v > 0.0 {
            active += 1.0;
        }
    }
    let density = if kv_len > 0 {
        active / kv_len as f32
    } else {
        0.0
    };

    // 1) Depth tuning
    if density < cfg.target_mask_density && state.depth < cfg.max_depth {
        state.depth += 1;
    } else if density > cfg.max_mask_density && state.depth > cfg.min_depth {
        state.depth -= 1;
    }

    // 2) GPU threshold tuning (fixed: cast to f32, then back to usize)
    if active < cfg.min_gpu_threshold as f32 {
        state.gpu_threshold =
            ((state.gpu_threshold as f32 * 0.9) as usize).max(cfg.min_gpu_threshold);
    } else if active > cfg.max_gpu_threshold as f32 {
        state.gpu_threshold =
            ((state.gpu_threshold as f32 * 1.1) as usize).min(cfg.max_gpu_threshold);
    }

    // No compression here — transformer optimization is runtime-only.
}
