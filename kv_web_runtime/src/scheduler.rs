//! scheduler.rs
//!
//! Global optimization scheduler for KV‑Webb + BitDrop_v2 + Polygonal‑KV geometry.
//!
//! This ties together:
//! - core KvWeb optimization
//! - integration optimization
//! - transformer KV optimization
//! - GPU mask‑building optimization
//!
//! It does NOT change any existing logic; it only coordinates the
//! per‑module optimization loops you already defined.

use kv_web_core::{
    KvWeb,
    KvWebOptimizationConfig,
    KvWebOptimizationState,
    optimize_kv_web,
};

use kv_web_integration::{
    KvWebIntegration,
    IntegrationOptimizationConfig,
    IntegrationOptimizationState,
    optimize_integration,
};

use kv_web_integration::gpu::{
    GpuOptimizationConfig,
    GpuOptimizationState,
    optimize_gpu,
};

use kv_web_integration::transformer::{
    TransformerKV,
    TransformerOptimizationConfig,
    TransformerOptimizationState,
    optimize_transformer_kv,
};

use kv_web_core::WebNodeId;

/// Global scheduler configuration.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub kv_web_cfg: KvWebOptimizationConfig,
    pub integration_cfg: IntegrationOptimizationConfig,
    pub transformer_cfg: TransformerOptimizationConfig,
    pub gpu_cfg: GpuOptimizationConfig,

    /// Root node for region‑based optimizations.
    pub default_root: WebNodeId,
    /// Default depth for region queries when optimizing.
    pub default_depth: usize;
}

/// Global scheduler state.
#[derive(Debug, Clone)]
pub struct SchedulerState {
    pub kv_web_state: KvWebOptimizationState,
    pub integration_state: IntegrationOptimizationState,
    pub transformer_state: TransformerOptimizationState,
    pub gpu_state: GpuOptimizationState,
}

impl Default for SchedulerState {
    fn default() -> Self {
        Self {
            kv_web_state: KvWebOptimizationState::default(),
            integration_state: IntegrationOptimizationState::default(),
            transformer_state: TransformerOptimizationState::default(),
            gpu_state: GpuOptimizationState::default(),
        }
    }
}

/// Global optimization scheduler.
/// Owns no data; just coordinates optimization across subsystems.
pub struct KvWebScheduler {
    pub cfg: SchedulerConfig,
    pub state: SchedulerState,
}

impl KvWebScheduler {
    pub fn new(cfg: SchedulerConfig) -> Self {
        Self {
            cfg,
            state: SchedulerState::default(),
        }
    }

    /// Run core KvWeb optimization.
    pub fn tick_kv_web(&mut self, web: &mut KvWeb) {
        optimize_kv_web(
            web,
            &mut self.state.kv_web_state,
            &self.cfg.kv_web_cfg,
        );
    }

    /// Run integration optimization.
    pub fn tick_integration<'a>(
        &mut self,
        integration: &KvWebIntegration<'a>,
    ) {
        optimize_integration(
            integration,
            self.cfg.default_root,
            &mut self.state.integration_state,
            &self.cfg.integration_cfg,
        );
    }

    /// Run transformer KV optimization.
    pub fn tick_transformer<'a>(
        &mut self,
        transformer: &TransformerKV<'a>,
    ) {
        optimize_transformer_kv(
            transformer,
            self.cfg.default_root,
            &mut self.state.transformer_state,
            &self.cfg.transformer_cfg,
        );
    }

    /// Run GPU optimization for mask building.
    pub fn tick_gpu(
        &mut self,
        web: &KvWeb,
        kv_len: usize,
    ) {
        optimize_gpu(
            web,
            self.cfg.default_root,
            self.cfg.default_depth,
            kv_len,
            &mut self.state.gpu_state,
            &self.cfg.gpu_cfg,
        );
    }

    /// Full‑stack optimization tick:
    /// - core KvWeb
    /// - integration
    /// - transformer
    /// - GPU
    pub fn tick_all<'a>(
        &mut self,
        web: &mut KvWeb,
        integration: &KvWebIntegration<'a>,
        transformer: &TransformerKV<'a>,
    ) {
        // 1) core web
        self.tick_kv_web(web);

        // 2) integration (depends on web)
        self.tick_integration(integration);

        // 3) transformer KV routing (depends on web + cache)
        self.tick_transformer(transformer);

        // 4) GPU tuning (depends on web + cache length)
        let kv_len = transformer.cache.len();
        self.tick_gpu(web, kv_len);
    }
}
