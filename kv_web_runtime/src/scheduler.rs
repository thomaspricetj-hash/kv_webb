//! scheduler.rs
//!
//! Global optimization scheduler for KV‑Webb + BitDrop_v2 + Polygonal‑KV geometry.
//!
//! This ties together:
//! - core KvWeb optimization
//! - integration optimization
//! - transformer KV optimization
//! - GPU mask‑building optimization
//! - predictor subsystem (NEW)
//!
//! Max‑tier upgrades:
//! - cross‑link grid over subsystem states
//! - revolving‑door routing between subsystem flows
//! - fusion field combining subsystem metrics
//! - roundabout predictor + smoothing + memory + solver
//! - GPU‑ready compressed scheduler packets
//! - stability‑weighted subsystem routing
//! - subsystem‑level overflow buffer + volatility absorption
//! - subsystem‑level tunnel metrics + reliability scoring
//! - subsystem‑level cognitive weight + reinforcement
//!
//! All original logic preserved.

use kv_web_core::{
    KvWeb,
    KvWebOptimizationConfig,
    KvWebOptimizationState,
    optimize_kv_web,
    WebNodeId,
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

use kv_web_runtime::predictor::{
    KvWebPredictor,
    KvWebPredictorConfig,
    KvWebPredictorMemory,
};

use serde::{Serialize, Deserialize};
use std::time::Instant;

// ────────────────────────────────────────────────────────────────
//   MAX‑TIER SUBSYSTEM METRICS (TUNNEL, CACHE, OVERFLOW, COGNITIVE)
// ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubsystemTunnelMetrics {
    pub latency_ms: f32,
    pub jitter_ms: f32,
    pub congestion: f32,
    pub stability: f32,
    pub loss_rate: f32,
}

impl SubsystemTunnelMetrics {
    pub fn stable_default() -> Self {
        Self {
            latency_ms: 1.0,
            jitter_ms: 0.0,
            congestion: 0.1,
            stability: 1.0,
            loss_rate: 0.0,
        }
    }

    pub fn score(&self) -> f32 {
        let latency_term = (1.0 / (1.0 + self.latency_ms)).min(1.0);
        let jitter_term = (1.0 - self.jitter_ms).max(0.0);
        let congestion_term = (1.0 - self.congestion).max(0.0);
        let stability_term = self.stability;
        let loss_term = (1.0 - self.loss_rate).max(0.0);

        (latency_term * 0.2)
            + (jitter_term * 0.2)
            + (congestion_term * 0.2)
            + (stability_term * 0.3)
            + (loss_term * 0.1)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubsystemCacheMetrics {
    pub hit: u64,
    pub miss: u64,
    pub reliability: f32,
}

impl SubsystemCacheMetrics {
    pub fn new() -> Self {
        Self {
            hit: 0,
            miss: 0,
            reliability: 1.0,
        }
    }

    pub fn record_hit(&mut self) {
        self.hit += 1;
        self.update();
    }

    pub fn record_miss(&mut self) {
        self.miss += 1;
        self.update();
    }

    fn update(&mut self) {
        let total = self.hit + self.miss;
        if total == 0 {
            self.reliability = 1.0;
        } else {
            self.reliability = (self.hit as f32 / total as f32).clamp(0.1, 1.0);
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubsystemOverflowBuffer {
    pub volatility: f32,
    pub absorption: f32,
}

impl SubsystemOverflowBuffer {
    pub fn new() -> Self {
        Self {
            volatility: 0.0,
            absorption: 0.5,
        }
    }

    pub fn absorb(&mut self, spike: f32) {
        self.volatility = (self.volatility * 0.9) + spike * 0.1;
    }

    pub fn stabilized_bias(&self) -> f32 {
        (1.0 - self.volatility * 0.1).clamp(0.5, 1.0) * self.absorption
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubsystemCognitiveState {
    pub cognitive_weight: f32,
    pub stability_factor: f32,
}

impl SubsystemCognitiveState {
    pub fn new() -> Self {
        Self {
            cognitive_weight: 1.0,
            stability_factor: 1.0,
        }
    }

    pub fn reinforce(&mut self, success: bool) {
        if success {
            self.cognitive_weight = (self.cognitive_weight + 0.05).min(2.0);
            self.stability_factor = (self.stability_factor + 0.05).min(2.0);
        } else {
            self.cognitive_weight = (self.cognitive_weight * 0.97).max(0.5);
            self.stability_factor = (self.stability_factor - 0.05).max(0.1);
        }
    }
}

// ────────────────────────────────────────────────────────────────
//   ORIGINAL SCHEDULER CONFIG + STATE
// ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    pub kv_web_cfg: KvWebOptimizationConfig,
    pub integration_cfg: IntegrationOptimizationConfig,
    pub transformer_cfg: TransformerOptimizationConfig,
    pub gpu_cfg: GpuOptimizationConfig,

    pub predictor_cfg: KvWebPredictorConfig,

    pub default_root: WebNodeId,
    pub default_depth: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerState {
    pub kv_web_state: KvWebOptimizationState,
    pub integration_state: IntegrationOptimizationState,
    pub transformer_state: TransformerOptimizationState,
    pub gpu_state: GpuOptimizationState,

    pub predictor_memory: KvWebPredictorMemory,

    // MAX‑tier subsystem metrics
    pub tunnel: SubsystemTunnelMetrics,
    pub cache: SubsystemCacheMetrics,
    pub overflow: SubsystemOverflowBuffer,
    pub cognitive: SubsystemCognitiveState,
}

impl Default for SchedulerState {
    fn default() -> Self {
        Self {
            kv_web_state: KvWebOptimizationState::default(),
            integration_state: IntegrationOptimizationState::default(),
            transformer_state: TransformerOptimizationState::default(),
            gpu_state: GpuOptimizationState::default(),
            predictor_memory: KvWebPredictorMemory {
                patterns: Vec::new(),
                decay: 0.9,
            },

            tunnel: SubsystemTunnelMetrics::stable_default(),
            cache: SubsystemCacheMetrics::new(),
            overflow: SubsystemOverflowBuffer::new(),
            cognitive: SubsystemCognitiveState::new(),
        }
    }
}

// ────────────────────────────────────────────────────────────────
//   CROSS‑LINK GRID + REVOLVING DOORS + FUSION FIELD
// ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerCrossLinkGrid {
    pub kv_web_score: f32,
    pub integration_score: f32,
    pub transformer_score: f32,
    pub gpu_score: f32,
    pub predictor_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerRevolvingDoor {
    pub door_id: usize,
    pub entry_subsystem: &'static str,
    pub exit_subsystem: &'static str,
    pub flow_strength: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerFusionField {
    pub fused_scores: Vec<f32>, // [kv_web, integration, transformer, gpu, predictor]
}

// ────────────────────────────────────────────────────────────────
//   ROUNDABOUT PREDICTOR + MEMORY + SOLVER
// ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerRoundaboutPredictorConfig {
    pub passes: usize,
    pub min_bias: f32,
    pub max_bias: f32,
    pub smoothing_strength: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerRoundaboutChain {
    pub subsystems: Vec<&'static str>,
    pub total_bias: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerRoundaboutPattern {
    pub chain: SchedulerRoundaboutChain,
    pub weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerRoundaboutPatternMemory {
    pub patterns: Vec<SchedulerRoundaboutPattern>,
    pub decay: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerRoundaboutSolverResult {
    pub chosen_subsystem: &'static str,
    pub bias: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerRoundaboutPacket {
    pub tag: &'static str,
    pub fused_scores: Vec<f32>,
    pub chain: Vec<&'static str>,
    pub chain_total_bias: f32,
    pub patterns: Vec<SchedulerRoundaboutPattern>,
    pub chosen_subsystem: &'static str,
    pub chosen_bias: f32,
}

// ────────────────────────────────────────────────────────────────
//   GLOBAL SCHEDULER
// ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
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

    // ────────────────────────────────────────────────────────────
    //   ORIGINAL TICK FUNCTIONS (UNCHANGED)
    // ────────────────────────────────────────────────────────────

    pub fn tick_kv_web(&mut self, web: &mut KvWeb) {
        optimize_kv_web(
            web,
            &mut self.state.kv_web_state,
            &self.cfg.kv_web_cfg,
        );
    }

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

    pub fn tick_predictor(&mut self, web: &KvWeb) {
        let _packet = web.predict_activity_compressed(
            &self.cfg.predictor_cfg,
            &mut self.state.predictor_memory,
            32,
        );
    }

    // ────────────────────────────────────────────────────────────
    //   MAX‑TIER CROSS‑LINK GRID
    // ────────────────────────────────────────────────────────────

    fn build_cross_link_grid(&self) -> SchedulerCrossLinkGrid {
        SchedulerCrossLinkGrid {
            kv_web_score: self.state.kv_web_state.last_score,
            integration_score: self.state.integration_state.last_score,
            transformer_score: self.state.transformer_state.last_score,
            gpu_score: self.state.gpu_state.last_score,
            predictor_score: self.state.predictor_memory.patterns.last()
                .map(|p| p.weight)
                .unwrap_or(0.0),
        }
    }

    // ────────────────────────────────────────────────────────────
    //   MAX‑TIER REVOLVING DOORS
    // ────────────────────────────────────────────────────────────

    fn build_revolving_doors(&self, grid: &SchedulerCrossLinkGrid) -> Vec<SchedulerRevolvingDoor> {
        let mut doors = Vec::new();

        let scores = [
            ("kv_web", grid.kv_web_score),
            ("integration", grid.integration_score),
            ("transformer", grid.transformer_score),
            ("gpu", grid.gpu_score),
            ("predictor", grid.predictor_score),
        ];

        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let entry = sorted.last().unwrap();
        let exit = sorted.first().unwrap();

        let flow_strength = (exit.1 - entry.1).abs();

        doors.push(SchedulerRevolvingDoor {
            door_id: 0,
            entry_subsystem: entry.0,
            exit_subsystem: exit.0,
            flow_strength,
        });

        doors
    }

    // ────────────────────────────────────────────────────────────
    //   MAX‑TIER FUSION FIELD
    // ────────────────────────────────────────────────────────────

    fn build_fusion_field(&self, grid: &SchedulerCrossLinkGrid, doors: &[SchedulerRevolvingDoor]) -> SchedulerFusionField {
        let mut fused = vec![
            grid.kv_web_score,
            grid.integration_score,
            grid.transformer_score,
            grid.gpu_score,
            grid.predictor_score,
        ];

        for door in doors {
            for (i, name) in ["kv_web", "integration", "transformer", "gpu", "predictor"]
                .iter()
                .enumerate()
            {
                if *name == door.exit_subsystem {
                    fused[i] *= 1.0 + door.flow_strength * 0.1;
                } else if *name == door.entry_subsystem {
                    fused[i] *= 1.0 - door.flow_strength * 0.05;
                }
            }
        }

        // MAX‑tier: apply tunnel + cache + overflow + cognitive bias
        let tunnel_score = self.state.tunnel.score();
        let cache_score = self.state.cache.reliability;
        let overflow_bias = self.state.overflow.stabilized_bias();
        let cognitive_weight = self.state.cognitive.cognitive_weight;

        for v in &mut fused {
            *v = *v * 0.7
                + tunnel_score * 0.1
                + cache_score * 0.1
                + overflow_bias * 0.05
                + cognitive_weight * 0.05;
        }

        let max = fused.iter().cloned().fold(0.0f32, f32::max);
        if max > 0.0 {
            for v in &mut fused {
                *v /= max;
            }
        }

        SchedulerFusionField { fused_scores: fused }
    }

    // ────────────────────────────────────────────────────────────
    //   MAX‑TIER ROUNDABOUT PREDICTOR
    // ────────────────────────────────────────────────────────────

    fn run_roundabout_predictor(
        &self,
        fusion: &SchedulerFusionField,
        cfg: &SchedulerRoundaboutPredictorConfig,
    ) -> SchedulerRoundaboutChain {
        let subsystems = ["kv_web", "integration", "transformer", "gpu", "predictor"];
        let mut visited = vec![false; 5];
        let mut chain = Vec::new();
        let mut total = 0.0f32;

        for _ in 0..cfg.passes {
            let mut best_idx = None;
            let mut best_bias = cfg.min_bias;

            for (i, b) in fusion.fused_scores.iter().enumerate() {
                if visited[i] {
                    continue;
                }
                if *b > best_bias && *b <= cfg.max_bias {
                    best_bias = *b;
                    best_idx = Some(i);
                }
            }

            if let Some(idx) = best_idx {
                visited[idx] = true;
                chain.push(subsystems[idx]);
                total += best_bias;
            } else {
                break;
            }
        }

        SchedulerRoundaboutChain {
            subsystems: chain,
            total_bias: total,
        }
    }

    // ────────────────────────────────────────────────────────────
    //   MAX‑TIER ROUNDABOUT SMOOTHING
    // ────────────────────────────────────────────────────────────

    fn smooth_roundabout_chain(
        &self,
        chain: &mut SchedulerRoundaboutChain,
        fusion: &SchedulerFusionField,
        strength: f32,
    ) {
        if chain.subsystems.len() < 3 {
            return;
        }

        let subsystems = ["kv_web", "integration", "transformer", "gpu", "predictor"];
        let mut new_total = 0.0f32;

        for (i, name) in chain.subsystems.iter().enumerate() {
            let mut local_sum = 0.0f32;
            let mut local_count = 0.0f32;

            for j in i.saturating_sub(1)..=(i + 1).min(chain.subsystems.len() - 1) {
                let idx = subsystems.iter().position(|x| x == chain.subsystems[j]).unwrap();
                local_sum += fusion.fused_scores[idx];
                local_count += 1.0;
            }

            if local_count > 0.0 {
                let avg = local_sum / local_count;
                let idx = subsystems.iter().position(|x| x == *name).unwrap();
                let base = fusion.fused_scores[idx];
                new_total += avg * strength + base * (1.0 - strength);
            }
        }

        chain.total_bias = new_total;
    }

    // ────────────────────────────────────────────────────────────
    //   MAX‑TIER ROUNDABOUT MEMORY
    // ────────────────────────────────────────────────────────────

    fn update_roundabout_memory(
        &self,
        memory: &mut SchedulerRoundaboutPatternMemory,
        chain: &SchedulerRoundaboutChain,
    ) {
        for pattern in &mut memory.patterns {
            pattern.weight *= memory.decay;
        }

        memory.patterns.push(SchedulerRoundaboutPattern {
            chain: chain.clone(),
            weight: 1.0,
        });

        memory.patterns.retain(|p| p.weight > 0.01);
    }

    // ────────────────────────────────────────────────────────────
    //   MAX‑TIER ROUNDABOUT MEMORY BIAS
    // ────────────────────────────────────────────────────────────

    fn apply_roundabout_bias(
        &self,
        fusion: &mut SchedulerFusionField,
        memory: &SchedulerRoundaboutPatternMemory,
    ) {
        let subsystems = ["kv_web",

