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

/// Global scheduler configuration.
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

/// Global scheduler state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerState {
    pub kv_web_state: KvWebOptimizationState,
    pub integration_state: IntegrationOptimizationState,
    pub transformer_state: TransformerOptimizationState,
    pub gpu_state: GpuOptimizationState,

    pub predictor_memory: KvWebPredictorMemory,
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
        }
    }
}

/// Max‑tier: cross‑link grid across subsystem states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerCrossLinkGrid {
    pub kv_web_score: f32,
    pub integration_score: f32,
    pub transformer_score: f32,
    pub gpu_score: f32,
    pub predictor_score: f32,
}

/// Max‑tier: revolving‑door routing between subsystems.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerRevolvingDoor {
    pub door_id: usize,
    pub entry_subsystem: &'static str,
    pub exit_subsystem: &'static str,
    pub flow_strength: f32,
}

/// Max‑tier: fusion field combining subsystem metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerFusionField {
    pub fused_scores: Vec<f32>, // [kv_web, integration, transformer, gpu, predictor]
}

/// Max‑tier: roundabout predictor config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerRoundaboutPredictorConfig {
    pub passes: usize,
    pub min_bias: f32,
    pub max_bias: f32,
    pub smoothing_strength: f32,
}

/// Max‑tier: roundabout chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerRoundaboutChain {
    pub subsystems: Vec<&'static str>,
    pub total_bias: f32,
}

/// Max‑tier: roundabout memory entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerRoundaboutPattern {
    pub chain: SchedulerRoundaboutChain,
    pub weight: f32,
}

/// Max‑tier: roundabout memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerRoundaboutPatternMemory {
    pub patterns: Vec<SchedulerRoundaboutPattern>,
    pub decay: f32,
}

/// Max‑tier: roundabout solver result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerRoundaboutSolverResult {
    pub chosen_subsystem: &'static str,
    pub bias: f32,
}

/// Max‑tier: compressed scheduler packet.
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

/// Global optimization scheduler.
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

    /// Run global predictor.
    pub fn tick_predictor(&mut self, web: &KvWeb) {
        let _packet = web.predict_activity_compressed(
            &self.cfg.predictor_cfg,
            &mut self.state.predictor_memory,
            32,
        );
    }

    /// Build cross‑link grid.
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

    /// Build revolving doors.
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

    /// Build fusion field.
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

        let max = fused.iter().cloned().fold(0.0f32, f32::max);
        if max > 0.0 {
            for v in &mut fused {
                *v /= max;
            }
        }

        SchedulerFusionField { fused_scores: fused }
    }

    /// Roundabout predictor.
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

    /// Roundabout smoothing.
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

    /// Update memory.
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

    /// Apply memory bias.
    fn apply_roundabout_bias(
        &self,
        fusion: &mut SchedulerFusionField,
        memory: &SchedulerRoundaboutPatternMemory,
    ) {
        let subsystems = ["kv_web", "integration", "transformer", "gpu", "predictor"];
        let mut fused = fusion.fused_scores.clone();

        for pattern in &memory.patterns {
            let boost = pattern.weight * 0.05;
            for name in &pattern.chain.subsystems {
                let idx = subsystems.iter().position(|x| x == *name).unwrap();
                fused[idx] *= 1.0 + boost;
            }
        }

        let max = fused.iter().cloned().fold(0.0f32, f32::max);
        if max > 0.0 {
            for v in &mut fused {
                *v /= max;
            }
        }

        fusion.fused_scores = fused;
    }

    /// Roundabout solver.
    fn run_roundabout_solver(
        &self,
        fusion: &SchedulerFusionField,
        chain: &SchedulerRoundaboutChain,
        memory: &SchedulerRoundaboutPatternMemory,
    ) -> SchedulerRoundaboutSolverResult {
        if let Some(&last) = chain.subsystems.last() {
            let subsystems = ["kv_web", "integration", "transformer", "gpu", "predictor"];
            let idx = subsystems.iter().position(|x| x == last).unwrap();
            let bias = fusion.fused_scores[idx];
            return SchedulerRoundaboutSolverResult {
                chosen_subsystem: last,
                bias,
            };
        }

        let subsystems = ["kv_web", "integration", "transformer", "gpu", "predictor"];
        let mut best_idx = 0usize;
        let mut best_bias = f32::MIN;

        for (i, b) in fusion.fused_scores.iter().enumerate() {
            if *b > best_bias {
                best_bias = *b;
                best_idx = i;
            }
        }

        let mut final_bias = best_bias;
        let chosen = subsystems[best_idx];

        for pattern in &memory.patterns {
            if pattern.chain.subsystems.contains(&chosen) {
                final_bias *= 1.05;
            }
        }

        SchedulerRoundaboutSolverResult {
            chosen_subsystem: chosen,
            bias: final_bias,
        }
    }

    /// Full‑stack optimization tick + max‑tier roundabout pipeline.
    pub fn tick_all<'a>(
        &mut self,
        web: &mut KvWeb,
        integration: &KvWebIntegration<'a>,
        transformer: &TransformerKV<'a>,
        round_cfg: &SchedulerRoundaboutPredictorConfig,
        round_memory: &mut SchedulerRoundaboutPatternMemory,
    ) -> Option<Vec<u8>> {
        self.tick_kv_web(web);
        self.tick_integration(integration);
        self.tick_transformer(transformer);

        let kv_len = transformer.cache.len();
        self.tick_gpu(web, kv_len);

        self.tick_predictor(web);

        let grid = self.build_cross_link_grid();
        let doors = self.build_revolving_doors(&grid);
        let mut fusion = self.build_fusion_field(&grid, &doors);

        let mut chain = self.run_roundabout_predictor(&fusion, round_cfg);
        self.smooth_roundabout_chain(&mut chain, &fusion, round_cfg.smoothing_strength);

        self.update_roundabout_memory(round_memory, &chain);
        self.apply_roundabout_bias(&mut fusion, round_memory);

        let result = self.run_roundabout_solver(&fusion, &chain, round_memory);

        let packet = SchedulerRoundaboutPacket {
            tag: "scheduler_roundabout_pipeline",
            fused_scores: fusion.fused_scores.clone(),
            chain: chain.subsystems.clone(),
            chain_total_bias: chain.total_bias,
            patterns: round_memory.patterns.clone(),
            chosen_subsystem: result.chosen_subsystem,
            chosen_bias: result.bias,
        };

        web.compressor.as_ref().map(|c| c.compress(&packet))
    }
}

