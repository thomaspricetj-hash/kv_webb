//! semantic.rs
//!
//! Semantic clustering utilities for KV Web with polygonal KV geometry.
//!
//! Upgrades added:
//! - parallel token clustering
//! - polygon-aware semantic similarity
//! - dual-layer scratch pads (semantic + geometry)
//! - semantic zoning + indexing
//! - GPU-ready compressed packets
//! - parallel semantic node linking
//! - Prompt‑Injection Firewall (SPIF) + multi-attack semantic firewalls
//! - Auto‑Threat Detection Engine (ATDE) with hybrid detect/log/block/adapt
//!
//! MAX‑tier upgrades added:
//! - semantic roundabout hubs per root node
//! - multilayer exit scoring over semantic zones
//! - circulation when no stable semantic exit is available
//! - scratchpad‑aware semantic exit hinting
//! - stability‑weighted semantic routing decisions
//!
//! MAX‑tier firewall upgrades added:
//! - Reverse‑Adversarial Hardening (RAH)
//! - Attack signature capture + rolling history
//! - Reverse‑mask generation over adversarial patterns
//! - Zone‑aware reverse masks
//! - Adversarial signature clustering
//! - Reverse‑weighted firewall threshold adaptation
//! - Revolving‑door adaptive threshold cycling (MAX‑tier)
//! - False Door Deception Layer (MAX‑tier)
//!
//! All upgrades are backwards-compatible.

use kv_web_core::{KvWeb, TokenId, WebNodeId, EdgeKind};
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::time::Instant;

pub type Embedding = Vec<f32>;

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct PolygonId(pub u32);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SemanticPolygon {
    pub id: PolygonId,
    pub centroid: Embedding,
    pub radius: f32,
    pub face_index: u8,
}

/// Dual-layer scratch pad for semantic clustering.
/// Layer A = raw semantic similarity
/// Layer B = polygon geometry bias
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticScratchPad {
    pub layer_a: Vec<f32>,
    pub layer_b: Vec<f32>,
}

/// Zoning + indexing for semantic clusters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticZoning {
    pub root: WebNodeId,
    pub nodes: Vec<WebNodeId>,
    pub index_map: Vec<WebNodeId>,
    pub zones: Vec<SemanticZone>,
    pub scratch: SemanticScratchPad,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticZone {
    pub zone_id: usize,
    pub start: usize,
    pub end: usize,
    pub centroid_node: Option<WebNodeId>,
    pub size: usize,
}

#[derive(Debug, Clone)]
pub struct SemanticOptimizationConfig {
    pub min_similarity_threshold: f32,
    pub max_similarity_threshold: f32,
    pub target_cluster_size: usize,
    pub max_cluster_size: usize,
    pub min_radius: f32,
    pub max_radius: f32,
}

/// Firewall mode + suspicion level.

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FirewallMode {
    Normal,
    Paranoid,
    Adaptive,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SuspicionLevel {
    Allow,
    Suspicious,
    Block,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatEvent {
    pub level: SuspicionLevel,
    pub reason_code: &'static str,
    pub similarity_to_root: f32,
    pub zone_coherence: f32,
    pub polygon_distance: f32,
    pub embedding_variance: f32,
    pub flip_ratio: f32,
}

/// MAX-tier: captured adversarial signature for reverse hardening.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackSignature {
    pub embedding_variance: f32,
    pub spike: f32,
    pub zone_coherence: f32,
    pub root_similarity: f32,
    pub flip_ratio: f32,
    pub polygon_distance: f32,
}

impl AttackSignature {
    pub fn from_event(
        var: f32,
        spike: f32,
        zone_coherence: f32,
        root_similarity: f32,
        flip_ratio: f32,
        polygon_distance: f32,
    ) -> Self {
        Self {
            embedding_variance: var,
            spike,
            zone_coherence,
            root_similarity,
            flip_ratio,
            polygon_distance,
        }
    }

    pub fn as_vec(&self) -> Vec<f32> {
        vec![
            self.embedding_variance,
            self.spike,
            self.zone_coherence,
            self.root_similarity,
            self.flip_ratio,
            self.polygon_distance,
        ]
    }
}

/// MAX-tier: adversarial cluster centroid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdversarialCluster {
    pub centroid: Vec<f32>,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirewallHistory {
    pub signatures: Vec<AttackSignature>,
    pub max_entries: usize,
}

impl FirewallHistory {
    pub fn new(max_entries: usize) -> Self {
        Self {
            signatures: Vec::new(),
            max_entries,
        }
    }

    pub fn record(&mut self, sig: AttackSignature) {
        self.signatures.push(sig);
        if self.signatures.len() > self.max_entries {
            let overflow = self.signatures.len() - self.max_entries;
            self.signatures.drain(0..overflow);
        }
    }

    /// MAX-tier reverse mask: aggregate adversarial pattern into a compact vector.
    pub fn reverse_mask(&self) -> Option<Vec<f32>> {
        if self.signatures.is_empty() {
            return None;
        }

        let mut sum_var = 0.0;
        let mut sum_spike = 0.0;
        let mut sum_zone = 0.0;
        let mut sum_root = 0.0;
        let mut sum_flip = 0.0;
        let mut sum_poly = 0.0;

        let len = self.signatures.len() as f32;

        for s in &self.signatures {
            sum_var += s.embedding_variance;
            sum_spike += s.spike;
            sum_zone += s.zone_coherence;
            sum_root += s.root_similarity;
            sum_flip += s.flip_ratio;
            sum_poly += s.polygon_distance;
        }

        Some(vec![
            sum_var / len,
            sum_spike / len,
            sum_zone / len,
            sum_root / len,
            sum_flip / len,
            sum_poly / len,
        ])
    }

    /// MAX-tier: simple adversarial clustering over signatures.
    pub fn clusters(&self, max_clusters: usize) -> Vec<AdversarialCluster> {
        if self.signatures.is_empty() || max_clusters == 0 {
            return Vec::new();
        }

        let mut clusters: Vec<AdversarialCluster> = Vec::new();

        for sig in &self.signatures {
            let v = sig.as_vec();
            let mut best_idx: Option<usize> = None;
            let mut best_sim = -1.0;

            for (i, c) in clusters.iter().enumerate() {
                let sim = cosine_similarity(&c.centroid, &v);
                if sim > best_sim {
                    best_sim = sim;
                    best_idx = Some(i);
                }
            }

            if let Some(idx) = best_idx {
                if best_sim >= 0.8 {
                    let c = &mut clusters[idx];
                    let count_f = c.count as f32;
                    for (j, val) in v.iter().enumerate() {
                        c.centroid[j] =
                            (c.centroid[j] * count_f + *val) / (count_f + 1.0);
                    }
                    c.count += 1;
                } else if clusters.len() < max_clusters {
                    clusters.push(AdversarialCluster {
                        centroid: v.clone(),
                        count: 1,
                    });
                }
            } else if clusters.len() < max_clusters {
                clusters.push(AdversarialCluster {
                    centroid: v.clone(),
                    count: 1,
                });
            }
        }

        clusters
    }

    /// MAX-tier: zone-aware reverse mask (filter by low zone coherence).
    pub fn zone_reverse_mask(&self, zone_threshold: f32) -> Option<Vec<f32>> {
        let filtered: Vec<&AttackSignature> = self
            .signatures
            .iter()
            .filter(|s| s.zone_coherence < zone_threshold)
            .collect();

        if filtered.is_empty() {
            return None;
        }

        let mut sum = vec![0.0; 6];
        let len = filtered.len() as f32;

        for s in filtered {
            let v = s.as_vec();
            for (i, val) in v.iter().enumerate() {
                sum[i] += *val;
            }
        }

        for v in &mut sum {
            *v /= len;
        }

        Some(sum)
    }
}

/// MAX-tier: revolving-door firewall phase state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RevolvingDoorState {
    pub door_phase: u8,
    pub door_cycle: u64,
    pub door_seed: f32,
}

impl RevolvingDoorState {
    pub fn new() -> Self {
        Self {
            door_phase: 0,
            door_cycle: 0,
            door_seed: 1.0,
        }
    }

    pub fn advance(&mut self) {
        self.door_phase = (self.door_phase + 1) % 4;
        self.door_cycle = self.door_cycle.wrapping_add(1);
        self.door_seed = (self.door_seed * 1.013).fract().max(0.1);
    }
}

// ────────────────────────────────────────────────────────────────
//   FALSE DOOR DECEPTION LAYER (MAX‑TIER)
// ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FalseDoor {
    pub door_id: u32,
    pub bait_vector: Vec<f32>,
    pub phase: u8,
    pub decay: f32,
}

impl FalseDoor {
    pub fn new(door_id: u32, bait_vector: Vec<f32>, phase: u8) -> Self {
        Self {
            door_id,
            bait_vector,
            phase,
            decay: 1.0,
        }
    }

    pub fn update_phase(&mut self, new_phase: u8, seed: f32) {
        self.phase = new_phase;
        self.decay = (self.decay * (1.0 - 0.01 * seed)).max(0.05);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FalseDoorLayer {
    pub doors: Vec<FalseDoor>,
    pub next_id: u32,
}

impl FalseDoorLayer {
    pub fn new() -> Self {
        Self {
            doors: Vec::new(),
            next_id: 1,
        }
    }

    pub fn spawn_false_door(&mut self, centroid: &Embedding, phase: u8) {
        let mut bait = centroid.clone();
        for v in &mut bait {
            *v *= 0.75;
        }
        self.doors.push(FalseDoor::new(self.next_id, bait, phase));
        self.next_id += 1;
    }

    pub fn rotate_doors(&mut self, phase: u8, seed: f32) {
        for d in &mut self.doors {
            d.update_phase(phase, seed);
        }
    }
}

fn false_door_triggered(
    false_doors: &FalseDoorLayer,
    new_embedding: &Embedding,
) -> bool {
    for door in &false_doors.doors {
        let sim = cosine_similarity(&door.bait_vector, new_embedding);
        if sim >= 0.92 * door.decay {
            return true;
        }
    }
    false
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirewallConfig {
    pub mode: FirewallMode,
    pub entropy_min: f32,
    pub entropy_max: f32,
    pub variance_min: f32,
    pub variance_max: f32,
    pub spike_threshold: f32,
    pub zone_similarity_min: f32,
    pub root_similarity_min: f32,
    pub flip_ratio_max: f32,
    pub polygon_distance_factor: f32,
    pub history: FirewallHistory,
    pub revolving: RevolvingDoorState,
    pub false_doors: FalseDoorLayer,
}

impl Default for FirewallConfig {
    fn default() -> Self {
        FirewallConfig {
            mode: FirewallMode::Adaptive,
            entropy_min: 1e-4,
            entropy_max: 10.0,
            variance_min: 1e-4,
            variance_max: 10.0,
            spike_threshold: 0.35,
            zone_similarity_min: 0.20,
            root_similarity_min: 0.05,
            flip_ratio_max: 0.7,
            polygon_distance_factor: 2.0,
            history: FirewallHistory::new(256),
            revolving: RevolvingDoorState::new(),
            false_doors: FalseDoorLayer::new(),
        }
    }
}

/// MAX-tier: revolving-door threshold cycling.
fn revolving_door_step(cfg: &mut FirewallConfig) {
    cfg.revolving.advance();

    let phase = cfg.revolving.door_phase;
    let seed = cfg.revolving.door_seed;

    // Rotate false doors with phase
    cfg.false_doors.rotate_doors(phase, seed);

    match phase {
        0 => {
            cfg.zone_similarity_min =
                (cfg.zone_similarity_min + 0.005 * seed).min(0.6);
            cfg.root_similarity_min =
                (cfg.root_similarity_min + 0.005 * seed).min(0.4);
            cfg.spike_threshold =
                (cfg.spike_threshold * (1.0 + 0.01 * seed)).min(0.5);
        }
        1 => {
            cfg.flip_ratio_max =
                (cfg.flip_ratio_max * (1.0 - 0.02 * seed)).max(0.3);
            cfg.polygon_distance_factor =
                (cfg.polygon_distance_factor * (1.0 + 0.02 * seed)).min(4.0);
        }
        2 => {
            cfg.variance_min =
                (cfg.variance_min * (1.0 + 0.01 * seed)).min(0.5);
            cfg.variance_max =
                (cfg.variance_max * (1.0 - 0.01 * seed)).max(1.0);
        }
        3 => {
            let clusters = cfg.history.clusters(4);
            if !clusters.is_empty() {
                let strongest = clusters
                    .iter()
                    .max_by(|a, b| a.count.cmp(&b.count))
                    .unwrap();
                let v = &strongest.centroid;
                if v.len() >= 6 {
                    let var = v[0];
                    let spike = v[1];
                    let zone = v[2];
                    let root = v[3];
                    let flip = v[4];
                    let poly = v[5];

                    if zone < cfg.zone_similarity_min {
                        cfg.zone_similarity_min =
                            (cfg.zone_similarity_min + 0.01 * seed).min(0.6);
                    }
                    if root < cfg.root_similarity_min {
                        cfg.root_similarity_min =
                            (cfg.root_similarity_min + 0.01 * seed).min(0.4);
                    }
                    if spike > cfg.spike_threshold {
                        cfg.spike_threshold =
                            (cfg.spike_threshold * (1.0 - 0.02 * seed)).max(0.1);
                    }
                    if flip > cfg.flip_ratio_max {
                        cfg.flip_ratio_max =
                            (cfg.flip_ratio_max * (1.0 - 0.02 * seed)).max(0.3);
                    }
                    if poly > cfg.polygon_distance_factor {
                        cfg.polygon_distance_factor =
                            (cfg.polygon_distance_factor * (1.0 + 0.02 * seed)).min(4.0);
                    }
                    if var > cfg.variance_max {
                        cfg.variance_max =
                            (cfg.variance_max * (1.0 - 0.02 * seed)).max(1.0);
                    }
                }
            }
        }
        _ => {}
    }

    cfg.zone_similarity_min = cfg.zone_similarity_min.clamp(0.15, 0.6);
    cfg.root_similarity_min = cfg.root_similarity_min.clamp(0.03, 0.4);
    cfg.spike_threshold = cfg.spike_threshold.clamp(0.1, 0.6);
    cfg.flip_ratio_max = cfg.flip_ratio_max.clamp(0.3, 0.9);
    cfg.polygon_distance_factor = cfg.polygon_distance_factor.clamp(1.0, 4.0);
    cfg.variance_min = cfg.variance_min.clamp(1e-4, 0.5);
    cfg.variance_max = cfg.variance_max.clamp(1.0, 10.0);
}

/// Compute cosine similarity between two embeddings.
fn cosine_similarity(a: &Embedding, b: &Embedding) -> f32 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }

    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }

    dot / (na.sqrt() * nb.sqrt())
}

/// Polygon-aware semantic similarity modifier.
fn polygon_similarity_bias(
    web: &KvWeb,
    a: WebNodeId,
    b: WebNodeId,
    base_sim: f32,
) -> f32 {
    let node_a = match web.nodes.get(&a) {
        Some(n) => n,
        None => return base_sim,
    };
    let node_b = match web.nodes.get(&b) {
        Some(n) => n,
        None => return base_sim,
    };

    let poly_a = match &node_a.polygon {
        Some(p) => p,
        None => return base_sim,
    };
    let poly_b = match &node_b.polygon {
        Some(p) => p,
        None => return base_sim,
    };

    let face_bonus = if poly_a.face_index == poly_b.face_index { 0.15 } else { 0.0 };

    let mut centroid_dist = 0.0;
    for (ca, cb) in poly_a.centroid.iter().zip(poly_b.centroid.iter()) {
        centroid_dist += (ca - cb).abs();
    }

    let centroid_penalty =
        (centroid_dist / (poly_a.radius + poly_b.radius + 1.0)).min(0.25);

    base_sim + face_bonus - centroid_penalty
}

/// Compute centroid embedding for a set of tokens.
fn centroid_embedding(
    tokens: &Vec<TokenId>,
    embeddings: &HashMap<TokenId, Embedding>,
) -> Embedding {
    if tokens.is_empty() {
        return Vec::new();
    }

    let mut dim = 0;
    for t in tokens {
        if let Some(e) = embeddings.get(t) {
            dim = e.len();
            break;
        }
    }

    if dim == 0 {
        return Vec::new();
    }

    let mut sum = vec![0.0; dim];
    let mut count = 0.0;

    for t in tokens {
        if let Some(e) = embeddings.get(t) {
            for (i, v) in e.iter().enumerate() {
                sum[i] += v;
            }
            count += 1.0;
        }
    }

    if count == 0.0 {
        return sum;
    }

    for v in &mut sum {
        *v /= count;
    }

    sum
}

/// Build dual-layer scratch pad for semantic clustering.
fn build_semantic_scratch_pad(
    web: &KvWeb,
    root: WebNodeId,
    nodes: &[WebNodeId],
    sims: &[f32],
) -> SemanticScratchPad {
    let layer_a = sims.to_vec();

    let mut layer_b = Vec::with_capacity(nodes.len());
    for node in nodes {
        let bias = polygon_similarity_bias(web, root, *node, 1.0);
        layer_b.push(bias);
    }

    SemanticScratchPad { layer_a, layer_b }
}

// ────────────────────────────────────────────────────────────────
//   MULTI-ATTACK SEMANTIC FIREWALL (SPIF+)
// ────────────────────────────────────────────────────────────────

fn compute_embedding_stats(new_embedding: &Embedding) -> (f32, f32) {
    let len = new_embedding.len() as f32;
    if len <= 0.0 {
        return (0.0, 0.0);
    }
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for v in new_embedding {
        sum += *v;
        sum_sq += v * v;
    }
    let mean = sum / len;
    let var = (sum_sq / len) - mean * mean;
    (mean, var)
}

fn compute_heatmap_spike(heatmap_layers: Option<&[f32]>) -> f32 {
    if let Some(layers) = heatmap_layers {
        if layers.len() >= 2 {
            let mut spike = 0.0;
            for w in layers.windows(2) {
                spike += (w[0] - w[1]).abs();
            }
            return spike;
        }
    }
    0.0
}

fn compute_zone_coherence(
    web: &KvWeb,
    embeddings: &HashMap<TokenId, Embedding>,
    new_embedding: &Embedding,
    zoning: Option<&SemanticZoning>,
) -> f32 {
    if let Some(z) = zoning {
        let mut best_sim = 0.0;
        for node_id in &z.nodes {
            if let Some(n) = web.nodes.get(node_id) {
                let c = centroid_embedding(&n.tokens, embeddings);
                let sim = cosine_similarity(&c, new_embedding);
                if sim > best_sim {
                    best_sim = sim;
                }
            }
        }
        return best_sim;
    }
    0.0
}

fn compute_root_similarity(
    web: &KvWeb,
    embeddings: &HashMap<TokenId, Embedding>,
    new_embedding: &Embedding,
    zoning: Option<&SemanticZoning>,
) -> f32 {
    if let Some(z) = zoning {
        if let Some(root_node) = web.nodes.get(&z.root) {
            let root_centroid = centroid_embedding(&root_node.tokens, embeddings);
            return cosine_similarity(&root_centroid, new_embedding);
        }
    }
    0.0
}

fn compute_flip_ratio(
    web: &KvWeb,
    embeddings: &HashMap<TokenId, Embedding>,
    new_embedding: &Embedding,
    zoning: Option<&SemanticZoning>,
) -> f32 {
    if let Some(z) = zoning {
        if let Some(root_node) = web.nodes.get(&z.root) {
            let root_centroid = centroid_embedding(&root_node.tokens, embeddings);
            let mut flips = 0;
            let mut total = 0;
            for (r, v) in root_centroid.iter().zip(new_embedding.iter()) {
                if *r != 0.0 && *v != 0.0 {
                    let rs = r.is_sign_negative();
                    let vs = v.is_sign_negative();
                    if rs != vs {
                        flips += 1;
                    }
                    total += 1;
                }
            }
            if total > 0 {
                return flips as f32 / total as f32;
            }
        }
    }
    0.0
}

fn compute_polygon_distance(
    polygon_region: Option<&kv_web_core::PolygonRegion>,
    new_embedding: &Embedding,
) -> f32 {
    if let Some(poly) = polygon_region {
        let mut dist = 0.0;
        for (a, b) in poly.centroid.iter().zip(new_embedding.iter()) {
            dist += (a - b).abs();
        }
        return dist;
    }
    0.0
}

fn evaluate_threat(
    cfg: &FirewallConfig,
    _mean: f32,
    var: f32,
    spike: f32,
    zone_coherence: f32,
    root_similarity: f32,
    flip_ratio: f32,
    polygon_distance: f32,
) -> ThreatEvent {
    let mut level = SuspicionLevel::Allow;
    let mut reason = "NONE";

    if var < cfg.variance_min || var > cfg.variance_max {
        level = SuspicionLevel::Suspicious;
        reason = "EMBEDDING_VARIANCE_SUSPECT";
    }

    if spike > cfg.spike_threshold {
        level = SuspicionLevel::Suspicious;
        reason = "HEATMAP_SPIKE";
    }

    if zone_coherence < cfg.zone_similarity_min {
        level = SuspicionLevel::Block;
        reason = "ZONE_COHERENCE_LOW";
    }

    if root_similarity < cfg.root_similarity_min && zone_coherence >= cfg.zone_similarity_min {
        level = SuspicionLevel::Suspicious;
        reason = "ROOT_SIMILARITY_LOW";
    }

    if flip_ratio > cfg.flip_ratio_max {
        level = SuspicionLevel::Block;
        reason = "POLARITY_FLIP_HIGH";
    }

    if polygon_distance > cfg.polygon_distance_factor * var.max(1.0) {
        level = SuspicionLevel::Block;
        reason = "GEOMETRY_BREAK";
    }

    ThreatEvent {
        level,
        reason_code: reason,
        similarity_to_root: root_similarity,
        zone_coherence,
        polygon_distance,
        embedding_variance: var,
        flip_ratio,
    }
}

fn adapt_firewall_config(cfg: &mut FirewallConfig, event: &ThreatEvent) {
    match cfg.mode {
        FirewallMode::Adaptive => {
            match event.level {
                SuspicionLevel::Block => {
                    cfg.zone_similarity_min = (cfg.zone_similarity_min + 0.01).min(0.5);
                    cfg.root_similarity_min = (cfg.root_similarity_min + 0.01).min(0.3);
                    cfg.spike_threshold = (cfg.spike_threshold * 0.95).max(0.1);
                }
                SuspicionLevel::Suspicious => {
                    cfg.zone_similarity_min = (cfg.zone_similarity_min + 0.005).min(0.4);
                    cfg.root_similarity_min = (cfg.root_similarity_min + 0.005).min(0.25);
                }
                SuspicionLevel::Allow => {
                    cfg.zone_similarity_min = (cfg.zone_similarity_min * 0.999).max(0.15);
                    cfg.root_similarity_min = (cfg.root_similarity_min * 0.999).max(0.03);
                }
            }
        }
        FirewallMode::Paranoid => {
            cfg.zone_similarity_min = (cfg.zone_similarity_min + 0.01).min(0.6);
            cfg.root_similarity_min = (cfg.root_similarity_min + 0.01).min(0.4);
            cfg.spike_threshold = (cfg.spike_threshold * 0.95).max(0.1);
        }
        FirewallMode::Normal => {}
    }
}

fn record_attack_signature(
    cfg: &mut FirewallConfig,
    var: f32,
    spike: f32,
    zone_coherence: f32,
    root_similarity: f32,
    flip_ratio: f32,
    polygon_distance: f32,
) {
    let sig = AttackSignature::from_event(
        var,
        spike,
        zone_coherence,
        root_similarity,
        flip_ratio,
        polygon_distance,
    );
    cfg.history.record(sig);
}

pub fn firewall_reverse_mask(cfg: &FirewallConfig) -> Option<Vec<f32>> {
    cfg.history.reverse_mask()
}

pub fn firewall_zone_reverse_mask(cfg: &FirewallConfig, zone_threshold: f32) -> Option<Vec<f32>> {
    cfg.history.zone_reverse_mask(zone_threshold)
}

pub fn firewall_adversarial_clusters(
    cfg: &FirewallConfig,
    max_clusters: usize,
) -> Vec<AdversarialCluster> {
    cfg.history.clusters(max_clusters)
}

fn apply_reverse_mask(cfg: &mut FirewallConfig, mask: &[f32]) {
    if mask.len() < 6 {
        return;
    }

    let var = mask[0];
    let spike = mask[1];
    let zone = mask[2];
    let root = mask[3];
    let flip = mask[4];
    let poly = mask[5];

    if var > cfg.variance_max {
        cfg.variance_max = (cfg.variance_max * 0.9).max(1.0);
    }
    if spike > cfg.spike_threshold {
        cfg.spike_threshold = (cfg.spike_threshold * 0.9).max(0.1);
    }
    if zone < cfg.zone_similarity_min {
        cfg.zone_similarity_min = (cfg.zone_similarity_min + 0.01).min(0.6);
    }
    if root < cfg.root_similarity_min {
        cfg.root_similarity_min = (cfg.root_similarity_min + 0.01).min(0.4);
    }
    if flip > cfg.flip_ratio_max {
        cfg.flip_ratio_max = (cfg.flip_ratio_max * 0.95).max(0.3);
    }
    if poly > cfg.polygon_distance_factor {
        cfg.polygon_distance_factor = (cfg.polygon_distance_factor * 1.05).min(4.0);
    }
}

fn spif_detect(
    web: &KvWeb,
    embeddings: &HashMap<TokenId, Embedding>,
    new_embedding: &Embedding,
    polygon_region: Option<&kv_web_core::PolygonRegion>,
    zoning: Option<&SemanticZoning>,
    heatmap_layers: Option<&[f32]>,
) -> bool {
    let mut cfg = FirewallConfig::default();

    // False door check first
    if false_door_triggered(&cfg.false_doors, new_embedding) {
        let (_, var) = compute_embedding_stats(new_embedding);
        record_attack_signature(
            &mut cfg,
            var,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        );
        return true;
    }

    let (mean, var) = compute_embedding_stats(new_embedding);
    let spike = compute_heatmap_spike(heatmap_layers);
    let zone_coherence = compute_zone_coherence(web, embeddings, new_embedding, zoning);
    let root_similarity = compute_root_similarity(web, embeddings, new_embedding, zoning);
    let flip_ratio = compute_flip_ratio(web, embeddings, new_embedding, zoning);
    let polygon_distance = compute_polygon_distance(polygon_region, new_embedding);

    let event = evaluate_threat(
        &cfg,
        mean,
        var,
        spike,
        zone_coherence,
        root_similarity,
        flip_ratio,
        polygon_distance,
    );

    record_attack_signature(
        &mut cfg,
        var,
        spike,
        zone_coherence,
        root_similarity,
        flip_ratio,
        polygon_distance,
    );

    if let Some(mask) = cfg.history.reverse_mask() {
        apply_reverse_mask(&mut cfg, &mask);
    }

    if let Some(zone_mask) = cfg.history.zone_reverse_mask(cfg.zone_similarity_min) {
        apply_reverse_mask(&mut cfg, &zone_mask);
    }

    revolving_door_step(&mut cfg);

    adapt_firewall_config(&mut cfg, &event);

    match event.level {
        SuspicionLevel::Allow => false,
        SuspicionLevel::Suspicious => false,
        SuspicionLevel::Block => true,
    }
}

/// Extension methods for semantic clustering on KvWeb.
pub trait KvWebSemantic {
    fn cluster_tokens_parallel(
        &mut self,
        embeddings: &HashMap<TokenId, Embedding>,
        similarity_threshold: f32,
    ) -> Vec<WebNodeId>;

    fn link_semantic_nodes_parallel(
        &mut self,
        embeddings: &HashMap<TokenId, Embedding>,
        similarity_threshold: f32,
    );

    fn build_polygonal_semantic_regions(
        &mut self,
        embeddings: &HashMap<TokenId, Embedding>,
        similarity_threshold: f32,
    ) -> Vec<SemanticPolygon>;

    fn semantic_index_and_zone(
        &mut self,
        embeddings: &HashMap<TokenId, Embedding>,
        root: WebNodeId,
        threshold: f32,
        num_zones: usize,
    ) -> SemanticZoning;

    fn semantic_index_and_zone_compressed(
        &mut self,
        embeddings: &HashMap<TokenId, Embedding>,
        root: WebNodeId,
        threshold: f32,
        num_zones: usize,
    ) -> Option<Vec<u8>>;

    fn optimize_semantic(
        &mut self,
        embeddings: &HashMap<TokenId, Embedding>,
        similarity_threshold: &mut f32,
        cfg: &SemanticOptimizationConfig,
    ) -> Option<Vec<u8>>;
}

impl KvWebSemantic for KvWeb {

    fn cluster_tokens_parallel(
        &mut self,
        embeddings: &HashMap<TokenId, Embedding>,
        similarity_threshold: f32,
    ) -> Vec<WebNodeId> {

        let tokens: Vec<TokenId> = embeddings.keys().cloned().collect();
        let mut clusters: Vec<Vec<TokenId>> = Vec::new();

        for token in tokens {
            let emb = &embeddings[&token];

            if spif_detect(self, embeddings, emb, None, None, None) {
                continue;
            }

            let mut placed = false;

            for cluster in &mut clusters {
                let centroid = centroid_embedding(cluster, embeddings);
                let sim = cosine_similarity(&centroid, emb);

                if sim >= similarity_threshold {
                    cluster.push(token);
                    placed = true;
                    break;
                }
            }

            if !placed {
                clusters.push(vec![token]);
            }
        }

        let mut node_ids = Vec::new();
        for cluster in clusters {
            if cluster.is_empty() {
                continue;
            }

            let label = format!("cluster_{}", cluster.len());
            let score = cluster.len() as f32;

            let id = self.create_node(cluster, Some(label), score);
            node_ids.push(id);
        }

        node_ids
    }

    fn link_semantic_nodes_parallel(
        &mut self,
        embeddings: &HashMap<TokenId, Embedding>,
        similarity_threshold: f32,
    ) {
        let node_ids: Vec<WebNodeId> = self.nodes.keys().cloned().collect();

        let candidate_edges: Vec<(WebNodeId, WebNodeId, f32)> =
            node_ids.par_iter().enumerate().flat_map(|(i, a_id)| {
                let mut edges = Vec::new();

                let a_node = match self.nodes.get(a_id) {
                    Some(n) => n,
                    None => return edges,
                };

                let a_centroid = centroid_embedding(&a_node.tokens, embeddings);

                if spif_detect(self, embeddings, &a_centroid, a_node.polygon.as_ref(), None, None) {
                    return edges;
                }

                for b_id in node_ids.iter().skip(i + 1) {
                    let b_node = match self.nodes.get(b_id) {
                        Some(n) => n,
                        None => continue,
                    };

                    let b_centroid = centroid_embedding(&b_node.tokens, embeddings);

                    if spif_detect(self, embeddings, &b_centroid, b_node.polygon.as_ref(), None, None) {
                        continue;
                    }

                    let base_sim = cosine_similarity(&a_centroid, &b_centroid);
                    let sim = polygon_similarity_bias(self, *a_id, *b_id, base_sim);

                    if sim >= similarity_threshold {
                        edges.push((*a_id, *b_id, sim));
                    }
                }

                edges
            }).collect();

        for (a, b, sim) in candidate_edges {
            self.add_edge(a, b, sim, EdgeKind::Semantic);
            self.add_edge(b, a, sim, EdgeKind::Semantic);
        }
    }

    fn build_polygonal_semantic_regions(
        &mut self,
        embeddings: &HashMap<TokenId, Embedding>,
        similarity_threshold: f32,
    ) -> Vec<SemanticPolygon> {

        let node_ids: Vec<WebNodeId> = self.nodes.keys().cloned().collect();
        let mut polygons = Vec::new();
        let mut next_id = 1;

        for node_id in &node_ids {
            let node = match self.nodes.get(node_id) {
                Some(n) => n,
                None => continue,
            };

            let centroid = centroid_embedding(&node.tokens, embeddings);
            if centroid.is_empty() {
                continue;
            }

            if spif_detect(self, embeddings, &centroid, node.polygon.as_ref(), None, None) {
                continue;
            }

            let mut radius_acc = 0.0;
            let mut radius_count = 0.0;

            for other_id in &node_ids {
                if other_id == node_id {
                    continue;
                }

                let other = match self.nodes.get(other_id) {
                    Some(n) => n,
                    None => continue,
                };

                let other_centroid = centroid_embedding(&other.tokens, embeddings);
                if other_centroid.is_empty() {
                    continue;
                }

                let sim = cosine_similarity(&centroid, &other_centroid);
                if sim >= similarity_threshold {
                    radius_acc += sim;
                    radius_count += 1.0;
                }
            }

            let radius = if radius_count > 0.0 {
                radius_acc / radius_count
            } else {
                similarity_threshold
            };

            let face_index = if radius >= 0.9 {
                3
            } else if radius >= 0.7 {
                2
            } else if radius >= 0.5 {
                1
            } else {
                0
            };

            polygons.push(SemanticPolygon {
                id: PolygonId(next_id),
                centroid,
                radius,
                face_index,
            });

            next_id += 1;
        }

        polygons
    }

    fn semantic_index_and_zone(
        &mut self,
        embeddings: &HashMap<TokenId, Embedding>,
        root: WebNodeId,
        threshold: f32,
        num_zones: usize,
    ) -> SemanticZoning {

        let root_node = self.nodes.get(&root).unwrap();
        let root_centroid = centroid_embedding(&root_node.tokens, embeddings);

        let mut nodes = Vec::new();
        let mut sims = Vec::new();

        for (id, node) in &self.nodes {
            let centroid = centroid_embedding(&node.tokens, embeddings);
            let base = cosine_similarity(&root_centroid, &centroid);
            let sim = polygon_similarity_bias(self, root, *id, base);

            if spif_detect(self, embeddings, &centroid, node.polygon.as_ref(), None, None) {
                continue;
            }

            if sim >= threshold {
                nodes.push(*id);
                sims.push(sim);
            }
        }

        let mut index_map = nodes.clone();
        index_map.sort_by(|a, b| {
            let ia = nodes.iter().position(|x| x == a).unwrap();
            let ib = nodes.iter().position(|x| x == b).unwrap();
            let sa = sims[ia];
            let sb = sims[ib];
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });

        let num_zones = num_zones.max(1);
        let len = index_map.len();
        let zone_size = (len as f32 / num_zones as f32).ceil() as usize;

        let mut zones = Vec::new();
        let mut zone_id = 0;
        let mut start = 0;

        while start < len {
            let end = (start + zone_size).min(len);
            let slice = &index_map[start..end];

            let centroid_node = slice.get(slice.len() / 2).cloned();

            zones.push(SemanticZone {
                zone_id,
                start,
                end,
                centroid_node,
                size: slice.len(),
            });

            zone_id += 1;
            start = end;
        }

        let scratch = build_semantic_scratch_pad(self, root, &nodes, &sims);

        SemanticZoning {
            root,
            nodes,
            index_map,
            zones,
            scratch,
        }
    }

    fn semantic_index_and_zone_compressed(
        &mut self,
        embeddings: &HashMap<TokenId, Embedding>,
        root: WebNodeId,
        threshold: f32,
        num_zones: usize,
    ) -> Option<Vec<u8>> {

        let sz = self.semantic_index_and_zone(embeddings, root, threshold, num_zones);

        self.compressor.as_ref().map(|c| {
            c.compress(&(
                "semantic_index_and_zone",
                root,
                threshold,
                num_zones,
                &sz.nodes,
                &sz.index_map,
                &sz.zones,
                &sz.scratch.layer_a,
                &sz.scratch.layer_b,
            ))
        })
    }

    fn optimize_semantic(
        &mut self,
        embeddings: &HashMap<TokenId, Embedding>,
        similarity_threshold: &mut f32,
        cfg: &SemanticOptimizationConfig,
    ) -> Option<Vec<u8>> {

        let node_ids: Vec<WebNodeId> = self.nodes.keys().cloned().collect();
        if node_ids.is_empty() {
            return None;
        }

        let mut cluster_sizes = Vec::new();
        for (_, node) in &self.nodes {
            cluster_sizes.push(node.tokens.len());
        }

        let avg_cluster_size = if cluster_sizes.is_empty() {
            0.0
        } else {
            cluster_sizes.iter().sum::<usize>() as f32 / cluster_sizes.len() as f32
        };

        if avg_cluster_size < cfg.target_cluster_size as f32 {
            *similarity_threshold =
                (*similarity_threshold * 0.95).max(cfg.min_similarity_threshold);
        } else if avg_cluster_size > cfg.max_cluster_size as f32 {
            *similarity_threshold =
                (*similarity_threshold * 1.05).min(cfg.max_similarity_threshold);
        }

        let mut radii = Vec::new();
        for (_, node) in &self.nodes {
            if let Some(poly) = &node.polygon {
                radii.push(poly.radius);
            }
        }

        let avg_radius = if radii.is_empty() {
            0.0
        } else {
            radii.iter().sum::<f32>() / radii.len() as f32
        };

        if avg_radius < cfg.min_radius {
            *similarity_threshold =
                (*similarity_threshold * 0.95).max(cfg.min_similarity_threshold);
        } else if avg_radius > cfg.max_radius {
            *similarity_threshold =
                (*similarity_threshold * 1.05).min(cfg.max_similarity_threshold);
        }

        self.compressor.as_ref().map(|c| {
            c.compress(&(
                "optimize_semantic",
                avg_cluster_size,
                avg_radius,
                *similarity_threshold,
            ))
        })
    }
}

// ────────────────────────────────────────────────────────────────
//   SEMANTIC ROUNDABOUT MAX‑TIER UPGRADE
// ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticPriority {
    High,
    Standard,
    Low,
}

#[derive(Debug, Clone)]
pub struct SemanticPacket {
    pub id: u64,
    pub priority: SemanticPriority,
    pub root: WebNodeId,
    pub created_at: Instant,
    pub last_attempt: Instant,
    pub hops: u32,
    pub last_exit_zone: Option<usize>,
    pub route_score: f32,
    pub stability_factor: f32,
}

impl SemanticPacket {
    pub fn new(id: u64, root: WebNodeId, priority: SemanticPriority) -> Self {
        let now = Instant::now();
        Self {
            id,
            priority,
            root,
            created_at: now,
            last_attempt: now,
            hops: 0,
            last_exit_zone: None,
            route_score: 0.0,
            stability_factor: 1.0,
        }
    }

    pub fn escalate(&mut self) {
        self.priority = match self.priority {
            SemanticPriority::High => SemanticPriority::High,
            SemanticPriority::Standard => SemanticPriority::High,
            SemanticPriority::Low => SemanticPriority::Standard,
        };
        self.stability_factor = (self.stability_factor + 0.05).min(2.0);
    }

    pub fn reinforce(&mut self, success: bool) {
        if success {
            self.stability_factor = (self.stability_factor + 0.05).min(2.0);
        } else {
            self.stability_factor = (self.stability_factor - 0.05).max(0.1);
        }
    }
}

#[derive(Debug, Clone)]
pub struct SemanticRoundaboutConfig {
    pub max_hops_before_escalation: u32,
    pub max_age_before_force_exit_ms: u64,
}

impl Default for SemanticRoundaboutConfig {
    fn default() -> Self {
        Self {
            max_hops_before_escalation: 8,
            max_age_before_force_exit_ms: 250,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SemanticExitCandidate {
    pub zone_id: usize,
    pub node_id: WebNodeId,
    pub score: f32,
}

#[derive(Debug)]
pub enum SemanticRouteDecision {
    Circulate(SemanticPacket),
    Exit {
        packet: SemanticPacket,
        zone_id: usize,
        node_id: WebNodeId,
    },
}

pub trait KvWebSemanticRoundabout {
    fn route_semantic_packet(
        &mut self,
        embeddings: &HashMap<TokenId, Embedding>,
        zoning: &SemanticZoning,
        cfg: &SemanticRoundaboutConfig,
        packet: SemanticPacket,
    ) -> SemanticRouteDecision;
}

impl KvWebSemanticRoundabout for KvWeb {
    fn route_semantic_packet(
        &mut self,
        embeddings: &HashMap<TokenId, Embedding>,
        zoning: &SemanticZoning,
        cfg: &SemanticRoundaboutConfig,
        packet: SemanticPacket,
    ) -> SemanticRouteDecision {
        let mut packet = packet;
        let now = Instant::now();
        packet.hops += 1;
        packet.last_attempt = now;

        let age_ms = now.duration_since(packet.created_at).as_millis() as u64;
        if age_ms >= cfg.max_age_before_force_exit_ms
            || packet.hops >= cfg.max_hops_before_escalation
        {
            packet.escalate();
        }

        let mut candidates: Vec<SemanticExitCandidate> = Vec::new();

        for zone in &zoning.zones {
            let zone_id = zone.zone_id;
            let slice = &zoning.index_map[zone.start..zone.end];

            for node_id in slice {
                if let Some(node) = self.nodes.get(node_id) {
                    let centroid = centroid_embedding(&node.tokens, embeddings);
                    if centroid.is_empty() {
                        continue;
                    }

                    if spif_detect(
                        self,
                        embeddings,
                        &centroid,
                        node.polygon.as_ref(),
                        Some(zoning),
                        Some(&zoning.scratch.layer_a),
                    ) {
                        continue;
                    }

                    let base_sim = cosine_similarity(&centroid, &centroid);
                    let bias = polygon_similarity_bias(self, zoning.root, *node_id, base_sim);

                    let scratch_idx = zoning.nodes.iter().position(|n| n == node_id);
                    let scratch_sim = scratch_idx
                        .map(|i| zoning.scratch.layer_a[i])
                        .unwrap_or(0.0);

                    let score = bias * 0.5 + scratch_sim * 0.5;

                    candidates.push(SemanticExitCandidate {
                        zone_id,
                        node_id: *node_id,
                        score,
                    });
                }
            }
        }

        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut chosen: Option<SemanticExitCandidate> = None;

        if let Some(hint_zone) = packet.last_exit_zone {
            if let Some(hinted) = candidates.iter().find(|c| c.zone_id == hint_zone) {
                chosen = Some(hinted.clone());
            }
        }

        if chosen.is_none() {
            chosen = candidates.first().cloned();
        }

        match chosen {
            None => {
                packet.reinforce(false);
                SemanticRouteDecision::Circulate(packet)
            }
            Some(exit) => {
                packet.route_score = exit.score;
                packet.last_exit_zone = Some(exit.zone_id);
                packet.reinforce(true);

                SemanticRouteDecision::Exit {
                    packet,
                    zone_id: exit.zone_id,
                    node_id: exit.node_id,
                }
            }
        }
    }
}
