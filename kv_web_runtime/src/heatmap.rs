//! heatmap.rs
//!
//! Token-level heatmaps for KV Web + BitDrop_v2 max-tier compression
//! + Polygonal-KV geometry upgrade.
//!
//! Adds:
//! - polygon-weighted heatmaps
//! - face-index semantic bias
//! - centroid-distance penalty
//! - radius-aware smoothing
//! - multi-layer heatmap
//! - multi-layer scratch pad
//! - per-layer index maps
//! - zoning (semantic regions)
//! - performance metrics (timing)
//! - auto-optimizer over zones + layers
//!
//! All upgrades are backwards-compatible.

use kv_web_core::KvWeb;
use std::time::Instant;
use serde::Serialize;

/// Optimization config for heatmaps.
#[derive(Debug, Clone)]
pub struct HeatmapOptimizationConfig {
    pub min_smoothing_strength: f32,
    pub max_smoothing_strength: f32,
    pub target_heat_variance: f32,
    pub max_heat_variance: f32,
}

/// Per-zone statistics.
#[derive(Debug, Clone, Serialize)]
pub struct ZoneStats {
    pub zone_id: u32,
    pub size: usize,
    pub avg_heat: f32,
    pub max_heat: f32,
    pub min_heat: f32,
    pub centroid_idx: usize,
}

/// Multi-layer heatmap + scratch pad + index maps + zoning + performance metrics.
#[derive(Debug, Clone)]
pub struct MultiLayerHeatmap {
    pub layers: Vec<Vec<f32>>,        // heat values per layer
    pub scratch: Vec<Vec<f32>>,       // scratch pad per layer
    pub index_maps: Vec<Vec<usize>>,  // token index mapping per layer

    // Zoning
    pub zone_maps: Vec<Vec<u32>>,             // zone ID per token per layer
    pub zone_ranges: Vec<Vec<(usize, usize)>>, // (start, end) ranges per zone per layer
    pub zone_stats: Vec<Vec<ZoneStats>>,      // metrics per zone per layer

    // Performance metrics
    pub build_time_ms: u128,
    pub layer_times_ms: Vec<u128>,
    pub index_times_ms: Vec<u128>,
    pub smoothing_times_ms: Vec<u128>,
    pub zoning_times_ms: Vec<u128>,
}

/// Polygon-aware score modifier.
fn polygon_score_bias(web: &KvWeb, node_score: f32, node_id: kv_web_core::WebNodeId) -> f32 {
    let node = match web.nodes.get(&node_id) {
        Some(n) => n,
        None => return node_score,
    };

    let poly = match &node.polygon {
        Some(p) => p,
        None => return node_score,
    };

    let face_bonus = match poly.face_index {
        3 => 0.20,
        2 => 0.10,
        1 => 0.05,
        _ => 0.0,
    };

    let mut centroid_mag = 0.0;
    for v in &poly.centroid {
        centroid_mag += v.abs();
    }

    let centroid_penalty = f32::min(centroid_mag / (poly.radius + 1.0), 0.25);

    node_score + face_bonus - centroid_penalty
}

/// Build a polygon-aware heatmap over tokens.
pub fn token_score_heatmap(web: &KvWeb, kv_len: usize) -> Vec<f32> {
    let mut heat = vec![0.0; kv_len];

    for (id, node) in &web.nodes {
        for t in &node.tokens {
            let idx = t.0;
            if idx < kv_len {
                let base = node.score;
                let biased = polygon_score_bias(web, base, *id);
                heat[idx] = biased;
            }
        }
    }

    heat
}

/// Normalize a heatmap to [0, 1].
pub fn normalize_heatmap(heat: &mut [f32]) {
    let mut max = 0.0;
    for &v in heat.iter() {
        if v > max {
            max = v;
        }
    }
    if max <= 0.0 {
        return;
    }
    for v in heat.iter_mut() {
        *v /= max;
    }
}

/// Optional smoothing pass (Gaussian-lite).
pub fn smooth_heatmap(heat: &mut [f32]) {
    if heat.len() < 3 {
        return;
    }

    let mut out = heat.to_vec();
    for i in 1..heat.len() - 1 {
        out[i] = (heat[i - 1] + heat[i] * 2.0 + heat[i + 1]) / 4.0;
    }
    heat.copy_from_slice(&out);
}

/// Build zoning for a single layer: zone_map, ranges, stats.
fn build_zones_for_layer(layer: &[f32]) -> (Vec<u32>, Vec<(usize, usize)>, Vec<ZoneStats>) {
    let mut zone_map = vec![0u32; layer.len()];
    let mut zones: Vec<(usize, usize)> = Vec::new();
    let mut stats: Vec<ZoneStats> = Vec::new();

    if layer.is_empty() {
        return (zone_map, zones, stats);
    }

    let mut current_zone = 0u32;
    let mut start = 0usize;
    zone_map[0] = current_zone;

    // Simple zoning: threshold-based segmentation on heat difference.
    for i in 1..layer.len() {
        let prev = layer[i - 1];
        let curr = layer[i];

        if (prev - curr).abs() > 0.15 {
            zones.push((start, i - 1));
            start = i;
            current_zone += 1;
        }

        zone_map[i] = current_zone;
    }

    zones.push((start, layer.len() - 1));

    // Build zone stats.
    for (zone_id, (s, e)) in zones.iter().enumerate() {
        let slice = &layer[*s..=*e];

        let size = slice.len();
        let avg_heat = slice.iter().sum::<f32>() / size as f32;
        let max_heat = slice.iter().cloned().fold(f32::MIN, f32::max);
        let min_heat = slice.iter().cloned().fold(f32::MAX, f32::min);

        let mut centroid_idx = *s;
        let mut best = f32::MIN;
        for i in *s..=*e {
            if layer[i] > best {
                best = layer[i];
                centroid_idx = i;
            }
        }

        stats.push(ZoneStats {
            zone_id: zone_id as u32,
            size,
            avg_heat,
            max_heat,
            min_heat,
            centroid_idx,
        });
    }

    (zone_map, zones, stats)
}

/// Build a multi-layer heatmap with scratch pads, index maps, zoning, and performance metrics.
pub fn build_multi_layer_heatmap(
    web: &KvWeb,
    kv_len: usize,
    num_layers: usize,
    base_smoothing_strength: f32,
) -> MultiLayerHeatmap {
    let num_layers = num_layers.max(1);

    let build_start = Instant::now();

    let mut layer_times_ms = Vec::with_capacity(num_layers);
    let mut index_times_ms = Vec::with_capacity(num_layers);
    let mut smoothing_times_ms = Vec::with_capacity(num_layers);
    let mut zoning_times_ms = Vec::with_capacity(num_layers);

    // Base layer timing
    let layer_start = Instant::now();
    let mut base = token_score_heatmap(web, kv_len);
    normalize_heatmap(&mut base);
    layer_times_ms.push(layer_start.elapsed().as_millis());

    let mut layers = Vec::with_capacity(num_layers);
    let mut scratch = Vec::with_capacity(num_layers);
    let mut index_maps = Vec::with_capacity(num_layers);
    let mut zone_maps = Vec::with_capacity(num_layers);
    let mut zone_ranges = Vec::with_capacity(num_layers);
    let mut zone_stats = Vec::with_capacity(num_layers);

    layers.push(base.clone());
    scratch.push(vec![0.0; kv_len]);

    // Index map timing
    let idx_start = Instant::now();
    index_maps.push((0..kv_len).collect::<Vec<usize>>());
    index_times_ms.push(idx_start.elapsed().as_millis());

    // Zoning timing for base layer
    let zone_start = Instant::now();
    let (zmap0, zranges0, zstats0) = build_zones_for_layer(&base);
    zone_maps.push(zmap0);
    zone_ranges.push(zranges0);
    zone_stats.push(zstats0);
    zoning_times_ms.push(zone_start.elapsed().as_millis());

    let mut current = base;

    // Additional layers
    for layer_idx in 1..num_layers {
        let layer_start = Instant::now();
        let mut next = current.clone();

        let passes = (base_smoothing_strength * layer_idx as f32)
            .round()
            .max(1.0) as usize;

        let smoothing_start = Instant::now();
        for _ in 0..passes {
            smooth_heatmap(&mut next);
        }
        smoothing_times_ms.push(smoothing_start.elapsed().as_millis());

        normalize_heatmap(&mut next);
        layer_times_ms.push(layer_start.elapsed().as_millis());

        layers.push(next.clone());
        scratch.push(vec![0.0; kv_len]);

        // Index map timing
        let idx_start = Instant::now();
        let mut idx_map: Vec<usize> = (0..kv_len).collect();
        idx_map.sort_by(|a, b| next[*b].partial_cmp(&next[*a]).unwrap_or(std::cmp::Ordering::Equal));
        index_maps.push(idx_map);
        index_times_ms.push(idx_start.elapsed().as_millis());

        // Zoning timing
        let zone_start = Instant::now();
        let (zmap, zranges, zstats) = build_zones_for_layer(&next);
        zone_maps.push(zmap);
        zone_ranges.push(zranges);
        zone_stats.push(zstats);
        zoning_times_ms.push(zone_start.elapsed().as_millis());

        current = next;
    }

    MultiLayerHeatmap {
        layers,
        scratch,
        index_maps,
        zone_maps,
        zone_ranges,
        zone_stats,
        build_time_ms: build_start.elapsed().as_millis(),
        layer_times_ms,
        index_times_ms,
        smoothing_times_ms,
        zoning_times_ms,
    }
}

/// Max-tier optimization loop for heatmaps.
pub fn optimize_heatmap(
    web: &KvWeb,
    kv_len: usize,
    smoothing_strength: &mut f32,
    cfg: &HeatmapOptimizationConfig,
) -> Option<Vec<u8>> {
    let heat = token_score_heatmap(web, kv_len);

    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut count = 0.0;

    for v in &heat {
        sum += *v;
        sum_sq += v * v;
        count += 1.0;
    }

    let mean = sum / count;
    let variance = (sum_sq / count) - (mean * mean);

    if variance < cfg.target_heat_variance {
        *smoothing_strength =
            (*smoothing_strength * 1.05).min(cfg.max_smoothing_strength);
    } else if variance > cfg.max_heat_variance {
        *smoothing_strength =
            (*smoothing_strength * 0.9).max(cfg.min_smoothing_strength);
    }

    web.compressor.as_ref().map(|c| {
        c.compress(&(
            "optimize_heatmap",
            kv_len,
            variance,
            *smoothing_strength,
        ))
    })
}

/// Auto-optimizer over multi-layer heatmap + zoning.
/// Uses top-layer variance + zone stats to tune smoothing strength.
pub fn optimize_multi_layer_heatmap(
    web: &KvWeb,
    kv_len: usize,
    base_smoothing_strength: &mut f32,
    cfg: &HeatmapOptimizationConfig,
    num_layers: usize,
) -> Option<Vec<u8>> {
    let ml = build_multi_layer_heatmap(web, kv_len, num_layers, *base_smoothing_strength);

    let mut layer_variances = Vec::with_capacity(ml.layers.len());
    for layer in &ml.layers {
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut count = 0.0;

        for v in layer {
            sum += *v;
            sum_sq += v * v;
            count += 1.0;
        }

        let mean = sum / count;
        let variance = (sum_sq / count) - (mean * mean);
        layer_variances.push(variance);
    }

    // Auto-optimizer: use top-layer variance + zone stats to tune smoothing.
    if let Some(&top_var) = layer_variances.last() {
        if top_var < cfg.target_heat_variance {
            *base_smoothing_strength =
                (*base_smoothing_strength * 1.05).min(cfg.max_smoothing_strength);
        } else if top_var > cfg.max_heat_variance {
            *base_smoothing_strength =
                (*base_smoothing_strength * 0.9).max(cfg.min_smoothing_strength);
        }
    }

    // Optional: derive a simple zone complexity score from top layer zones.
    let mut zone_complexity = 0.0f32;
    if let Some(top_zones) = ml.zone_stats.last() {
        for zs in top_zones {
            zone_complexity += zs.avg_heat * zs.size as f32;
        }
    }

    web.compressor.as_ref().map(|c| {
        c.compress(&(
            "optimize_multi_layer_heatmap",
            kv_len,
            layer_variances,
            *base_smoothing_strength,
            ml.layers.len(),
            ml.index_maps.clone(),
            ml.build_time_ms,
            ml.layer_times_ms,
            ml.index_times_ms,
            ml.smoothing_times_ms,
            ml.zoning_times_ms,
            ml.zone_stats.clone(),
            zone_complexity,
        ))
    })
}

