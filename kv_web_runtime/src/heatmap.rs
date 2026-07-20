//! heatmap.rs
//!
//! Token-level heatmaps for KV Web + BitDrop_v2 max-tier compression.
//! Useful for visualizing relevance, drift, pruning effects,
//! semantic clustering, and transformer mask weighting.

use kv_web_core::{KvWeb, KvWebCompressor};

/// Build a simple heatmap over tokens based on node scores.
/// Each token gets the score of its node (or 0.0 if unassigned).
pub fn token_score_heatmap(web: &KvWeb, kv_len: usize) -> Vec<f32> {
    let mut heat = vec![0.0; kv_len];

    for (_id, node) in &web.nodes {
        for t in &node.tokens {
            let idx = t.0;
            if idx < kv_len {
                heat[idx] = node.score;
            }
        }
    }

    heat
}

/// Compressed heatmap.
/// Returns a BitDrop_v2 compressed packet containing the raw heatmap.
pub fn token_score_heatmap_compressed(web: &KvWeb, kv_len: usize) -> Option<Vec<u8>> {
    let heat = token_score_heatmap(web, kv_len);
    web.compressor.as_ref().map(|c| {
        c.compress(&(
            "token_score_heatmap",
            kv_len,
            &heat
        ))
    })
}

/// Normalize a heatmap to [0, 1].
pub fn normalize_heatmap(heat: &mut [f32]) {
    let mut max = 0.0;

    // Find max value
    for &v in heat.iter() {
        if v > max {
            max = v;
        }
    }

    // Avoid division by zero
    if max <= 0.0 {
        return;
    }

    // Normalize
    for v in heat.iter_mut() {
        *v /= max;
    }
}

/// Compressed normalized heatmap.
/// Returns a BitDrop_v2 compressed packet containing the normalized heatmap.
pub fn normalize_heatmap_compressed(web: &KvWeb, kv_len: usize) -> Option<Vec<u8>> {
    let mut heat = token_score_heatmap(web, kv_len);
    normalize_heatmap(&mut heat);

    web.compressor.as_ref().map(|c| {
        c.compress(&(
            "normalize_heatmap",
            kv_len,
            &heat
        ))
    })
}

/// Optional smoothing pass (Gaussian-lite).
/// Helps visualization by reducing sharp spikes.
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

/// Compressed smoothed heatmap.
/// Returns a BitDrop_v2 compressed packet containing the smoothed heatmap.
pub fn smooth_heatmap_compressed(web: &KvWeb, kv_len: usize) -> Option<Vec<u8>> {
    let mut heat = token_score_heatmap(web, kv_len);
    smooth_heatmap(&mut heat);

    web.compressor.as_ref().map(|c| {
        c.compress(&(
            "smooth_heatmap",
            kv_len,
            &heat
        ))
    })
}
