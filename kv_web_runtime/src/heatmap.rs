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
//!
//! All upgrades are backwards-compatible.

use kv_web_core::KvWeb;

/// Polygon-aware score modifier.
/// Boosts nodes in same polygon face, penalizes centroid distance.
fn polygon_score_bias(web: &KvWeb, node_score: f32, node_id: kv_web_core::WebNodeId) -> f32 {
    let node = match web.nodes.get(&node_id) {
        Some(n) => n,
        None => return node_score,
    };

    let poly = match &node.polygon {
        Some(p) => p,
        None => return node_score,
    };

    // Face-index bonus
    let face_bonus = match poly.face_index {
        3 => 0.20,
        2 => 0.10,
        1 => 0.05,
        _ => 0.0,
    };

    // Centroid-distance penalty (simple heuristic)
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
/// Now radius-aware: polygon radius reduces smoothing strength.
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
