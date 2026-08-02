//! export.rs
//!
//! DAX‑MAX export utilities for KV‑Webb integration.
//!
//! This module provides unified export functions for:
//! - binary packets (semantic zoning, pruning, GPU buffers)
//! - simple text/JSON‑like diagnostics for firewall + KV subsets
//!
//! No external crates (like `serde_json`) are used — everything is
//! written as plain text for maximum portability.

use std::fs::File;
use std::io::{Write, BufWriter};
use std::path::Path;

use kv_web_runtime::semantic::{
    SemanticZoning,
    SemanticRouteDecision,
    FirewallConfig,
    firewall_reverse_mask,
    firewall_zone_reverse_mask,
    firewall_adversarial_clusters,
};

/// Export any binary packet to disk.
pub fn export_binary_packet<P: AsRef<Path>>(path: P, packet: &[u8]) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(packet)?;
    Ok(())
}

/// Export semantic zoning (uncompressed) as a simple text format.
pub fn export_semantic_zoning_text<P: AsRef<Path>>(
    path: P,
    zoning: &SemanticZoning,
) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "# SemanticZoning")?;
    writeln!(w, "root: {:?}", zoning.root)?;
    writeln!(w, "nodes: {:?}", zoning.nodes)?;
    writeln!(w, "index_map: {:?}", zoning.index_map)?;
    writeln!(w, "zones:")?;
    for z in &zoning.zones {
        writeln!(
            w,
            "  zone_id={} start={} end={} size={} centroid_node={:?}",
            z.zone_id, z.start, z.end, z.size, z.centroid_node
        )?;
    }
    writeln!(w, "scratch.layer_a: {:?}", zoning.scratch.layer_a)?;
    writeln!(w, "scratch.layer_b: {:?}", zoning.scratch.layer_b)?;

    Ok(())
}

/// Export semantic zoning (compressed DAX packet) as raw binary.
pub fn export_semantic_zoning_compressed<P: AsRef<Path>>(
    path: P,
    packet: &[u8],
) -> std::io::Result<()> {
    export_binary_packet(path, packet)
}

/// Export semantic routing decision as a simple text format.
pub fn export_semantic_route_decision_text<P: AsRef<Path>>(
    path: P,
    decision: &SemanticRouteDecision,
) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "# SemanticRouteDecision")?;
    match decision {
        SemanticRouteDecision::Circulate(packet) => {
            writeln!(w, "kind: Circulate")?;
            writeln!(w, "packet.id: {}", packet.id)?;
            writeln!(w, "packet.priority: {:?}", packet.priority)?;
            writeln!(w, "packet.root: {:?}", packet.root)?;
            writeln!(w, "packet.hops: {}", packet.hops)?;
            writeln!(w, "packet.last_exit_zone: {:?}", packet.last_exit_zone)?;
            writeln!(w, "packet.route_score: {}", packet.route_score)?;
            writeln!(w, "packet.stability_factor: {}", packet.stability_factor)?;
            writeln!(w, "packet.cognitive_weight: {}", packet.cognitive_weight)?;
            writeln!(w, "packet.tunnel_bias: {}", packet.tunnel_bias)?;
            writeln!(w, "packet.cache_reliability: {}", packet.cache_reliability)?;
            writeln!(w, "packet.predictor_confidence: {}", packet.predictor_confidence)?;
            writeln!(w, "packet.overflow_stability: {}", packet.overflow_stability)?;
        }
        SemanticRouteDecision::Exit { packet, zone_id, node_id } => {
            writeln!(w, "kind: Exit")?;
            writeln!(w, "zone_id: {}", zone_id)?;
            writeln!(w, "node_id: {:?}", node_id)?;
            writeln!(w, "packet.id: {}", packet.id)?;
            writeln!(w, "packet.priority: {:?}", packet.priority)?;
            writeln!(w, "packet.root: {:?}", packet.root)?;
            writeln!(w, "packet.hops: {}", packet.hops)?;
            writeln!(w, "packet.last_exit_zone: {:?}", packet.last_exit_zone)?;
            writeln!(w, "packet.route_score: {}", packet.route_score)?;
            writeln!(w, "packet.stability_factor: {}", packet.stability_factor)?;
            writeln!(w, "packet.cognitive_weight: {}", packet.cognitive_weight)?;
            writeln!(w, "packet.tunnel_bias: {}", packet.tunnel_bias)?;
            writeln!(w, "packet.cache_reliability: {}", packet.cache_reliability)?;
            writeln!(w, "packet.predictor_confidence: {}", packet.predictor_confidence)?;
            writeln!(w, "packet.overflow_stability: {}", packet.overflow_stability)?;
        }
    }

    Ok(())
}

/// Export pruning roundabout compressed packet.
pub fn export_pruning_packet<P: AsRef<Path>>(
    path: P,
    packet: &[u8],
) -> std::io::Result<()> {
    export_binary_packet(path, packet)
}

/// Export KV subset (keys + values) as a simple text format.
pub fn export_kv_subset_text<P: AsRef<Path>>(
    path: P,
    keys: &[Vec<f32>],
    values: &[Vec<f32>],
) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "# KVSubset")?;
    writeln!(w, "keys_len: {}", keys.len())?;
    writeln!(w, "values_len: {}", values.len())?;

    writeln!(w, "keys:")?;
    for (i, k) in keys.iter().enumerate() {
        writeln!(w, "  [{}] {:?}", i, k)?;
    }

    writeln!(w, "values:")?;
    for (i, v) in values.iter().enumerate() {
        writeln!(w, "  [{}] {:?}", i, v)?;
    }

    Ok(())
}

// ============================================================================
// MAX‑TIER FIREWALL EXPORTS
// ============================================================================

/// Export firewall reverse mask (DAX adversarial memory) as text.
pub fn export_firewall_reverse_mask_text<P: AsRef<Path>>(
    path: P,
    cfg: &FirewallConfig,
) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "# FirewallReverseMask")?;
    if let Some(mask) = firewall_reverse_mask(cfg) {
        writeln!(w, "mask_len: {}", mask.len())?;
        writeln!(w, "mask: {:?}", mask)?;
    } else {
        writeln!(w, "mask: <none>")?;
    }

    Ok(())
}

/// Export firewall zone‑reverse mask as text.
pub fn export_firewall_zone_reverse_mask_text<P: AsRef<Path>>(
    path: P,
    cfg: &FirewallConfig,
    zone_threshold: f32,
) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "# FirewallZoneReverseMask")?;
    writeln!(w, "zone_threshold: {}", zone_threshold)?;
    if let Some(mask) = firewall_zone_reverse_mask(cfg, zone_threshold) {
        writeln!(w, "mask_len: {}", mask.len())?;
        writeln!(w, "mask: {:?}", mask)?;
    } else {
        writeln!(w, "mask: <none>")?;
    }

    Ok(())
}

/// Export firewall adversarial clusters as text.
pub fn export_firewall_adversarial_clusters_text<P: AsRef<Path>>(
    path: P,
    cfg: &FirewallConfig,
    max_clusters: usize,
) -> std::io::Result<()> {
    let clusters = firewall_adversarial_clusters(cfg, max_clusters);

    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "# FirewallAdversarialClusters")?;
    writeln!(w, "max_clusters: {}", max_clusters)?;
    writeln!(w, "actual_clusters: {}", clusters.len())?;

    for (i, c) in clusters.iter().enumerate() {
        writeln!(w, "cluster[{}]: {:?}", i, c)?;
    }

    Ok(())
}

// ============================================================================
// GPU‑READY EXPORTS
// ============================================================================

/// Export GPU‑ready binary buffer (semantic or pruning).
pub fn export_gpu_buffer<P: AsRef<Path>>(
    path: P,
    buffer: &[u8],
) -> std::io::Result<()> {
    export_binary_packet(path, buffer)
}

// ============================================================================
// Unified export interface
// ============================================================================

/// Unified export enum for any DAX/MAX packet.
pub enum ExportPacket<'a> {
    SemanticZoningCompressed(&'a [u8]),
    Pruning(&'a [u8]),
    SemanticRoute(&'a SemanticRouteDecision),
    KVSubset {
        keys: &'a [Vec<f32>],
        values: &'a [Vec<f32>],
    },
    FirewallReverseMask(&'a FirewallConfig),
    FirewallZoneMask(&'a FirewallConfig, f32),
    FirewallClusters(&'a FirewallConfig, usize),
}

/// Unified export dispatcher (text + binary).
pub fn export<P: AsRef<Path>>(
    path: P,
    packet: ExportPacket,
) -> std::io::Result<()> {
    match packet {
        ExportPacket::SemanticZoningCompressed(bytes) => export_semantic_zoning_compressed(path, bytes),
        ExportPacket::Pruning(bytes) => export_pruning_packet(path, bytes),
        ExportPacket::SemanticRoute(decision) => export_semantic_route_decision_text(path, decision),
        ExportPacket::KVSubset { keys, values } => export_kv_subset_text(path, keys, values),
        ExportPacket::FirewallReverseMask(cfg) => export_firewall_reverse_mask_text(path, cfg),
        ExportPacket::FirewallZoneMask(cfg, thr) => export_firewall_zone_reverse_mask_text(path, cfg, thr),
        ExportPacket::FirewallClusters(cfg, max) => export_firewall_adversarial_clusters_text(path, cfg, max),
    }
}
