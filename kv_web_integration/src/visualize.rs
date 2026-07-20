//! visualize.rs
//!
//! Utilities to visualize the KV Web structure.
//!
//! This is purely diagnostic / debugging output:
//! - ASCII graph dumps
//! - DOT (Graphviz) export
//! - region‑focused views

use kv_web_core::{KvWeb, WebNodeId, TokenId, WebEdge};
use crate::KvWebRuntime;
use std::fmt::Write;

/// Render a simple ASCII summary of the whole web.
pub fn ascii_overview(web: &KvWeb) -> String {
    let mut out = String::new();

    writeln!(
        &mut out,
        "KV Web Overview: {} nodes, {} edges",
        web.nodes.len(),
        web.edges.len()
    ).ok();

    for (id, node) in &web.nodes {
        writeln!(
            &mut out,
            "  Node {:?}: tokens={:?}, label={:?}, score={:.3}",
            id,
            node.tokens.iter().map(|t| t.0).collect::<Vec<_>>(),
            node.label,
            node.score
        ).ok();
    }

    for edge in &web.edges {
        writeln!(
            &mut out,
            "  Edge {:?} -> {:?} (w={:.3}, kind={:?})",
            edge.from,
            edge.to,
            edge.weight,
            edge.kind
        ).ok();
    }

    out
}

/// Render a region‑focused ASCII view.
pub fn ascii_region(web: &KvWeb, root: WebNodeId, depth: usize) -> String {
    let mut out = String::new();

    let region_nodes = web.nodes_in_region(root, depth);

    writeln!(
        &mut out,
        "Region from {:?} (depth={}): {} nodes",
        root,
        depth,
        region_nodes.len()
    ).ok();

    for id in &region_nodes {
        if let Some(node) = web.nodes.get(id) {
            writeln!(
                &mut out,
                "  Node {:?}: tokens={:?}, label={:?}, score={:.3}",
                id,
                node.tokens.iter().map(|t| t.0).collect::<Vec<_>>(),
                node.label,
                node.score
            ).ok();
        }
    }

    out
}

/// Export the web as DOT (Graphviz).
pub fn to_dot(web: &KvWeb) -> String {
    let mut out = String::new();

    writeln!(&mut out, "digraph KvWeb {{").ok();
    writeln!(&mut out, "  rankdir=LR;").ok();

    // nodes
    for (id, node) in &web.nodes {
        let label = node.label.clone().unwrap_or_else(|| format!("Node {:?}", id.0));
        writeln!(
            &mut out,
            r#"  n{} [label="{}\nscore={:.3}\ntokens={:?}"];"#,
            id.0,
            label.replace('"', "'"),
            node.score,
            node.tokens.iter().map(|t| t.0).collect::<Vec<_>>()
        ).ok();
    }

    // edges
    for edge in &web.edges {
        writeln!(
            &mut out,
            "  n{} -> n{} [label=\"w={:.3}, {:?}\"];",
            edge.from.0,
            edge.to.0,
            edge.weight,
            edge.kind
        ).ok();
    }

    writeln!(&mut out, "}}").ok();

    out
}

/// Export a region as DOT (Graphviz), filtering to nodes in the region.
pub fn region_to_dot(web: &KvWeb, root: WebNodeId, depth: usize) -> String {
    let mut out = String::new();

    let region_nodes = web.nodes_in_region(root, depth);
    let region_set: std::collections::HashSet<WebNodeId> =
        region_nodes.iter().cloned().collect();

    writeln!(&mut out, "digraph KvWebRegion {{").ok();
    writeln!(&mut out, "  rankdir=LR;").ok();

    // nodes
    for id in &region_nodes {
        if let Some(node) = web.nodes.get(id) {
            let label = node.label.clone().unwrap_or_else(|| format!("Node {:?}", id.0));
            writeln!(
                &mut out,
                r#"  n{} [label="{}\nscore={:.3}\ntokens={:?}"];"#,
                id.0,
                label.replace('"', "'"),
                node.score,
                node.tokens.iter().map(|t| t.0).collect::<Vec<_>>()
            ).ok();
        }
    }

    // edges inside region
    for edge in &web.edges {
        if region_set.contains(&edge.from) && region_set.contains(&edge.to) {
            writeln!(
                &mut out,
                "  n{} -> n{} [label=\"w={:.3}, {:?}\"];",
                edge.from.0,
                edge.to.0,
                edge.weight,
                edge.kind
            ).ok();
        }
    }

    writeln!(&mut out, "}}").ok();

    out
}
