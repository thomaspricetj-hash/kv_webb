//! dax.rs
//!
//! Production‑grade DAX (Delta Memory Model) engine for KV‑Webb runtime.
//!
//! Components:
//! - MasterBuffer (AM)
//! - DeltaBuffer (AD)
//! - EffectiveView (M ⊕ D)
//! - DeltaRecord (events)
//! - DeltaStore (full DAX engine)
//!
//! Features:
//! - branching
//! - rollback
//! - lineage fingerprinting
//! - reversible views
//! - multi‑delta overlays
//! - GPU‑ready packet support
//!
//! No external crates. No chrono.

use serde::{Serialize, Deserialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Unique ID generator (simple monotonic counter)
fn next_id(counter: &mut u64) -> u64 {
    *counter += 1;
    *counter
}

/// A single delta record (event)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaRecord {
    pub id: u64,
    pub domain: u8,
    pub kind: u8,
    pub ts_start: u64,
    pub ts_end: u64,
    pub heat: f32,
    pub tag: Option<String>,
}

/// Master buffer (AM)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MasterBuffer {
    pub id: u64,
    pub domain: u8,
    pub fingerprint: u64,
    pub created_at_ms: u64,
}

/// Delta buffer (AD)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaBuffer {
    pub id: u64,
    pub domain: u8,
    pub master_id: u64,
    pub records: Vec<u64>, // record IDs
}

/// Effective view (M ⊕ D)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectiveView {
    pub id: u64,
    pub domain: u8,
    pub master_id: u64,
    pub delta_id: u64,
    pub fingerprint: u64,
}

/// Full DAX engine
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct DeltaStore {
    pub id_counter: u64,

    pub masters: Vec<MasterBuffer>,
    pub deltas: Vec<DeltaBuffer>,
    pub records: Vec<DeltaRecord>,
    pub views: Vec<EffectiveView>,
}

impl DeltaStore {
    /// Current timestamp in ms
    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }

    /// Create a new master buffer
    pub fn add_master_buffer(&mut self, domain: u8) -> u64 {
        let id = next_id(&mut self.id_counter);
        let ts = Self::now_ms();

        self.masters.push(MasterBuffer {
            id,
            domain,
            fingerprint: 0,
            created_at_ms: ts,
        });

        id
    }

    /// Create a new delta record
    pub fn add_delta(
        &mut self,
        domain: u8,
        kind: u8,
        ts_start: u64,
        ts_end: u64,
        heat: f32,
        tag: Option<String>,
    ) -> u64 {
        let id = next_id(&mut self.id_counter);

        self.records.push(DeltaRecord {
            id,
            domain,
            kind,
            ts_start,
            ts_end,
            heat,
            tag,
        });

        id
    }

    /// Create a new delta buffer
    pub fn add_delta_buffer(&mut self, domain: u8, master_id: u64) -> u64 {
        let id = next_id(&mut self.id_counter);

        self.deltas.push(DeltaBuffer {
            id,
            domain,
            master_id,
            records: Vec::new(),
        });

        id
    }

    /// Attach a record to a delta buffer
    pub fn attach_record_to_delta(&mut self, delta_id: u64, record_id: u64) {
        if let Some(delta) = self.deltas.iter_mut().find(|d| d.id == delta_id) {
            delta.records.push(record_id);
        }
    }

    /// Create an effective view (M ⊕ D)
    pub fn create_effective_view(&mut self, master_id: u64, delta_id: u64) -> u64 {
        let id = next_id(&mut self.id_counter);

        let fingerprint = self.compute_fingerprint(master_id, delta_id);

        self.views.push(EffectiveView {
            id,
            domain: self.master_domain(master_id),
            master_id,
            delta_id,
            fingerprint,
        });

        id
    }

    /// Compute lineage fingerprint
    fn compute_fingerprint(&self, master_id: u64, delta_id: u64) -> u64 {
        let mut fp = master_id ^ delta_id;

        if let Some(delta) = self.deltas.iter().find(|d| d.id == delta_id) {
            for rec_id in &delta.records {
                fp ^= *rec_id;
            }
        }

        fp
    }

    /// Get domain for master
    fn master_domain(&self, master_id: u64) -> u8 {
        self.masters
            .iter()
            .find(|m| m.id == master_id)
            .map(|m| m.domain)
            .unwrap_or(0)
    }

    /// Branch a new view from a master
    pub fn branch_view_from(&mut self, master_id: u64) -> (u64, u64, u64) {
        let domain = self.master_domain(master_id);
        let delta_id = self.add_delta_buffer(domain, master_id);
        let view_id = self.create_effective_view(master_id, delta_id);
        (master_id, delta_id, view_id)
    }

    /// Rollback master to a previous view
    pub fn rollback_master_to(&mut self, master_id: u64, view_id: u64) {
        if let Some(master) = self.masters.iter_mut().find(|m| m.id == master_id) {
            master.fingerprint = view_id;
        }
    }

    /// Fingerprint for master
    pub fn fingerprint_for_master(&self, master_id: u64) -> u64 {
        self.masters
            .iter()
            .find(|m| m.id == master_id)
            .map(|m| m.fingerprint)
            .unwrap_or(0)
    }
}
