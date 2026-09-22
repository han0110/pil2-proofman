//! Proof-done completion accounting, modelled as an owned capability.
//!
//! A GPU proof is launched on one thread and completes asynchronously: the C++ harvest fires
//! `proof_done_callback`, which crosses the FFI into a Rust channel where a worker settles the unit.
//! The old bare [`Counter`](crate::Counter) plus process-global channel had two lifetime bugs: the
//! count could underflow (`fetch_sub` past zero wedged the phase until a 10-min timeout), and
//! single-owner-at-a-time was an unenforced convention. So [`DeviceCompletions`] owns the one
//! callback registration and hands out a [`CompletionOwner`] **one at a time**, made safe by:
//!
//! - **Idempotent settling, keyed by `(id, kind)`.** [`Ledger`] is a set of outstanding units, not a
//!   bare count (a basic proof and its recursive successor share a numeric id); settling an absent
//!   unit is a no-op, so a duplicate/late completion can neither double-count nor wrap past zero.
//! - **Drop drains before it releases, and the owner holds no sender.** The sender is moved into the
//!   C registration; clearing it in `Drop` disconnects the workers' receivers so they exit. Sharing
//!   only [`Arc<Ledger>`](Ledger), the owner never waits on a worker that is waiting on it.

use std::collections::HashSet;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::Duration;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crossbeam_channel::{unbounded, Receiver};
use proofman_starks_lib_c::{
    clear_proof_done_callback_c, get_stream_proofs_c, get_stream_proofs_non_blocking_c, register_proof_done_callback_c,
    CompletionMsg,
};
use proofman_starks_lib_c::{PROOF_TIMING_COMPUTING_WC, PROOF_TIMING_PREPARING_WC, PROOF_TIMING_SECTIONS};

use proofman_common::ProofType;

use crate::CancellationInfo;

pub const RECORD_KIND_WITNESS: usize = 8;
pub const RECORD_KIND_COMMIT: usize = 9;
pub const RECORD_KIND_EXECUTE: usize = 10;
pub const RECORD_KIND_PRE_CALCULATE: usize = 11;
pub const RECORD_KIND_CHALLENGE: usize = 12;
pub const RECORD_KIND_RECOMPUTE: usize = 13;
pub const RECORD_KIND_RECURSION_WITNESS: usize = 14;
pub const RECORD_KIND_AGGREGATION_WITNESS: usize = 15;

pub const RECORD_KIND_PER_INSTANCE: [usize; 7] = [
    ProofType::Basic as usize,
    ProofType::Compressor as usize,
    ProofType::Recursive1 as usize,
    RECORD_KIND_WITNESS,
    RECORD_KIND_COMMIT,
    RECORD_KIND_RECOMPUTE,
    RECORD_KIND_RECURSION_WITNESS,
];

/// Poll cadence while draining outstanding units — unchanged from the previous busy-poll wait.
const POLL_INTERVAL: Duration = Duration::from_micros(100);
/// How long `Drop` pumps the non-blocking harvest for stragglers before giving up. Bounded because
/// on a cancelled job they may never complete; a late straggler is a no-op (idempotent settling).
const DRAIN_BUDGET: Duration = Duration::from_secs(5);
/// How long [`SlotToken::take`] waits before warning that the capability is still held. Only a
/// call-site ordering bug gets it here, and that would otherwise present as a silent hang.
const SLOT_WAIT_WARN_AFTER: Duration = Duration::from_secs(10);

/// One outstanding proof unit: `(id, ProofType as usize)`. Keying on the discriminant distinguishes
/// a basic proof from the recursive proof that reuses its id, without coupling to the enum.
type UnitKey = (u64, usize);

/// Process-lifetime owner of the single proof-done callback registration. Held by `ProofMan`; hands
/// out one [`CompletionOwner`] at a time via [`Self::acquire`], released when that owner drops.
pub struct DeviceCompletions {
    /// `true` while a `CompletionOwner` is alive. This is the "one at a time" guarantee.
    slot: Arc<Mutex<bool>>,
    /// Monotonic id handed to each owner (diagnostics only; see [`Ledger::epoch`]).
    next_epoch: AtomicU64,
}

impl Default for DeviceCompletions {
    fn default() -> Self {
        Self::new()
    }
}

impl DeviceCompletions {
    pub fn new() -> Self {
        Self { slot: Arc::new(Mutex::new(false)), next_epoch: AtomicU64::new(1) }
    }

    /// Take the right to receive proof-done completions. Blocks until any previous [`CompletionOwner`]
    /// has dropped — turning "stop the aggregation service before arming" from convention into an
    /// enforced invariant (the callers are already sequential, so it never really contends).
    ///
    /// `d_buffers` is used by the owner's `Drop` for the final blocking harvest; null on the CPU backend.
    pub fn acquire(&self, d_buffers: DeviceBuffersPtr) -> CompletionOwner {
        let slot = SlotToken::take(&self.slot);
        let epoch = self.next_epoch.fetch_add(1, Ordering::Relaxed);
        let ledger = Arc::new(Ledger::new(epoch));

        // The sender is MOVED into the C registration; owner and workers hold only receivers. That's
        // why clearing the registration in `Drop` disconnects every receiver.
        let (tx, rx) = unbounded::<CompletionMsg>();
        register_proof_done_callback_c(tx);

        CompletionOwner { ledger, rx, d_buffers, _slot: slot }
    }
}

/// Device-buffer pointer carried across threads for the final harvest in `Drop`. Owned by the device
/// layer (outlives every [`CompletionOwner`]); the wrapper only carries it without making the owner
/// non-`Send`.
#[derive(Clone, Copy)]
pub struct DeviceBuffersPtr(pub *mut std::ffi::c_void);

// SAFETY: the pointer refers to device-layer state that lives for the whole process and is only
// passed to FFI entry points that are internally synchronised.
unsafe impl Send for DeviceBuffersPtr {}
unsafe impl Sync for DeviceBuffersPtr {}

/// Proof that the holder owns the completion capability. A `bool` behind a mutex rather than a held
/// `MutexGuard`, so the capability can span an arbitrary scope without borrowing from
/// [`DeviceCompletions`]. Waiting is a short poll — the sites that take it are already sequential.
struct SlotToken {
    slot: Arc<Mutex<bool>>,
}

impl SlotToken {
    fn take(slot: &Arc<Mutex<bool>>) -> Self {
        let start = std::time::Instant::now();
        let mut warned = false;
        loop {
            {
                // A poisoned slot only means a previous holder panicked; the flag is still
                // meaningful, so recover rather than abort teardown.
                let mut held = slot.lock().unwrap_or_else(|p| p.into_inner());
                if !*held {
                    *held = true;
                    return Self { slot: Arc::clone(slot) };
                }
            }
            // The callers are sequential, so this never really contends: waiting for more than a
            // moment means a previous owner was never released (typically an aggregation service
            // left running before an `acquire`). There is nothing to recover here — the wait is
            // unbounded by design — but it must not look like a silent hang.
            if !warned && start.elapsed() >= SLOT_WAIT_WARN_AFTER {
                warned = true;
                tracing::warn!(
                    "Waiting >{}s for the proof-done completion capability: a previous CompletionOwner \
                     is still alive. Stop the outer-aggregation service before acquiring.",
                    SLOT_WAIT_WARN_AFTER.as_secs()
                );
            }
            std::thread::sleep(POLL_INTERVAL);
        }
    }
}

impl Drop for SlotToken {
    fn drop(&mut self) {
        let mut held = self.slot.lock().unwrap_or_else(|p| p.into_inner());
        *held = false;
    }
}

/// Shared accounting for one completion epoch: which units are outstanding, and how many. Cloned as
/// `Arc<Ledger>` into every worker to arm and settle units. Deliberately holds no channel sender, so
/// a worker never keeps the registration alive (see module docs).
pub struct Ledger {
    epoch: u64,
    /// Outstanding units. The set — not a bare count — is the source of truth, which is what makes
    /// [`Self::settle`] idempotent and underflow-free.
    outstanding: Mutex<HashSet<UnitKey>>,
    /// Mirrors `outstanding.len()` so waiters have a cheap predicate without taking the set lock.
    remaining: AtomicUsize,
    wait_lock: Mutex<()>,
    cvar: Condvar,
}

impl Ledger {
    fn new(epoch: u64) -> Self {
        Self {
            epoch,
            outstanding: Mutex::new(HashSet::new()),
            remaining: AtomicUsize::new(0),
            wait_lock: Mutex::new(()),
            cvar: Condvar::new(),
        }
    }

    /// Monotonic id for this owner's registration. Diagnostics only — cross-owner completions are
    /// handled structurally (channel disconnect + idempotent settling), not by filtering on it.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Number of units still outstanding.
    pub fn remaining(&self) -> usize {
        self.remaining.load(Ordering::Acquire)
    }

    /// Record a new unit as in flight, returning a guard that settles it if the caller never hands it
    /// to the async pipeline. Re-arming an already-outstanding unit is a no-op for the count (the set
    /// rejects the duplicate), so a double-arm cannot inflate the ledger.
    #[must_use = "dropping the token settles the unit immediately; hold it until the work is launched"]
    pub fn arm(self: &Arc<Self>, id: u64, kind: usize) -> ProofToken {
        let key = (id, kind);
        {
            // Update the set and its mirror count under one lock, so `remaining` is never observed
            // out of step with the set and an interleaved `settle` can't fetch_sub below its size.
            let mut set = self.outstanding.lock().unwrap_or_else(|p| p.into_inner());
            if set.insert(key) {
                self.remaining.fetch_add(1, Ordering::AcqRel);
            }
        }
        ProofToken { ledger: Arc::clone(self), key, armed: true }
    }

    /// Adopt settling of an *already*-outstanding unit without changing the count — a worker taking
    /// over an in-flight unit to launch. The returned guard settles on drop unless
    /// [`commit`](ProofToken::commit)ted, so a failed launch still balances the ledger.
    #[must_use = "dropping the token settles the unit immediately; hold it until the work is launched"]
    pub fn adopt(self: &Arc<Self>, id: u64, kind: usize) -> ProofToken {
        ProofToken { ledger: Arc::clone(self), key: (id, kind), armed: true }
    }

    /// Settle one outstanding unit. Idempotent: a unit that is not outstanding (already settled,
    /// never armed, or from a previous owner) leaves the ledger untouched — removing the old
    /// `fetch_sub(1) - 1` underflow and the ordered-`reset()` workaround that dodged the wrap.
    pub fn settle(&self, id: u64, kind: usize) {
        // Remove from the set and decrement the mirror count under one lock (see `arm`). The notify
        // is done after releasing it, holding only `wait_lock`, to keep the two locks unnested.
        let now = {
            let mut set = self.outstanding.lock().unwrap_or_else(|p| p.into_inner());
            if !set.remove(&(id, kind)) {
                return;
            }
            self.remaining.fetch_sub(1, Ordering::AcqRel) - 1
        };
        if now == 0 {
            let _g = self.wait_lock.lock().unwrap_or_else(|p| p.into_inner());
            self.cvar.notify_all();
        }
    }

    /// Block until every unit has settled, the job is cancelled, or `timeout` elapses; returns `true`
    /// only if the ledger reached zero. `pump` runs each iteration to drive the non-blocking harvest.
    /// The condvar is the mechanism; the 100 µs poll is only a backstop.
    pub fn wait_settled<P: FnMut()>(
        &self,
        mut pump: P,
        cancellation_info: &RwLock<CancellationInfo>,
        timeout: Option<Duration>,
    ) -> bool {
        let start = std::time::Instant::now();
        let mut guard = self.wait_lock.lock().unwrap_or_else(|p| p.into_inner());
        loop {
            if self.remaining.load(Ordering::Acquire) == 0 {
                return true;
            }
            let cancelled = {
                let info = cancellation_info.read().unwrap_or_else(|p| p.into_inner());
                info.token.is_cancelled()
            };
            if cancelled {
                return false;
            }
            if let Some(limit) = timeout {
                if start.elapsed() >= limit {
                    // Name the units that never settled: the only trace a wedged pipeline leaves.
                    let mut left: Vec<UnitKey> =
                        self.outstanding.lock().unwrap_or_else(|p| p.into_inner()).iter().copied().collect();
                    left.sort_unstable();
                    tracing::warn!(
                        "ledger: {} unit(s) never settled after {:?}: {:?} (id, kind)",
                        left.len(),
                        limit,
                        left
                    );
                    return false;
                }
            }
            pump();
            let (g, _) = self.cvar.wait_timeout(guard, POLL_INTERVAL).unwrap_or_else(|p| p.into_inner());
            guard = g;
        }
    }
}

/// The exclusive right to receive proof-done completions for one epoch (construct via
/// [`DeviceCompletions::acquire`]). Held by the phase's main thread only, never cloned into a worker,
/// so its `Drop` can tear the epoch down; workers instead get an [`Arc<Ledger>`](Ledger) and a receiver.
pub struct CompletionOwner {
    ledger: Arc<Ledger>,
    rx: Receiver<CompletionMsg>,
    /// Used by `Drop` to run the final blocking harvest before releasing the registration.
    d_buffers: DeviceBuffersPtr,
    _slot: SlotToken,
}

impl CompletionOwner {
    /// The shared ledger; clone into each worker so it can arm/adopt/settle.
    pub fn ledger(&self) -> Arc<Ledger> {
        Arc::clone(&self.ledger)
    }

    /// A receiver clone for a worker to consume completions from. All clones disconnect when this
    /// owner is dropped, which is how workers learn to exit (no sentinel counting).
    pub fn receiver(&self) -> Receiver<CompletionMsg> {
        self.rx.clone()
    }

    /// A monotonic id for this owner (diagnostics only; see [`Ledger::epoch`]).
    pub fn epoch(&self) -> u64 {
        self.ledger.epoch()
    }

    /// Units still outstanding.
    pub fn remaining(&self) -> usize {
        self.ledger.remaining()
    }

    /// See [`Ledger::wait_settled`].
    pub fn wait_settled<P: FnMut()>(
        &self,
        pump: P,
        cancellation_info: &RwLock<CancellationInfo>,
        timeout: Option<Duration>,
    ) -> bool {
        self.ledger.wait_settled(pump, cancellation_info, timeout)
    }
}

impl Drop for CompletionOwner {
    /// Drain, **then** release — in that order. Releasing first would drop still-in-flight
    /// completions (the lost-decrement bug); centralizing the order here means no call site can get
    /// it wrong. The drain is bounded because a cancelled job's units may never complete; giving up
    /// is safe because [`Ledger::settle`] is idempotent, so a late straggler is a no-op.
    fn drop(&mut self) {
        // (1) Give outstanding completions a bounded chance to land, pumping the non-blocking
        //     harvest so the GPU side can deliver them.
        let start = std::time::Instant::now();
        while self.ledger.remaining() > 0 && start.elapsed() < DRAIN_BUDGET {
            if !self.d_buffers.0.is_null() {
                get_stream_proofs_non_blocking_c(self.d_buffers.0);
            }
            std::thread::sleep(POLL_INTERVAL);
        }

        // (2) Final blocking harvest so anything already completed on the device is delivered.
        if !self.d_buffers.0.is_null() {
            get_stream_proofs_c(self.d_buffers.0);
        }

        let leaked = self.ledger.remaining();
        if leaked > 0 {
            // Expected on a cancelled job; noteworthy otherwise. Not fatal: the next owner starts
            // from an empty ledger, and late completions for this epoch are ignored.
            tracing::debug!(
                "CompletionOwner(epoch {}) released with {} unit(s) unsettled (expected after cancellation)",
                self.ledger.epoch(),
                leaked
            );
        }

        // (3) Release the registration. This drops the sole sender, so every receiver clone
        //     disconnects and its worker loop exits — no sentinel needed.
        clear_proof_done_callback_c();
    }
}

/// RAII guard for one in-flight unit. Dropping an uncommitted token settles its unit, so a unit
/// cannot leak on a failed launch, a cancellation break, or an unwind. [`ProofToken::commit`] hands
/// settlement to the async completion path once the work is genuinely launched.
#[must_use = "dropping the token settles the unit immediately; hold it until the work is launched"]
pub struct ProofToken {
    ledger: Arc<Ledger>,
    key: UnitKey,
    armed: bool,
}

impl ProofToken {
    /// The async completion now owns settling this unit.
    pub fn commit(mut self) {
        self.armed = false;
    }

    /// The proof id of this unit.
    pub fn id(&self) -> u64 {
        self.key.0
    }
}

impl Drop for ProofToken {
    fn drop(&mut self) {
        if self.armed {
            self.ledger.settle(self.key.0, self.key.1);
        }
    }
}

#[derive(Clone, Debug)]
pub struct ProofRecord {
    pub id: u64,
    pub proof_type: usize,
    pub airgroup_id: usize,
    pub air_id: usize,
    pub stream_id: u32,
    pub start_ms: u64,
    pub end_ms: u64,
    pub breakdown_ms: [u32; PROOF_TIMING_SECTIONS],
}

pub struct ProofRecorder {
    origin: Mutex<(Instant, u64)>,
    open: Mutex<HashMap<UnitKey, (Instant, ProofRecord)>>,
    done: Mutex<Vec<ProofRecord>>,
    job: AtomicU64,
    reserved: Mutex<(u64, u64)>,
    stash: Mutex<HashMap<usize, (Instant, Instant)>>,
}

impl Default for ProofRecorder {
    fn default() -> Self {
        Self::new()
    }
}

impl ProofRecorder {
    pub fn new() -> Self {
        Self {
            origin: Mutex::new((Instant::now(), unix_ms())),
            open: Mutex::new(HashMap::new()),
            done: Mutex::new(Vec::new()),
            job: AtomicU64::new(0),
            reserved: Mutex::new((u64::MAX, 0)),
            stash: Mutex::new(HashMap::new()),
        }
    }

    pub fn record_launch(&self, id: u64, kind: usize, airgroup_id: usize, air_id: usize, stream_id: u32) {
        self.record_launch_at(id, kind, airgroup_id, air_id, stream_id, Instant::now());
    }

    pub fn record_launch_at(
        &self,
        id: u64,
        kind: usize,
        airgroup_id: usize,
        air_id: usize,
        stream_id: u32,
        launch: Instant,
    ) {
        let origin = self.origin.lock().unwrap_or_else(|p| p.into_inner());
        let start_ms = origin.1 + start_offset_ms(origin.0, launch) as u64;
        let record = ProofRecord {
            id,
            proof_type: kind,
            airgroup_id,
            air_id,
            stream_id,
            start_ms,
            end_ms: start_ms,
            breakdown_ms: [0; PROOF_TIMING_SECTIONS],
        };
        self.open.lock().unwrap_or_else(|p| p.into_inner()).insert((id, kind), (launch, record));
    }

    pub fn record_completion(&self, id: u64, kind: usize, breakdown_ms: [u32; PROOF_TIMING_SECTIONS]) {
        self.record_completion_at(id, kind, breakdown_ms, Instant::now());
    }

    pub fn record_completion_at(&self, id: u64, kind: usize, breakdown_ms: [u32; PROOF_TIMING_SECTIONS], at: Instant) {
        let origin = self.origin.lock().unwrap_or_else(|p| p.into_inner());
        let end_ms = origin.1 + end_offset_ms(origin.0, at) as u64;
        let record = self.open.lock().unwrap_or_else(|p| p.into_inner()).remove(&(id, kind));
        if let Some((_, mut record)) = record {
            record.end_ms = end_ms;
            merge_sections(&mut record.breakdown_ms, breakdown_ms);
            self.push(record);
        }
    }

    pub fn record_section(&self, id: u64, kind: usize, index: usize, start: Instant) {
        self.record_section_us(id, kind, index, start.elapsed().as_micros() as u64);
    }

    pub fn record_section_us(&self, id: u64, kind: usize, index: usize, micros: u64) {
        let ms = section_ms(micros);
        if let Some((_, record)) = self.open.lock().unwrap_or_else(|p| p.into_inner()).get_mut(&(id, kind)) {
            record.breakdown_ms[index] = ms;
        }
    }

    pub fn record_launch_section(&self, id: u64, kind: usize, index: usize, at: Instant) {
        if let Some((launch, record)) = self.open.lock().unwrap_or_else(|p| p.into_inner()).get_mut(&(id, kind)) {
            record.breakdown_ms[index] = section_ms(at.saturating_duration_since(*launch).as_micros() as u64);
        }
    }

    pub fn record_sections(&self, id: u64, kind: usize, breakdown_ms: [u32; PROOF_TIMING_SECTIONS]) {
        if let Some((_, record)) = self.open.lock().unwrap_or_else(|p| p.into_inner()).get_mut(&(id, kind)) {
            merge_sections(&mut record.breakdown_ms, breakdown_ms);
        }
    }

    pub fn record_build(&self, id: u64, kind: usize, airgroup_id: usize, air_id: usize, span: (Instant, u64, u64)) {
        let (start, preparing_us, computing_us) = span;
        self.record_launch_at(id, kind, airgroup_id, air_id, u32::MAX, start);
        self.record_section_us(id, kind, PROOF_TIMING_PREPARING_WC, preparing_us);
        self.record_section_us(id, kind, PROOF_TIMING_COMPUTING_WC, computing_us);
        self.record_completion(id, kind, [0; PROOF_TIMING_SECTIONS]);
    }

    pub fn record_span(&self, id: u64, kind: usize, airgroup_id: usize, air_id: usize, start: Instant) {
        self.record_span_at(id, kind, airgroup_id, air_id, start, Instant::now());
    }

    pub fn stash_span(&self, key: usize, start: Instant) {
        self.stash.lock().unwrap_or_else(|p| p.into_inner()).insert(key, (start, Instant::now()));
    }

    pub fn record_stashed_span(&self, key: usize, id: u64, kind: usize, airgroup_id: usize, air_id: usize) {
        let span = self.stash.lock().unwrap_or_else(|p| p.into_inner()).remove(&key);
        if let Some((start, end)) = span {
            self.record_span_at(id, kind, airgroup_id, air_id, start, end);
        }
    }

    fn record_span_at(&self, id: u64, kind: usize, airgroup_id: usize, air_id: usize, start: Instant, end: Instant) {
        let origin = self.origin.lock().unwrap_or_else(|p| p.into_inner());
        let record = ProofRecord {
            id,
            proof_type: kind,
            airgroup_id,
            air_id,
            stream_id: u32::MAX,
            start_ms: origin.1 + start_offset_ms(origin.0, start) as u64,
            end_ms: origin.1 + end_offset_ms(origin.0, end) as u64,
            breakdown_ms: [0; PROOF_TIMING_SECTIONS],
        };
        self.push(record);
    }

    fn push(&self, mut record: ProofRecord) {
        let (base, span) = *self.reserved.lock().unwrap_or_else(|p| p.into_inner());
        if is_fold(record.proof_type) && record.id >= base {
            record.id += span;
        }
        self.done.lock().unwrap_or_else(|p| p.into_inner()).push(record);
    }

    pub fn close_epoch(&self) {
        self.open.lock().unwrap_or_else(|p| p.into_inner()).clear();
        self.stash.lock().unwrap_or_else(|p| p.into_inner()).clear();
    }

    pub fn reset(&self) {
        self.close_epoch();
        self.done.lock().unwrap_or_else(|p| p.into_inner()).clear();
        *self.origin.lock().unwrap_or_else(|p| p.into_inner()) = (Instant::now(), unix_ms());
    }

    pub fn next_job(&self) {
        self.job.fetch_add(1, Ordering::Relaxed);
        *self.reserved.lock().unwrap_or_else(|p| p.into_inner()) = (u64::MAX, 0);
    }

    pub fn take(&self) -> Vec<ProofRecord> {
        std::mem::take(&mut *self.done.lock().unwrap_or_else(|p| p.into_inner()))
    }

    pub fn export(&self) -> Vec<u64> {
        let mut words = vec![self.job.load(Ordering::Relaxed)];
        for r in self.take() {
            words.extend([r.id, r.proof_type as u64, r.airgroup_id as u64, r.air_id as u64, r.stream_id as u64]);
            words.extend([r.start_ms, r.end_ms]);
            words.extend(r.breakdown_ms.map(u64::from));
        }
        words
    }

    pub fn import(&self, words: &[u64]) -> bool {
        if words[0] != self.job.load(Ordering::Relaxed) {
            return false;
        }
        let mut done = self.done.lock().unwrap_or_else(|p| p.into_inner());
        let offset = done.iter().filter(|r| is_fold(r.proof_type)).map(|r| r.id + 1).max().unwrap_or(0);
        let mut end = offset;
        for w in words[1..].chunks_exact(7 + PROOF_TIMING_SECTIONS) {
            let id = if is_fold(w[1] as usize) { w[0] + offset } else { w[0] };
            if is_fold(w[1] as usize) {
                end = end.max(id + 1);
            }
            done.push(ProofRecord {
                id,
                proof_type: w[1] as usize,
                airgroup_id: w[2] as usize,
                air_id: w[3] as usize,
                stream_id: w[4] as u32,
                start_ms: w[5],
                end_ms: w[6],
                breakdown_ms: std::array::from_fn(|i| w[7 + i] as u32),
            });
        }
        drop(done);
        if end > offset {
            let mut reserved = self.reserved.lock().unwrap_or_else(|p| p.into_inner());
            *reserved = (reserved.0.min(offset), reserved.1 + end - offset);
        }
        true
    }
}

fn is_fold(kind: usize) -> bool {
    kind == ProofType::Recursive2 as usize || kind == RECORD_KIND_AGGREGATION_WITNESS
}

fn unix_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64
}

fn start_offset_ms(origin: Instant, at: Instant) -> u32 {
    at.saturating_duration_since(origin).as_millis() as u32
}

fn end_offset_ms(origin: Instant, at: Instant) -> u32 {
    at.saturating_duration_since(origin).as_nanos().div_ceil(1_000_000) as u32
}

fn section_ms(micros: u64) -> u32 {
    (micros as f64 / 1000.0).round() as u32
}

fn merge_sections(slots: &mut [u32; PROOF_TIMING_SECTIONS], breakdown_ms: [u32; PROOF_TIMING_SECTIONS]) {
    for (slot, ms) in slots.iter_mut().zip(breakdown_ms) {
        if ms != 0 {
            *slot = ms;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proofman_common::{BufferPool, TimedBufferPool};
    use proofman_fields::Goldilocks;
    use proofman_starks_lib_c::{PROOF_TIMING_LAUNCH_PROLOGUE, PROOF_TIMING_PREPARING_WC, PROOF_TIMING_WITNESS_EXPANSION};

    const BASIC: usize = 0;
    const RECURSIVE1: usize = 2;

    struct SlowPool;

    impl BufferPool<Goldilocks> for SlowPool {
        fn take_buffer(&self) -> Vec<Goldilocks> {
            std::thread::sleep(Duration::from_millis(5));
            Vec::new()
        }
    }

    /// A CPU-backend owner: a null device pointer makes `Drop`'s harvest a no-op, so these tests
    /// exercise the ledger without touching the GPU.
    fn owner() -> CompletionOwner {
        DeviceCompletions::new().acquire(DeviceBuffersPtr(std::ptr::null_mut()))
    }

    #[test]
    fn arm_then_drop_settles() {
        let o = owner();
        let l = o.ledger();
        let t = l.arm(7, BASIC);
        assert_eq!(l.remaining(), 1);
        drop(t);
        assert_eq!(l.remaining(), 0);
    }

    #[test]
    fn commit_transfers_settlement_to_the_async_path() {
        let o = owner();
        let l = o.ledger();
        l.arm(7, BASIC).commit();
        assert_eq!(l.remaining(), 1, "a committed unit stays outstanding until it completes");
        l.settle(7, BASIC);
        assert_eq!(l.remaining(), 0);
    }

    #[test]
    fn adopt_does_not_change_the_count_but_settles_on_drop() {
        let o = owner();
        let l = o.ledger();
        l.arm(7, BASIC).commit();
        let taken = l.adopt(7, BASIC);
        assert_eq!(l.remaining(), 1, "adopt must not double-count an already-armed unit");
        drop(taken);
        assert_eq!(l.remaining(), 0, "an uncommitted adopt settles the unit it took over");
    }

    #[test]
    fn settling_an_unknown_unit_is_a_no_op() {
        // The case that used to wrap the counter to usize::MAX and wedge the prove phase.
        let o = owner();
        let l = o.ledger();
        l.settle(1234, BASIC);
        assert_eq!(l.remaining(), 0);
        l.arm(1, BASIC).commit();
        l.settle(999, BASIC);
        assert_eq!(l.remaining(), 1, "a stray completion must not settle someone else's unit");
    }

    #[test]
    fn the_same_id_with_a_different_kind_is_a_distinct_unit() {
        // A basic proof and the recursive proof derived from it can share a numeric id.
        let o = owner();
        let l = o.ledger();
        l.arm(5, BASIC).commit();
        l.arm(5, RECURSIVE1).commit();
        assert_eq!(l.remaining(), 2, "(5, Basic) and (5, Recursive1) are different units");
        l.settle(5, BASIC);
        assert_eq!(l.remaining(), 1, "settling the basic unit must not settle the recursive one");
        l.settle(5, RECURSIVE1);
        assert_eq!(l.remaining(), 0);
    }

    #[test]
    fn settling_twice_only_counts_once() {
        let o = owner();
        let l = o.ledger();
        l.arm(3, BASIC).commit();
        l.settle(3, BASIC);
        l.settle(3, BASIC);
        assert_eq!(l.remaining(), 0);
    }

    #[test]
    fn a_panicking_worker_still_settles_its_unit() {
        let o = owner();
        let l = o.ledger();
        let l2 = Arc::clone(&l);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _t = l2.arm(11, BASIC);
            panic!("failure between arming and launch");
        }));
        assert!(r.is_err());
        assert_eq!(l.remaining(), 0);
    }

    #[test]
    fn workers_learn_to_exit_when_the_owner_drops() {
        let o = owner();
        let rx = o.receiver();
        drop(o);
        assert!(rx.recv().is_err(), "dropping the owner disconnects consumers; no sentinel needed");
    }

    #[test]
    fn the_capability_is_exclusive() {
        let completions = DeviceCompletions::new();
        let null = DeviceBuffersPtr(std::ptr::null_mut());
        let first = completions.acquire(null);
        let epoch = first.epoch();
        drop(first);
        let second = completions.acquire(null);
        assert!(second.epoch() > epoch, "each owner gets a fresh epoch");
    }

    #[test]
    fn records_keep_their_order_and_take_empties_the_recorder() {
        let recorder = ProofRecorder::new();
        recorder.record_launch(7, BASIC, 0, 1, 3);
        recorder.record_launch(7, RECURSIVE1, 0, 1, 4);
        recorder.record_completion(7, RECURSIVE1, [1; PROOF_TIMING_SECTIONS]);
        recorder.record_completion(7, BASIC, [0; PROOF_TIMING_SECTIONS]);

        let records = recorder.take();
        let kinds: Vec<usize> = records.iter().map(|r| r.proof_type).collect();
        assert_eq!(kinds, vec![RECURSIVE1, BASIC], "records keep the order they completed in");
        let launches: Vec<_> = records.iter().map(|r| (r.id, r.airgroup_id, r.air_id, r.stream_id)).collect();
        assert_eq!(launches, vec![(7, 0, 1, 4), (7, 0, 1, 3)], "each record keeps the identity of its launch");
        let breakdowns: Vec<_> = records.iter().map(|r| r.breakdown_ms).collect();
        assert_eq!(
            breakdowns,
            vec![[1; PROOF_TIMING_SECTIONS], [0; PROOF_TIMING_SECTIONS]],
            "each record keeps the breakdown of its completion"
        );
        assert!(records.iter().all(|r| r.start_ms <= r.end_ms), "a span never ends before it starts");
        assert!(recorder.take().is_empty(), "take drains the recorder");
    }

    #[test]
    fn a_record_carries_unix_milliseconds() {
        let before = unix_ms();
        let recorder = ProofRecorder::new();
        recorder.reset();
        recorder.record_span(0, RECORD_KIND_CHALLENGE, 0, 0, Instant::now());
        let records = recorder.take();
        assert!(
            before <= records[0].start_ms && records[0].end_ms <= unix_ms() + 1,
            "a record carries the host time in Unix milliseconds"
        );
    }

    #[test]
    fn a_launcher_section_survives_the_completion() {
        let recorder = ProofRecorder::new();
        recorder.record_launch(9, RECURSIVE1, 0, 0, 2);
        let expansion_start = Instant::now();
        std::thread::sleep(Duration::from_millis(5));
        recorder.record_section(9, RECURSIVE1, PROOF_TIMING_WITNESS_EXPANSION, expansion_start);
        recorder.record_section(9, BASIC, PROOF_TIMING_WITNESS_EXPANSION, expansion_start);

        let mut breakdown = [0; PROOF_TIMING_SECTIONS];
        breakdown[0] = 7;
        recorder.record_completion(9, RECURSIVE1, breakdown);

        let records = recorder.take();
        assert_eq!(records.len(), 1, "a section for a proof with no open record is dropped");
        assert_eq!(records[0].breakdown_ms[0], 7, "the completion fills the sections it reports");
        assert!(
            records[0].breakdown_ms[PROOF_TIMING_WITNESS_EXPANSION] >= 5,
            "the completion's zero does not erase a section the launcher measured"
        );
    }

    #[test]
    fn the_head_of_a_launch_is_measured_from_the_record_it_opened() {
        let recorder = ProofRecorder::new();
        recorder.record_launch(4, BASIC, 0, 0, 1);
        std::thread::sleep(Duration::from_millis(5));
        recorder.record_launch_section(4, BASIC, PROOF_TIMING_LAUNCH_PROLOGUE, Instant::now());
        recorder.record_launch_section(4, RECURSIVE1, PROOF_TIMING_LAUNCH_PROLOGUE, Instant::now());
        recorder.record_completion(4, BASIC, [0; PROOF_TIMING_SECTIONS]);

        let records = recorder.take();
        assert_eq!(records.len(), 1, "a head for a proof with no open record is dropped");
        assert!(
            records[0].breakdown_ms[PROOF_TIMING_LAUNCH_PROLOGUE] >= 5,
            "the head runs from the launch the record opened, which no callee stamps"
        );
    }

    #[test]
    fn a_drained_completion_closes_at_the_stamp_it_reports() {
        let recorder = ProofRecorder::new();
        recorder.record_launch(3, RECORD_KIND_COMMIT, 0, 0, u32::MAX);
        let harvested_at = Instant::now();
        std::thread::sleep(Duration::from_millis(20));

        let mut breakdown = [0; PROOF_TIMING_SECTIONS];
        breakdown[1] = 4;
        recorder.record_completion_at(3, RECORD_KIND_COMMIT, breakdown, harvested_at);

        let records = recorder.take();
        assert_eq!(records.len(), 1);
        assert!(
            records[0].end_ms < records[0].start_ms + 20,
            "the record closes when the harvest reported, not when the drain ran"
        );
        assert_eq!(records[0].breakdown_ms[1], 4, "the drained completion still fills its sections");
    }

    #[test]
    fn a_section_under_a_millisecond_rounds_the_way_the_harvest_rounds() {
        let recorder = ProofRecorder::new();
        recorder.record_launch(1, RECORD_KIND_RECOMPUTE, 0, 0, u32::MAX);
        recorder.record_section_us(1, RECORD_KIND_RECOMPUTE, PROOF_TIMING_PREPARING_WC, 600);
        recorder.record_completion(1, RECORD_KIND_RECOMPUTE, [0; PROOF_TIMING_SECTIONS]);

        let records = recorder.take();
        assert_eq!(records[0].breakdown_ms[PROOF_TIMING_PREPARING_WC], 1, "0.6 ms rounds up, as the harvest does");
    }

    #[test]
    fn a_span_under_a_millisecond_keeps_one_millisecond() {
        let origin = Instant::now();
        let at = origin + Duration::from_micros(1400);
        assert_eq!(start_offset_ms(origin, at), 1, "the start rounds down");
        assert_eq!(end_offset_ms(origin, at), 2, "the end rounds up");
        assert_eq!(end_offset_ms(origin, origin), 0, "an empty span stays empty");

        let recorder = ProofRecorder::new();
        let start = Instant::now();
        std::thread::sleep(Duration::from_micros(100));
        recorder.record_span(0, RECORD_KIND_CHALLENGE, 0, 0, start);

        let records = recorder.take();
        assert!(records[0].end_ms > records[0].start_ms, "a step of any positive length keeps its bar");
    }

    #[test]
    fn the_buffer_pool_stamps_a_take_on_another_thread_and_counts_the_later_waits() {
        let pool = TimedBufferPool::new(&SlowPool);
        assert!(pool.acquired_at().is_none(), "no take, no stamp");
        std::thread::scope(|scope| {
            scope.spawn(|| {
                let _ = BufferPool::<Goldilocks>::take_buffer(&pool);
            });
        });
        let acquired = pool.acquired_at().expect("the first take stamps the pool");
        assert_eq!(pool.later_waited_us(), 0, "the first wait stays outside the record");

        let _ = BufferPool::<Goldilocks>::take_buffer(&pool);
        assert!(pool.later_waited_us() >= 5_000, "a later wait accumulates");
        assert_eq!(pool.acquired_at(), Some(acquired), "a later take keeps the stamp");
    }

    #[test]
    fn an_import_keeps_the_stamps_and_shifts_the_fold_ids() {
        let (witness, aggregation, fold) =
            (RECORD_KIND_WITNESS, RECORD_KIND_AGGREGATION_WITNESS, ProofType::Recursive2 as usize);
        let primary = ProofRecorder::new();
        primary.record_span(0, aggregation, 0, 0, Instant::now());
        primary.record_span(0, fold, 0, 0, Instant::now());
        std::thread::sleep(Duration::from_millis(20));
        let secondary = ProofRecorder::new();
        secondary.record_span(0, aggregation, 0, 0, Instant::now());
        secondary.record_span(0, fold, 0, 0, Instant::now());
        secondary.record_span(3, witness, 0, 0, Instant::now());
        assert!(primary.import(&secondary.export()));
        primary.record_span(1, fold, 0, 0, Instant::now());

        let records = primary.take();
        let ids: Vec<_> = records.iter().map(|r| (r.proof_type, r.id)).collect();
        assert_eq!(ids, [(aggregation, 0), (fold, 0), (aggregation, 1), (fold, 1), (witness, 3), (fold, 2)]);
        assert!(
            records[2..5].iter().all(|r| r.start_ms >= records[0].start_ms + 15),
            "imported records keep their stamps"
        );

        secondary.next_job();
        secondary.record_span(0, fold, 0, 0, Instant::now());
        assert!(!primary.import(&secondary.export()), "a record of another job is rejected");
        assert!(primary.take().is_empty());
    }
}
