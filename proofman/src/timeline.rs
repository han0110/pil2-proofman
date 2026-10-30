//! Proving timeline. Each record (a Gantt bar) is an upstream timer span, a host timer macro or a
//! GPU timer, and the spans of the other timers fill it.
//!
//! Recording is on when `PROOFMAN_TIMELINE_DUMP` is set at the start of a job, or when the consumer
//! calls `proofman_util::timeline::set_recording(true)` before it. With `PROOFMAN_TIMELINE_DUMP=<path>`
//! the records not yet taken are written to `<path>` as JSON at the end of each proof. With several
//! MPI ranks, every rank sends its records to rank 0 at the end of each proof, so rank 0 holds the
//! timeline of the whole job. Every rank must record, or none. A host span outside every bar is
//! dropped.

use std::collections::HashMap;
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use proofman_common::{MpiCtx, ProofType};
use proofman_fields::PrimeField64;
use proofman_starks_lib_c::{set_timeline_recording_c, take_timeline_spans_c, timeline_now_ns_c};
use proofman_util::timeline::{self as host_sink, HostSpan};

use crate::{CancellationInfo, CancellationInfoExt, ProofMan};

pub const RECORD_KIND_WITNESS: usize = 8;
pub const RECORD_KIND_COMMIT: usize = 9;
pub const RECORD_KIND_EXECUTE: usize = 10;
pub const RECORD_KIND_PRE_CALCULATE: usize = 11;
pub const RECORD_KIND_CHALLENGE: usize = 12;
pub const RECORD_KIND_RECOMPUTE: usize = 13;
pub const RECORD_KIND_RECURSION_WITNESS: usize = 14;
pub const RECORD_KIND_AGGREGATION_WITNESS: usize = 15;
pub const RECORD_KIND_COMPRESSOR_WITNESS: usize = 16;

pub const RECORD_KIND_PER_INSTANCE: [usize; 8] = [
    ProofType::Basic as usize,
    ProofType::Compressor as usize,
    ProofType::Recursive1 as usize,
    RECORD_KIND_WITNESS,
    RECORD_KIND_COMMIT,
    RECORD_KIND_RECOMPUTE,
    RECORD_KIND_RECURSION_WITNESS,
    RECORD_KIND_COMPRESSOR_WITNESS,
];

/// The id of a record whose timer does not name its instance.
pub const RECORD_ID_UNKNOWN: u64 = u64::MAX;

pub const SPAN_SOURCE_HOST: u32 = 0;
pub const SPAN_SOURCE_GPU: u32 = 1;
pub const SPAN_SOURCE_GPU_CATEGORY: u32 = 2;

const BUFFER_MARKER: &str = "RECURSIVE2_WITNESS_BUFFER ";
const FOLD_MARKER: &str = "RECURSIVE2_FOLD ";

const DUMP_ENV: &str = "PROOFMAN_TIMELINE_DUMP";
const GATHER_TAG: usize = 999995;

static GATHER_SEQUENCE: AtomicU64 = AtomicU64::new(0);
static GATHER_SENDS: Mutex<Vec<JoinHandle<()>>> = Mutex::new(Vec::new());

/// A timer span. Times are microseconds from the origin of the job.
#[derive(Clone, Debug)]
pub struct Span {
    pub name: String,
    pub start_us: u64,
    pub end_us: u64,
    pub depth: u32,
    pub source: u32,
}

/// A bar of the timeline. `start_ms` and `end_ms` are Unix milliseconds.
#[derive(Clone, Debug)]
pub struct ProofRecord {
    pub id: u64,
    pub proof_type: usize,
    pub airgroup_id: usize,
    pub air_id: usize,
    pub start_ms: u64,
    pub end_ms: u64,
    pub start_us: u64,
    pub end_us: u64,
    pub spans: Vec<Span>,
}

struct Timeline {
    origin: Instant,
    origin_monotonic_ns: i64,
    origin_unix_us: u64,
    records: Vec<ProofRecord>,
}

impl Timeline {
    fn new() -> Self {
        Self {
            origin: Instant::now(),
            origin_monotonic_ns: timeline_now_ns_c(),
            origin_unix_us: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_micros() as u64,
            records: Vec::new(),
        }
    }

    fn collect(&mut self) {
        let (markers, host): (Vec<HostSpan>, Vec<HostSpan>) = host_sink::take()
            .into_iter()
            .partition(|s| s.name.starts_with(BUFFER_MARKER) || s.name.starts_with(FOLD_MARKER));
        let gpu = take_timeline_spans_c();
        if host.is_empty() && gpu.is_empty() {
            return;
        }
        let (origin, origin_monotonic_ns, origin_unix_us) =
            (self.origin, self.origin_monotonic_ns, self.origin_unix_us);
        let host_us = |at: Instant| at.saturating_duration_since(origin).as_micros() as u64;
        let gpu_us = |ns: i64| (ns.saturating_sub(origin_monotonic_ns).max(0) / 1000) as u64;
        let proving: Vec<&HostSpan> = host.iter().filter(|s| s.name == "GENERATING_INNER_PROOFS").collect();

        let mut records = Vec::new();
        let mut bars: HashMap<&str, Vec<(Instant, Instant, usize)>> = HashMap::new();
        let mut children = Vec::new();
        for span in &host {
            let Some((mut kind, mut id, mut airgroup_id, air_id)) = host_bar(&span.name) else {
                children.push(span);
                continue;
            };
            if kind == RECORD_KIND_WITNESS && proving.iter().any(|p| p.start <= span.start && span.end <= p.end) {
                kind = RECORD_KIND_RECOMPUTE;
            }
            if let Some((buffer_airgroup_id, buffer)) =
                witness_buffer(&markers, span).filter(|_| kind == RECORD_KIND_AGGREGATION_WITNESS)
            {
                airgroup_id = buffer_airgroup_id;
                id = fold_id(&markers, buffer, span.end).unwrap_or(RECORD_ID_UNKNOWN);
            }
            bars.entry(&span.thread).or_default().push((span.start, span.end, records.len()));
            records.push(record(id, kind, airgroup_id, air_id, host_us(span.start), host_us(span.end)));
        }

        let mut host_children: HashMap<usize, Vec<&HostSpan>> = HashMap::new();
        for span in children {
            let owner = bars.get(span.thread.as_str()).and_then(|bars| {
                bars.iter().filter(|b| b.0 <= span.start && span.end <= b.1).max_by_key(|b| b.0).map(|b| b.2)
            });
            if let Some(index) = owner {
                host_children.entry(index).or_default().push(span);
            }
        }
        for (index, spans) in host_children {
            records[index].spans = host_spans(spans, host_us);
        }
        let host_counts: Vec<usize> = records.iter().map(|r| r.spans.len()).collect();

        let mut by_key: HashMap<(usize, u64), usize> =
            records.iter().enumerate().map(|(i, r)| ((r.proof_type, r.id), i)).collect();
        for span in gpu {
            let kind = match span.proof_type {
                None => RECORD_KIND_COMMIT,
                Some(p) => ProofType::from_str(p).map_or(usize::MAX, |p| p as usize),
            };
            let (start_us, end_us) = (gpu_us(span.start_ns), gpu_us(span.end_ns));
            let index = *by_key.entry((kind, span.instance_id)).or_insert_with(|| {
                let (airgroup_id, air_id) = (span.airgroup_id as usize, span.air_id as usize);
                records.push(record(span.instance_id, kind, airgroup_id, air_id, start_us, end_us));
                records.len() - 1
            });
            let bar = &mut records[index];
            bar.start_us = bar.start_us.min(start_us);
            bar.end_us = bar.end_us.max(end_us);
            bar.spans.push(Span {
                name: span.name.to_string(),
                start_us,
                end_us,
                depth: span.depth,
                source: span.source,
            });
        }

        for (index, bar) in records.iter_mut().enumerate() {
            bar.start_ms = (origin_unix_us + bar.start_us) / 1000;
            bar.end_ms = (origin_unix_us + bar.end_us).div_ceil(1000);
            nest_categories(&mut bar.spans);
            nest_below_host(&mut bar.spans, host_counts.get(index).copied().unwrap_or(0));
            bar.spans.sort_by_key(|s| (s.start_us, s.source));
        }
        records.sort_by_key(|r| (r.start_us, r.proof_type));
        self.records.extend(records);
    }

    fn dump(&self, path: &str) {
        let spans = |spans: &[Span]| -> Vec<serde_json::Value> {
            spans
                .iter()
                .map(|s| {
                    serde_json::json!({
                        "name": s.name, "start_us": s.start_us, "end_us": s.end_us, "depth": s.depth, "source": s.source,
                    })
                })
                .collect()
        };
        let records: Vec<serde_json::Value> = self
            .records
            .iter()
            .map(|r| {
                serde_json::json!({
                    "id": r.id, "kind": kind_name(r.proof_type), "proof_type": r.proof_type,
                    "airgroup_id": r.airgroup_id, "air_id": r.air_id,
                    "start_us": r.start_us, "end_us": r.end_us, "spans": spans(&r.spans),
                })
            })
            .collect();
        let dump = serde_json::json!({ "origin_unix_ms": self.origin_unix_us / 1000, "records": records });
        if let Err(e) = std::fs::write(path, dump.to_string()) {
            tracing::warn!("Cannot write the proving timeline to {path}: {e}");
        }
    }

    /// Takes the records as words for rank 0.
    fn export(&mut self, sequence: u64) -> Vec<u64> {
        let mut words = vec![sequence, self.origin_unix_us, self.records.len() as u64];
        for r in std::mem::take(&mut self.records) {
            words.extend([r.id, r.proof_type as u64, r.airgroup_id as u64, r.air_id as u64]);
            words.extend([r.start_ms, r.end_ms, r.start_us, r.end_us]);
            push_spans(&mut words, &r.spans);
        }
        words
    }

    /// Appends the records of another rank, moved to the origin of this rank.
    fn import(&mut self, words: &[u64]) {
        let mut words = words.iter().copied();
        let mut next = move || words.next().expect("truncated timeline words");
        let _sequence = next();
        let shift = next() as i64 - self.origin_unix_us as i64;
        let rebase = |us: u64| (us as i64 + shift).max(0) as u64;
        // Fold ids restart on every rank.
        let recursive2 = ProofType::Recursive2 as usize;
        let ids = self.records.iter().filter(|r| r.proof_type == recursive2).map(|r| r.id + 1).max().unwrap_or(0);
        for _ in 0..next() {
            let (mut id, proof_type, airgroup_id, air_id) = (next(), next() as usize, next() as usize, next() as usize);
            let (start_ms, end_ms, start_us, end_us) = (next(), next(), rebase(next()), rebase(next()));
            if proof_type == recursive2 || (proof_type == RECORD_KIND_AGGREGATION_WITNESS && id != RECORD_ID_UNKNOWN) {
                id += ids;
            }
            let spans = read_spans(&mut next, rebase);
            self.records.push(ProofRecord {
                id,
                proof_type,
                airgroup_id,
                air_id,
                start_ms,
                end_ms,
                start_us,
                end_us,
                spans,
            });
        }
        self.records.sort_by_key(|r| (r.start_us, r.proof_type));
    }
}

static TIMELINE: Mutex<Option<Timeline>> = Mutex::new(None);

fn with_timeline<R>(f: impl FnOnce(&mut Timeline) -> R) -> R {
    f(TIMELINE.lock().unwrap_or_else(|p| p.into_inner()).get_or_insert_with(Timeline::new))
}

/// Sets the origin of a new job and drops what the previous job left.
pub(crate) fn start_job() {
    if std::env::var_os(DUMP_ENV).is_some() {
        host_sink::set_recording(true);
    }
    set_timeline_recording_c(host_sink::recording());
    host_sink::take();
    take_timeline_spans_c();
    *TIMELINE.lock().unwrap_or_else(|p| p.into_inner()) = Some(Timeline::new());
}

/// Collects the spans of a proof when it ends, gathers them on rank 0, and writes the dump if one is
/// asked for.
pub(crate) struct FinishOnDrop<'a>(pub &'a Arc<MpiCtx>, pub &'a RwLock<CancellationInfo>);

impl Drop for FinishOnDrop<'_> {
    fn drop(&mut self) {
        if !host_sink::recording() {
            return;
        }
        let FinishOnDrop(mpi_ctx, cancellation_info) = *self;
        let sequence = GATHER_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        with_timeline(|timeline| timeline.collect());
        if mpi_ctx.rank != 0 {
            let (mpi_ctx, words) = (mpi_ctx.clone(), with_timeline(|timeline| timeline.export(sequence)));
            let mut sends = GATHER_SENDS.lock().unwrap_or_else(|p| p.into_inner());
            sends.retain(|send| !send.is_finished());
            sends.push(std::thread::spawn(move || mpi_ctx.send_proof_to_rank(&words, GATHER_TAG, 0)));
            return;
        }
        let mut pending = mpi_ctx.n_processes - 1;
        while pending > 0 && !cancellation_info.read_recover().token.is_cancelled() {
            match mpi_ctx.check_incoming_proofs(GATHER_TAG) {
                // A word list of an earlier proof that rank 0 gave up on is dropped.
                Some(words) if words[0] == sequence => {
                    with_timeline(|timeline| timeline.import(&words));
                    pending -= 1;
                }
                Some(_) => {}
                None => std::thread::sleep(Duration::from_micros(100)),
            }
        }
        if let Some(path) = std::env::var_os(DUMP_ENV) {
            with_timeline(|timeline| timeline.dump(&path.to_string_lossy()));
        }
    }
}

impl<F: PrimeField64> ProofMan<F> {
    /// Takes the records of the timeline, sorted by start.
    pub fn take_proof_records(&self) -> Vec<ProofRecord> {
        with_timeline(|timeline| {
            timeline.collect();
            std::mem::take(&mut timeline.records)
        })
    }
}

/// Marks the witness buffer that an aggregation witness of an airgroup fills, inside its bar.
pub(crate) fn mark_witness_buffer<T>(airgroup_id: usize, trace: &[T]) {
    host_sink::record(|| format!("{BUFFER_MARKER}{airgroup_id} {:p}", trace.as_ptr()), Instant::now());
}

/// Marks the fold that takes a witness buffer.
pub(crate) fn mark_fold<T>(id: usize, trace: &[T]) {
    host_sink::record(|| format!("{FOLD_MARKER}{id} {:p}", trace.as_ptr()), Instant::now());
}

/// The airgroup and the witness buffer of an aggregation witness bar, from the buffer marker inside
/// the bar.
fn witness_buffer<'a>(markers: &'a [HostSpan], bar: &HostSpan) -> Option<(usize, &'a str)> {
    markers.iter().find_map(|m| {
        let (airgroup_id, buffer) = m.name.strip_prefix(BUFFER_MARKER)?.split_once(' ')?;
        let inside = m.thread == bar.thread && bar.start <= m.start && m.end <= bar.end;
        inside.then_some((airgroup_id.parse().ok()?, buffer))
    })
}

/// The fold of a witness buffer, from the first fold marker of the buffer after the witness ends,
/// because the pool gives the buffer out again only after the fold takes it.
fn fold_id(markers: &[HostSpan], buffer: &str, end: Instant) -> Option<u64> {
    let (_, id) = markers
        .iter()
        .filter_map(|m| {
            let (id, fold_buffer) = m.name.strip_prefix(FOLD_MARKER)?.split_once(' ')?;
            (fold_buffer == buffer && end <= m.start).then_some((m.start, id))
        })
        .min_by_key(|&(start, _)| start)?;
    id.parse().ok()
}

fn push_str(words: &mut Vec<u64>, text: &str) {
    words.push(text.len() as u64);
    words.extend(text.as_bytes().chunks(8).map(|chunk| {
        let mut bytes = [0u8; 8];
        bytes[..chunk.len()].copy_from_slice(chunk);
        u64::from_le_bytes(bytes)
    }));
}

fn read_str(next: &mut impl FnMut() -> u64) -> String {
    let len = next() as usize;
    let bytes: Vec<u8> = (0..len.div_ceil(8)).flat_map(|_| next().to_le_bytes()).take(len).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn push_spans(words: &mut Vec<u64>, spans: &[Span]) {
    words.push(spans.len() as u64);
    for s in spans {
        push_str(words, &s.name);
        words.extend([s.start_us, s.end_us, s.depth as u64, s.source as u64]);
    }
}

fn read_spans(next: &mut impl FnMut() -> u64, rebase: impl Fn(u64) -> u64) -> Vec<Span> {
    (0..next())
        .map(|_| {
            let name = read_str(next);
            let (start_us, end_us, depth, source) = (rebase(next()), rebase(next()), next() as u32, next() as u32);
            Span { name, start_us, end_us, depth, source }
        })
        .collect()
}

fn record(id: u64, kind: usize, airgroup_id: usize, air_id: usize, start_us: u64, end_us: u64) -> ProofRecord {
    ProofRecord {
        id,
        proof_type: kind,
        airgroup_id,
        air_id,
        start_ms: 0,
        end_ms: 0,
        start_us,
        end_us,
        spans: Vec::new(),
    }
}

/// The record that an upstream host timer brackets, as (kind, id, airgroup, air).
fn host_bar(name: &str) -> Option<(usize, u64, usize, usize)> {
    match name {
        "EXECUTE" => return Some((RECORD_KIND_EXECUTE, 0, 0, 0)),
        "PRE_CALCULATE_WC" => return Some((RECORD_KIND_PRE_CALCULATE, 0, 0, 0)),
        "CALCULATE_INTERNAL_CONTRIBUTION" => return Some((RECORD_KIND_CHALLENGE, 0, 0, 0)),
        "GENERATE_WITNESS_AGGREGATION" => return Some((RECORD_KIND_AGGREGATION_WITNESS, RECORD_ID_UNKNOWN, 0, 0)),
        "GENERATE_VADCOP_FINAL_PROOF" => return Some((ProofType::VadcopFinal as usize, 0, 0, 0)),
        "GENERATE_VADCOP_FINAL_COMPRESSED_PROOF" => return Some((ProofType::VadcopFinalCompressed as usize, 0, 0, 0)),
        _ => {}
    }
    const PER_INSTANCE: [(&str, usize); 4] = [
        ("GENERATING_WC_", RECORD_KIND_WITNESS),
        ("GET_CONTRIBUTION_AIR_", RECORD_KIND_COMMIT),
        ("GENERATING_COMPRESSOR_WITNESS_", RECORD_KIND_COMPRESSOR_WITNESS),
        ("GENERATING_RECURSIVE1_WITNESS_", RECORD_KIND_RECURSION_WITNESS),
    ];
    let (rest, kind) = PER_INSTANCE.iter().find_map(|(prefix, kind)| Some((name.strip_prefix(prefix)?, *kind)))?;
    // The rest is "<id> [<airgroup>:<air>]".
    let (id, air) = rest.split_once(" [")?;
    let (airgroup_id, air_id) = air.strip_suffix(']')?.split_once(':')?;
    Some((kind, id.parse().ok()?, airgroup_id.parse().ok()?, air_id.parse().ok()?))
}

/// Host spans with their nesting depth, sorted by start.
fn host_spans(mut spans: Vec<&HostSpan>, host_us: impl Fn(Instant) -> u64) -> Vec<Span> {
    spans.sort_by_key(|s| (s.start, std::cmp::Reverse(s.end)));
    let mut open: Vec<Instant> = Vec::new();
    spans
        .into_iter()
        .map(|s| {
            while open.last().is_some_and(|&end| end <= s.start) {
                open.pop();
            }
            let depth = open.len() as u32;
            open.push(s.end);
            Span {
                name: s.name.to_string(),
                start_us: host_us(s.start),
                end_us: host_us(s.end),
                depth,
                source: SPAN_SOURCE_HOST,
            }
        })
        .collect()
}

/// A GPU category sits one level below the deepest GPU section that contains it.
fn nest_categories(spans: &mut [Span]) {
    let sections: Vec<(u64, u64, u32)> =
        spans.iter().filter(|s| s.source == SPAN_SOURCE_GPU).map(|s| (s.start_us, s.end_us, s.depth)).collect();
    for span in spans.iter_mut().filter(|s| s.source == SPAN_SOURCE_GPU_CATEGORY) {
        span.depth = sections
            .iter()
            .filter(|&&(start, end, _)| start <= span.start_us && span.end_us <= end)
            .map(|&(_, _, depth)| depth + 1)
            .max()
            .unwrap_or(0);
    }
}

/// The spans of the GPU timers and of the host steps around them sit below the first `host` spans
/// of a bar, its upstream host timers, which wrap the GPU calls they wait for.
fn nest_below_host(spans: &mut [Span], host: usize) {
    let below_host = spans[..host].iter().map(|s| s.depth + 1).max().unwrap_or(0);
    for span in &mut spans[host..] {
        span.depth += below_host;
    }
}

fn kind_name(kind: usize) -> &'static str {
    const PROOF_TYPES: [ProofType; 8] = [
        ProofType::Basic,
        ProofType::Compressor,
        ProofType::Recursive1,
        ProofType::Recursive2,
        ProofType::VadcopFinal,
        ProofType::VadcopFinalCompressed,
        ProofType::RecursiveF,
        ProofType::RecurserAggregator,
    ];
    const KINDS: [&str; 9] = [
        "witness",
        "commit",
        "execute",
        "pre_calculate",
        "challenge",
        "recompute",
        "recursion_witness",
        "aggregation_witness",
        "compressor_witness",
    ];
    match kind {
        0..8 => PROOF_TYPES[kind].into(),
        8..17 => KINDS[kind - 8],
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_bar_reads_the_upstream_timer_names() {
        let cases = [
            ("EXECUTE", Some((RECORD_KIND_EXECUTE, 0, 0, 0))),
            ("GENERATING_WC_12 [0:3]", Some((RECORD_KIND_WITNESS, 12, 0, 3))),
            ("GET_CONTRIBUTION_AIR_7 [1:20]", Some((RECORD_KIND_COMMIT, 7, 1, 20))),
            ("GENERATING_RECURSIVE1_WITNESS_4 [0:2]", Some((RECORD_KIND_RECURSION_WITNESS, 4, 0, 2))),
            ("GENERATE_WITNESS_AGGREGATION", Some((RECORD_KIND_AGGREGATION_WITNESS, RECORD_ID_UNKNOWN, 0, 0))),
            ("GENERATE_VADCOP_FINAL_PROOF", Some((ProofType::VadcopFinal as usize, 0, 0, 0))),
            ("GENERATE_VADCOP_FINAL_COMPRESSED_PROOF", Some((ProofType::VadcopFinalCompressed as usize, 0, 0, 0))),
            ("GENERATING_COMPRESSOR_WITNESS_4 [0:2]", Some((RECORD_KIND_COMPRESSOR_WITNESS, 4, 0, 2))),
            ("PREPARING_WC_12 [0:3]", None),
            ("GENERATING_WC_x [0:3]", None),
        ];
        for (name, bar) in cases {
            assert_eq!(host_bar(name), bar, "{name}");
        }
    }

    #[test]
    fn rank_zero_imports_another_rank_on_its_own_origin() {
        let timeline = |origin_unix_us| Timeline {
            origin: Instant::now(),
            origin_monotonic_ns: 0,
            origin_unix_us,
            records: Vec::new(),
        };
        let span = Span { name: "STARK_GPU_PROOF".into(), start_us: 10, end_us: 20, depth: 0, source: SPAN_SOURCE_GPU };
        let mut sender = timeline(1_000_500);
        let recursive2 = ProofType::Recursive2 as usize;
        sender.records.push(ProofRecord { spans: vec![span], ..record(0, recursive2, 0, 0, 5, 30) });
        sender.records.push(record(0, RECORD_KIND_AGGREGATION_WITNESS, 0, 0, 1, 4));
        let mut receiver = timeline(1_000_000);
        receiver.records.push(record(0, recursive2, 0, 0, 0, 1));

        receiver.import(&sender.export(3));
        assert!(sender.records.is_empty(), "export takes what it sends");
        let imported = receiver.records.iter().find(|r| r.proof_type == recursive2 && r.start_us == 505).unwrap();
        assert_eq!((imported.id, imported.end_us), (1, 530), "ids move past rank 0 ones");
        assert_eq!((imported.spans[0].name.as_str(), imported.spans[0].start_us), ("STARK_GPU_PROOF", 510));
        let witness = receiver.records.iter().find(|r| r.proof_type == RECORD_KIND_AGGREGATION_WITNESS).unwrap();
        assert_eq!(witness.id, 1, "the witness keeps the id of its fold");
    }

    #[test]
    fn aggregation_witness_takes_the_first_fold_of_its_buffer() {
        let origin = Instant::now();
        let at = |us: u64| origin + std::time::Duration::from_micros(us);
        let span = |name: &str, thread: &str, start, end| HostSpan {
            name: name.to_string().into(),
            start: at(start),
            end: at(end),
            thread: thread.into(),
        };
        let bar = span("GENERATE_WITNESS_AGGREGATION", "a", 0, 10);
        let markers = [
            span("RECURSIVE2_WITNESS_BUFFER 0 0x10", "b", 5, 5),
            span("RECURSIVE2_WITNESS_BUFFER 1 0x20", "a", 9, 9),
            span("RECURSIVE2_FOLD 3 0x20", "c", 40, 40),
            span("RECURSIVE2_FOLD 2 0x20", "c", 12, 12),
            span("RECURSIVE2_FOLD 1 0x10", "c", 11, 11),
        ];
        let (airgroup_id, buffer) = witness_buffer(&markers, &bar).unwrap();
        assert_eq!((airgroup_id, buffer), (1, "0x20"), "the marker of the bar names its airgroup and buffer");
        assert_eq!(fold_id(&markers, buffer, bar.end), Some(2));
        assert_eq!(fold_id(&markers[..2], buffer, bar.end), None, "a buffer no fold takes names no fold");
    }

    #[test]
    fn gpu_spans_and_their_host_steps_nest_below_the_upstream_host_timers() {
        let span =
            |name: &str, start_us, end_us, depth, source| Span { name: name.into(), start_us, end_us, depth, source };
        let mut spans = [
            span("GEN_RECURSIVE_PROOF", 10, 100, 0, SPAN_SOURCE_HOST),
            span("CIRCOM_WITNESS", 0, 9, 1, SPAN_SOURCE_HOST),
            span("STARK_GPU_WITNESS", 9, 40, 0, SPAN_SOURCE_GPU),
            span("NTT", 20, 30, 1, SPAN_SOURCE_GPU_CATEGORY),
            span("TIMELINE_EXPORT", 40, 60, 0, SPAN_SOURCE_HOST),
        ];
        nest_below_host(&mut spans, 2);
        let depths: Vec<_> = spans.iter().map(|s| (s.name.as_str(), s.depth)).collect();
        let want = [
            ("GEN_RECURSIVE_PROOF", 0),
            ("CIRCOM_WITNESS", 1),
            ("STARK_GPU_WITNESS", 2),
            ("NTT", 3),
            ("TIMELINE_EXPORT", 2),
        ];
        assert_eq!(depths, want, "the export keeps its rank against the GPU span it follows");
    }

    #[test]
    fn spans_nest_by_containment() {
        let origin = Instant::now();
        let at = |us: u64| origin + std::time::Duration::from_micros(us);
        let span = |name: &'static str, start, end| HostSpan {
            name: name.into(),
            start: at(start),
            end: at(end),
            thread: String::new(),
        };
        let (outer, inner, next) = (span("OUTER", 0, 10), span("INNER", 2, 5), span("NEXT", 10, 12));
        let spans = host_spans(vec![&next, &inner, &outer], |t| t.duration_since(origin).as_micros() as u64);
        let depths: Vec<_> = spans.iter().map(|s| (s.name.as_str(), s.depth)).collect();
        assert_eq!(depths, [("OUTER", 0), ("INNER", 1), ("NEXT", 0)]);
    }
}
