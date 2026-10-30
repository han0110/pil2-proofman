use std::ffi::{c_char, CStr};

/// Layout must match TimelineSpan in pil2-stark/src/utils/timeline.hpp.
#[repr(C)]
#[derive(Clone, Copy)]
struct TimelineSpan {
    name: *const c_char,
    proof_type: *const c_char,
    instance_id: u64,
    airgroup_id: u64,
    air_id: u64,
    start_ns: i64,
    end_ns: i64,
    depth: u32,
    source: u32,
}

extern "C" {
    fn set_timeline_recording(on: bool);
    fn take_timeline_spans(spans: *mut TimelineSpan, capacity: u64) -> u64;
    fn timeline_now_ns() -> i64;
}

/// A span of a GPU timer, in `CLOCK_MONOTONIC` nanoseconds. `proof_type` is `None` for a commit.
pub struct GpuSpan {
    pub name: &'static str,
    pub proof_type: Option<&'static str>,
    pub instance_id: u64,
    pub airgroup_id: u64,
    pub air_id: u64,
    pub start_ns: i64,
    pub end_ns: i64,
    pub depth: u32,
    pub source: u32,
}

pub fn set_timeline_recording_c(on: bool) {
    unsafe { set_timeline_recording(on) }
}

pub fn timeline_now_ns_c() -> i64 {
    unsafe { timeline_now_ns() }
}

pub fn take_timeline_spans_c() -> Vec<GpuSpan> {
    // The library interns the names for the life of the process.
    let interned = |name: *const c_char| unsafe { CStr::from_ptr(name) }.to_str().unwrap_or("?");
    let mut spans = Vec::new();
    let mut batch = vec![unsafe { std::mem::zeroed::<TimelineSpan>() }; 4096];
    loop {
        let n = unsafe { take_timeline_spans(batch.as_mut_ptr(), batch.len() as u64) } as usize;
        spans.extend(batch[..n].iter().map(|s| GpuSpan {
            name: interned(s.name),
            proof_type: (!s.proof_type.is_null()).then(|| interned(s.proof_type)),
            instance_id: s.instance_id,
            airgroup_id: s.airgroup_id,
            air_id: s.air_id,
            start_ns: s.start_ns,
            end_ns: s.end_ns,
            depth: s.depth,
            source: s.source,
        }));
        if n < batch.len() {
            return spans;
        }
    }
}
