//! Host span sink of the proving timeline. The timer macros push one span at each stop while
//! recording is on.

use std::borrow::Cow;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Instant;

static RECORDING: AtomicBool = AtomicBool::new(false);
static SPANS: Mutex<Vec<HostSpan>> = Mutex::new(Vec::new());

pub struct HostSpan {
    pub name: Cow<'static, str>,
    pub start: Instant,
    pub end: Instant,
    pub thread: String,
}

pub fn recording() -> bool {
    RECORDING.load(Ordering::Relaxed)
}

pub fn set_recording(on: bool) {
    RECORDING.store(on, Ordering::Relaxed);
}

pub fn record<N: Into<Cow<'static, str>>>(name: impl FnOnce() -> N, start: Instant) {
    if !recording() {
        return;
    }
    let end = Instant::now();
    let current = std::thread::current();
    let thread = current.name().map_or_else(|| format!("{:?}", current.id()), str::to_string);
    let span = HostSpan { name: name().into(), start, end, thread };
    SPANS.lock().unwrap_or_else(|p| p.into_inner()).push(span);
}

pub fn take() -> Vec<HostSpan> {
    std::mem::take(&mut *SPANS.lock().unwrap_or_else(|p| p.into_inner()))
}
