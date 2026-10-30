#ifndef TIMELINE_HPP
#define TIMELINE_HPP

#include <atomic>
#include <cstdint>
#include <string>

// Layout must match TimelineSpan in ffi_timeline.rs.
struct TimelineSpan {
    const char *name;
    const char *proofType;
    uint64_t instanceId;
    uint64_t airgroupId;
    uint64_t airId;
    int64_t startNs;
    int64_t endNs;
    uint32_t depth;
    uint32_t source;
};

#define TIMELINE_SOURCE_HOST 0
#define TIMELINE_SOURCE_GPU 1
#define TIMELINE_SOURCE_GPU_CATEGORY 2

extern std::atomic<bool> timelineRecording;

inline bool timeline_recording() { return timelineRecording.load(std::memory_order_relaxed); }

// Counts the calls of set_timeline_recording, one at the start of every job.
uint64_t timeline_epoch();

// The returned pointer stays valid for the life of the process.
const char *timelineIntern(const std::string &name);

void timelinePush(const TimelineSpan &span);

extern "C" {
    void set_timeline_recording(bool on);
    uint64_t take_timeline_spans(TimelineSpan *spans, uint64_t capacity);
    int64_t timeline_now_ns();
}

inline int64_t timelineHostStart() { return timeline_recording() ? timeline_now_ns() : 0; }

// Records a host span that started at startNs, the value timelineHostStart returned.
void timelineHostStop(const char *name, int64_t startNs, const char *proofType, uint64_t instanceId, uint64_t airgroupId, uint64_t airId);

#endif
