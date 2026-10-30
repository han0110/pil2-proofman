#include "timeline.hpp"

#include <algorithm>
#include <mutex>
#include <time.h>
#include <unordered_set>
#include <vector>

std::atomic<bool> timelineRecording{false};
static std::atomic<uint64_t> timelineEpochs{0};

static std::mutex timelineMutex;
static std::unordered_set<std::string> timelineNames;
static std::vector<TimelineSpan> timelineSpans;

const char *timelineIntern(const std::string &name) {
    std::lock_guard<std::mutex> lock(timelineMutex);
    return timelineNames.insert(name).first->c_str();
}

void timelinePush(const TimelineSpan &span) {
    std::lock_guard<std::mutex> lock(timelineMutex);
    timelineSpans.push_back(span);
}

void timelineHostStop(const char *name, int64_t startNs, const char *proofType, uint64_t instanceId, uint64_t airgroupId, uint64_t airId) {
    if (startNs == 0) return;
    timelinePush({timelineIntern(name), proofType == nullptr ? nullptr : timelineIntern(proofType), instanceId, airgroupId, airId, startNs, timeline_now_ns(), 0, TIMELINE_SOURCE_HOST});
}

void set_timeline_recording(bool on) {
    timelineRecording.store(on, std::memory_order_relaxed);
    timelineEpochs.fetch_add(1, std::memory_order_relaxed);
}

uint64_t timeline_epoch() { return timelineEpochs.load(std::memory_order_relaxed); }

uint64_t take_timeline_spans(TimelineSpan *spans, uint64_t capacity) {
    std::lock_guard<std::mutex> lock(timelineMutex);
    uint64_t n = std::min<uint64_t>(capacity, timelineSpans.size());
    std::copy(timelineSpans.end() - n, timelineSpans.end(), spans);
    timelineSpans.resize(timelineSpans.size() - n);
    return n;
}

int64_t timeline_now_ns() {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (int64_t)now.tv_sec * 1000000000 + now.tv_nsec;
}
