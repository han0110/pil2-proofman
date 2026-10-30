#ifndef TIMELINE_GPU_CUH
#define TIMELINE_GPU_CUH

#include <map>
#include <mutex>
#include "gpu_timer.cuh"
#include "timeline.hpp"

// An idle stream per device, which carries the reference events.
inline cudaStream_t timelineMarkerStream(int device) {
    static std::mutex mutex;
    static std::map<int, cudaStream_t> streams;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = streams.find(device);
    if (it != streams.end()) return it->second;
    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    return streams[device] = stream;
}

// The reference event of a device and the CLOCK_MONOTONIC time it completed at, taken once per
// recording epoch: a marker can queue behind the work of other streams, so an export reads its
// events against this one instead of recording its own.
inline bool timelineReference(int device, cudaEvent_t &event, int64_t &ns) {
    struct Reference {
        uint64_t epoch = 0;
        cudaEvent_t event = nullptr;
        int64_t ns = 0;
    };
    static std::mutex mutex;
    static std::map<int, Reference> references;
    std::lock_guard<std::mutex> lock(mutex);
    Reference &reference = references[device];
    if (reference.event == nullptr || reference.epoch != timeline_epoch()) {
        cudaStream_t stream = timelineMarkerStream(device);
        if (reference.event == nullptr && cudaEventCreate(&reference.event) != cudaSuccess) reference.event = nullptr;
        if (stream == nullptr || reference.event == nullptr || cudaEventRecord(reference.event, stream) != cudaSuccess ||
            cudaEventSynchronize(reference.event) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        reference.ns = timeline_now_ns();
        reference.epoch = timeline_epoch();
    }
    event = reference.event;
    ns = reference.ns;
    return true;
}

// Exports every section and category entry of the timer as CLOCK_MONOTONIC spans. Call it before
// the timer is cleared. proofType is nullptr for a commit.
inline void timelineExport(TimerGPU &timer, const char *proofType, uint64_t instanceId, uint64_t airgroupId, uint64_t airId) {
    if (!timeline_recording() || timer.device < 0 || (timer.timers.empty() && timer.multiTimers.empty())) return;
    int prev = -1;
    const bool switched = cudaGetDevice(&prev) == cudaSuccess && prev != timer.device && cudaSetDevice(timer.device) == cudaSuccess;
    cudaEvent_t reference;
    int64_t referenceNs = 0;
    if (!timelineReference(timer.device, reference, referenceNs)) {
        if (switched) cudaSetDevice(prev);
        return;
    }
    auto toNs = [&](cudaEvent_t event, int64_t &ns) {
        float ms = 0.0f;
        if (cudaEventSynchronize(event) != cudaSuccess || cudaEventElapsedTime(&ms, reference, event) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        ns = referenceNs + (int64_t)((double)ms * 1e6);
        return true;
    };
    TimelineSpan span = {nullptr, proofType == nullptr ? nullptr : timelineIntern(proofType), instanceId, airgroupId, airId, 0, 0, 0, TIMELINE_SOURCE_GPU};
    for (auto &[name, entry] : timer.timers) {
        span.name = timelineIntern(name);
        span.depth = entry.depth;
        if (toNs(entry.start, span.startNs) && toNs(entry.stop, span.endNs)) timelinePush(span);
    }
    span.source = TIMELINE_SOURCE_GPU_CATEGORY;
    span.depth = 0;
    for (auto &[name, entries] : timer.multiTimers) {
        span.name = timelineIntern(name);
        for (auto &entry : entries) {
            if (toNs(entry.start, span.startNs) && toNs(entry.stop, span.endNs)) timelinePush(span);
        }
    }
    if (switched) cudaSetDevice(prev);
}

#endif
