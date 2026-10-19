// One engine version as a self-contained plugin (libsdengine_<id>.so).
//
// The version's own engine.cpp is compiled into this file, so the plugin can
// free that version's model when it is done (g_model lives in engine.cpp's
// anonymous namespace). Every engine version since the first Android build
// has the same sdengine API: load_model, prepare_input, separate,
// request_cancel. Only the sdp_* functions below are exported; everything
// else is hidden, so several versions can be loaded in one process.
#include "engine.cpp"

#include <cstring>

#define SDP_EXPORT extern "C" __attribute__((visibility("default")))

namespace
{
void put_err(const std::string &e, char *buf, int len)
{
    if (!buf || len <= 0)
        return;
    std::strncpy(buf, e.c_str(), static_cast<size_t>(len - 1));
    buf[len - 1] = 0;
}
} // namespace

typedef int (*sdp_progress_fn)(void *ctx, float fraction);

SDP_EXPORT int sdp_abi() { return 1; }

SDP_EXPORT int sdp_load_model(const char *path, char *err, int errlen)
{
    const std::string e = sdengine::load_model(path);
    put_err(e, err, errlen);
    return e.empty() ? 1 : 0;
}

SDP_EXPORT void sdp_unload_model()
{
    std::lock_guard<std::mutex> lk(sdengine::g_model_mutex);
    sdengine::g_model.reset();
    sdengine::g_model_path.clear();
}

// Raw interleaved float32 at any rate -> stereo float32 at 44.1 kHz.
// Returns 1 ok, 0 error, -1 cancelled.
SDP_EXPORT int sdp_prepare(const char *raw, int rate, int channels, const char *out,
                           sdp_progress_fn cb, void *ctx, char *err, int errlen)
{
    const std::string e = sdengine::prepare_input(
        raw, rate, channels, out,
        [cb, ctx](float f, const char *) { return cb ? cb(ctx, f) != 0 : true; });
    if (e == "cancelled")
        return -1;
    put_err(e, err, errlen);
    return e.empty() ? 1 : 0;
}

// Writes <outdir>/<source>.wav for all six sources. rms6 receives the RMS of
// each. Returns 1 ok, 0 error, -1 cancelled.
SDP_EXPORT int sdp_separate(const char *prepared, const char *outdir, int threads,
                            int chunkSeconds, sdp_progress_fn cb, void *ctx,
                            double *rms6, char *err, int errlen)
{
    sdengine::SeparateOptions opts;
    opts.threads = threads;
    opts.chunkSeconds = chunkSeconds;
    sdengine::SeparateResult r = sdengine::separate(
        prepared, outdir, opts,
        [cb, ctx](float f, const char *) { return cb ? cb(ctx, f) != 0 : true; });
    if (r.cancelled)
        return -1;
    if (!r.ok)
    {
        put_err(r.error.empty() ? "Separation failed." : r.error, err, errlen);
        return 0;
    }
    if (rms6)
        for (int s = 0; s < sdengine::kNumSources; ++s)
            rms6[s] = r.rms[s];
    return 1;
}

SDP_EXPORT void sdp_cancel() { sdengine::request_cancel(); }
