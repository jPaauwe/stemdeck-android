#include "ort.hpp"

#include <dlfcn.h>

#include <cstdlib>
#include <mutex>
#include <stdexcept>

#include "onnxruntime_c_api.h"

namespace sda {
namespace {
std::mutex g_lock;
std::string g_lib_path;
const OrtApi* g_api = nullptr;
void* g_handle = nullptr;
OrtEnv* g_env = nullptr;

const OrtApi* api() {
    std::lock_guard<std::mutex> l(g_lock);
    if (g_api) return g_api;
    std::string path = g_lib_path;
    if (path.empty()) {
        const char* e = std::getenv("SDA_ORT_LIB");
        path = e && *e ? e : "libonnxruntime.so";
    }
    void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) throw std::runtime_error(std::string("ONNX Runtime not found: ") + dlerror());
    using GetApiBase = const OrtApiBase* (*)();
    auto get = (GetApiBase)dlsym(h, "OrtGetApiBase");
    if (!get) throw std::runtime_error("ONNX Runtime has no OrtGetApiBase");
    const OrtApi* a = get()->GetApi(ORT_API_VERSION);
    if (!a) throw std::runtime_error(std::string("ONNX Runtime too old: ") + get()->GetVersionString());
    OrtEnv* env = nullptr;
    if (OrtStatus* st = a->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "stemdeck", &env)) {
        std::string m = a->GetErrorMessage(st);
        a->ReleaseStatus(st);
        throw std::runtime_error("ONNX Runtime: " + m);
    }
    g_env = env;
    g_api = a;
    g_handle = h;
    return g_api;
}

void check(OrtStatus* st) {
    if (!st) return;
    std::string m = g_api->GetErrorMessage(st);
    g_api->ReleaseStatus(st);
    throw std::runtime_error("ONNX Runtime: " + m);
}
}  // namespace

struct OrtModel::Impl {
    OrtSession* session = nullptr;
    OrtSessionOptions* options = nullptr;
    OrtMemoryInfo* mem = nullptr;
    ~Impl() {
        if (!g_api) return;
        if (session) g_api->ReleaseSession(session);
        if (options) g_api->ReleaseSessionOptions(options);
        if (mem) g_api->ReleaseMemoryInfo(mem);
    }
};

void OrtModel::set_library_path(const std::string& path) {
    std::lock_guard<std::mutex> l(g_lock);
    g_lib_path = path;
}

std::string OrtModel::version() {
    api();
    using GetApiBase = const OrtApiBase* (*)();
    auto get = (GetApiBase)dlsym(g_handle, "OrtGetApiBase");
    return get ? get()->GetVersionString() : "?";
}

OrtModel::OrtModel(const std::string& path, int threads, const std::string& provider) : impl_(new Impl) {
    const OrtApi* a = api();
    check(a->CreateSessionOptions(&impl_->options));
    if (provider == "XNNPACK") {
        // XNNPACK runs its own thread pool; ONNX Runtime's is then kept to one.
        std::string n = std::to_string(threads > 0 ? threads : 4);
        const char* keys[] = {"intra_op_num_threads"};
        const char* vals[] = {n.c_str()};
        check(a->SessionOptionsAppendExecutionProvider(impl_->options, "XNNPACK", keys, vals, 1));
        check(a->SetIntraOpNumThreads(impl_->options, 1));
    } else if (provider == "NNAPI") {
        using Fn = OrtStatus* (*)(OrtSessionOptions*, uint32_t);
        auto fn = g_handle ? (Fn)dlsym(g_handle, "OrtSessionOptionsAppendExecutionProvider_Nnapi") : nullptr;
        if (!fn) throw std::runtime_error("this ONNX Runtime has no NNAPI provider");
        check(fn(impl_->options, 0));
        if (threads > 0) check(a->SetIntraOpNumThreads(impl_->options, threads));
    } else if (threads > 0) {
        check(a->SetIntraOpNumThreads(impl_->options, threads));
    }
    check(a->SetInterOpNumThreads(impl_->options, 1));
    check(a->SetSessionGraphOptimizationLevel(impl_->options, ORT_ENABLE_ALL));
    check(a->CreateSession(g_env, path.c_str(), impl_->options, &impl_->session));
    check(a->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &impl_->mem));
    OrtAllocator* alloc = nullptr;
    check(a->GetAllocatorWithDefaultOptions(&alloc));
    size_t n = 0;
    check(a->SessionGetInputCount(impl_->session, &n));
    for (size_t i = 0; i < n; i++) {
        char* name = nullptr;
        check(a->SessionGetInputName(impl_->session, i, alloc, &name));
        in_names_.push_back(name);
        check(a->AllocatorFree(alloc, name));
    }
    check(a->SessionGetOutputCount(impl_->session, &n));
    for (size_t i = 0; i < n; i++) {
        char* name = nullptr;
        check(a->SessionGetOutputName(impl_->session, i, alloc, &name));
        out_names_.push_back(name);
        check(a->AllocatorFree(alloc, name));
    }
}

OrtModel::~OrtModel() = default;

std::vector<OrtTensor> OrtModel::run(const float* data, const std::vector<int64_t>& shape) {
    return run_named({in_names_.at(0)}, {data}, {shape});
}

std::vector<OrtTensor> OrtModel::run_named(const std::vector<std::string>& names, const std::vector<const float*>& data,
                                           const std::vector<std::vector<int64_t>>& shapes) {
    const OrtApi* a = g_api;
    std::vector<OrtValue*> inputs(names.size(), nullptr);
    std::vector<OrtValue*> outputs(out_names_.size(), nullptr);
    auto release = [&]() {
        for (auto* v : inputs) if (v) a->ReleaseValue(v);
        for (auto* v : outputs) if (v) a->ReleaseValue(v);
    };
    std::vector<OrtTensor> result;
    try {
        std::vector<const char*> in_c, out_c;
        for (size_t i = 0; i < names.size(); i++) {
            size_t count = 1;
            for (auto d : shapes[i]) count *= (size_t)d;
            check(a->CreateTensorWithDataAsOrtValue(impl_->mem, const_cast<float*>(data[i]), count * sizeof(float),
                                                    shapes[i].data(), shapes[i].size(),
                                                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[i]));
            in_c.push_back(names[i].c_str());
        }
        for (auto& n : out_names_) out_c.push_back(n.c_str());
        check(a->Run(impl_->session, nullptr, in_c.data(), inputs.data(), inputs.size(), out_c.data(),
                     out_c.size(), outputs.data()));
        for (auto* v : outputs) {
            OrtTensorTypeAndShapeInfo* info = nullptr;
            check(a->GetTensorTypeAndShape(v, &info));
            size_t dims = 0, count = 0;
            check(a->GetDimensionsCount(info, &dims));
            OrtTensor t;
            t.shape.resize(dims);
            check(a->GetDimensions(info, t.shape.data(), dims));
            check(a->GetTensorShapeElementCount(info, &count));
            a->ReleaseTensorTypeAndShapeInfo(info);
            float* p = nullptr;
            check(a->GetTensorMutableData(v, (void**)&p));
            t.data.assign(p, p + count);
            result.push_back(std::move(t));
        }
    } catch (...) {
        release();
        throw;
    }
    release();
    return result;
}

}  // namespace sda
