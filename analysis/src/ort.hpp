// ONNX Runtime through its C API, loaded at run time with dlopen: on the
// phone it is the libonnxruntime.so the onnxruntime-android AAR packages, on
// a computer the one from ONNX Runtime's release archive (SDA_ORT_LIB). The
// same analysis code runs in both places.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sda {

struct OrtTensor {
    std::vector<int64_t> shape;
    std::vector<float> data;
};

class OrtModel {
   public:
    // Throws std::runtime_error with ONNX Runtime's message on failure.
    // `provider`: "" (CPU), "XNNPACK" or "NNAPI" (for measuring them; the app uses the CPU).
    OrtModel(const std::string& path, int threads, const std::string& provider = "");
    ~OrtModel();
    OrtModel(const OrtModel&) = delete;
    OrtModel& operator=(const OrtModel&) = delete;

    // One float input, float outputs (in the model's output order).
    std::vector<OrtTensor> run(const float* data, const std::vector<int64_t>& shape);
    std::vector<OrtTensor> run_named(const std::vector<std::string>& in_names, const std::vector<const float*>& data,
                                     const std::vector<std::vector<int64_t>>& shapes);
    const std::vector<std::string>& input_names() const { return in_names_; }
    const std::vector<std::string>& output_names() const { return out_names_; }

    // Where libonnxruntime.so is looked for; empty means the default search path.
    static void set_library_path(const std::string& path);
    static std::string version();

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::vector<std::string> in_names_, out_names_;
};

}  // namespace sda
