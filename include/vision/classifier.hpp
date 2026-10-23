#pragma once
#include <onnxruntime_cxx_api.h>
#include <array>
#include <string>
#include <vector>

namespace vision {
// One session per instance; run() reuses its caller-owned host input buffer.
class Classifier {
public:
    Classifier(const std::string& model, int threads, const std::string& provider);
    int image_size() const { return static_cast<int>(shape_[2]); }
    std::vector<float> run(std::vector<float>& input);
private:
    // Destruction order keeps Env alive longer than Session.
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "cpp-onnx-vision"};
    Ort::SessionOptions options_;
    Ort::Session session_{nullptr};
    Ort::MemoryInfo memory_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    std::array<int64_t, 4> shape_{};
    std::string input_name_, output_name_;
};
}  // namespace vision
