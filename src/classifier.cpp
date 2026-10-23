#include "vision/classifier.hpp"
#include <algorithm>
#include <stdexcept>

namespace vision {
Classifier::Classifier(const std::string& model, int threads, const std::string& provider) {
    if (threads < 1) throw std::invalid_argument("threads must be positive");
    options_.SetIntraOpNumThreads(threads);
    options_.SetInterOpNumThreads(1);
    options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (provider == "cuda") {
        const auto available = Ort::GetAvailableProviders();
        if (std::find(available.begin(), available.end(), "CUDAExecutionProvider") == available.end())
            throw std::runtime_error("CUDA requested but SDK has no CUDA execution provider");
        OrtCUDAProviderOptions cuda_options{};
        cuda_options.device_id = 0;
        options_.AppendExecutionProvider_CUDA(cuda_options);
    } else if (provider != "cpu") {
        throw std::invalid_argument("provider must be cpu or cuda");
    }
    session_ = Ort::Session(env_, model.c_str(), options_);
    if (session_.GetInputCount() != 1 || session_.GetOutputCount() != 1)
        throw std::runtime_error("Expected one input and one output");
    const auto input_type = session_.GetInputTypeInfo(0);
    const auto input_info = input_type.GetTensorTypeAndShapeInfo();
    const auto shape = input_info.GetShape();
    if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        shape.size() != 4 || shape[0] != 1 || shape[1] != 3 ||
        shape[2] < 1 || shape[2] > 2048 || shape[2] != shape[3])
        throw std::runtime_error("Expected static float32 [1,3,S,S], 1 <= S <= 2048");
    std::copy(shape.begin(), shape.end(), shape_.begin());
    const auto output_type = session_.GetOutputTypeInfo(0);
    if (output_type.GetTensorTypeAndShapeInfo().GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        throw std::runtime_error("Expected float32 classifier logits");
    Ort::AllocatorWithDefaultOptions allocator;
    input_name_ = session_.GetInputNameAllocated(0, allocator).get();
    output_name_ = session_.GetOutputNameAllocated(0, allocator).get();
}

std::vector<float> Classifier::run(std::vector<float>& input) {
    const auto expected = static_cast<std::size_t>(3 * shape_[2] * shape_[3]);
    if (input.size() != expected) throw std::invalid_argument("Input buffer size mismatch");
    auto tensor = Ort::Value::CreateTensor<float>(memory_, input.data(), input.size(), shape_.data(), shape_.size());
    const char* inputs[] = {input_name_.c_str()};
    const char* outputs[] = {output_name_.c_str()};
    auto values = session_.Run(Ort::RunOptions{nullptr}, inputs, &tensor, 1, outputs, 1);
    const auto info = values[0].GetTensorTypeAndShapeInfo();
    const auto shape = info.GetShape();
    // Accommodate [1,C] and [1,C,1,1], rejecting dense prediction maps.
    if (shape.size() < 2 || shape[0] != 1 || shape[1] <= 0 ||
        !std::all_of(shape.begin() + 2, shape.end(), [](int64_t d) { return d == 1; }))
        throw std::runtime_error("Expected classifier output [1,C] or [1,C,1,...]");
    const float* data = values[0].GetTensorData<float>();
    return {data, data + info.GetElementCount()};
}
}  // namespace vision
