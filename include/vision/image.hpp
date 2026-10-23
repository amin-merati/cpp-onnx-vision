#pragma once
#include <cstdint>
#include <istream>
#include <vector>

namespace vision {
struct Image {
    int width;
    int height;
    std::vector<std::uint8_t> rgb;
};
// P6 RGB, maxval=255. Bounded dimensions and exact raster-length checks.
Image read_ppm(std::istream& input);
// Aspect-preserving bilinear resize (half-pixel convention), centre crop,
// RGB -> float32 NCHW, ImageNet normalisation. Model input must be square.
std::vector<float> preprocess(const Image& image, int size);
std::vector<float> softmax(const std::vector<float>& logits);
double percentile(std::vector<double> samples, double fraction);
}  // namespace vision
