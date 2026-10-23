#include "vision/image.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <string>

namespace vision {
namespace {
std::string token(std::istream& in) {
    std::string value;
    while (in) {
        in >> std::ws;
        if (in.peek() != '#') break;
        in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    if (!(in >> value)) throw std::runtime_error("Incomplete PPM header");
    return value;
}
int dimension(const std::string& value) {
    std::size_t end = 0;
    const int result = std::stoi(value, &end);
    if (end != value.size() || result < 1 || result > 8192)
        throw std::runtime_error("Image dimensions must be between 1 and 8192");
    return result;
}
}  // namespace

Image read_ppm(std::istream& in) {
    if (token(in) != "P6") throw std::runtime_error("Expected binary RGB PPM (P6)");
    const int width = dimension(token(in));
    const int height = dimension(token(in));
    if (token(in) != "255") throw std::runtime_error("PPM maxval must be 255");
    // Consume only the raster separator: pixel values may themselves be whitespace.
    const int separator = in.get();
    if (separator == EOF || !std::isspace(static_cast<unsigned char>(separator)))
        throw std::runtime_error("Missing PPM raster separator");
    if (separator == '\r' && in.peek() == '\n') in.get();
    Image image{width, height, std::vector<std::uint8_t>(
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3)};
    const auto bytes = static_cast<std::streamsize>(image.rgb.size());
    in.read(reinterpret_cast<char*>(image.rgb.data()), bytes);
    if (in.gcount() != bytes) throw std::runtime_error("Truncated PPM raster");
    if (in.peek() != EOF) throw std::runtime_error("Trailing data: expected a single PPM image");
    return image;
}

std::vector<float> preprocess(const Image& image, int size) {
    if (size < 1 || size > 2048 || image.width < 1 || image.height < 1 ||
        image.width > 8192 || image.height > 8192 ||
        image.rgb.size() != static_cast<std::size_t>(image.width) *
                            static_cast<std::size_t>(image.height) * 3)
        throw std::invalid_argument("Invalid image or model image size");
    const int short_side = static_cast<int>(std::lround(size * (256.0 / 224.0)));
    const double scale = static_cast<double>(short_side) / std::min(image.width, image.height);
    const int resized_w = static_cast<int>(std::lround(image.width * scale));
    const int resized_h = static_cast<int>(std::lround(image.height * scale));
    const int offset_x = (resized_w - size) / 2;
    const int offset_y = (resized_h - size) / 2;
    const auto plane = static_cast<std::size_t>(size) * static_cast<std::size_t>(size);
    std::vector<float> tensor(3 * plane);
    constexpr std::array<double, 3> mean{0.485, 0.456, 0.406};
    constexpr std::array<double, 3> stddev{0.229, 0.224, 0.225};
    // Sample only the crop; avoid an intermediate resized RGB allocation.
    for (int y = 0; y < size; ++y) {
        const double sy = std::clamp((y + offset_y + 0.5) * image.height / resized_h - 0.5,
                                     0.0, static_cast<double>(image.height - 1));
        const int y0 = static_cast<int>(sy), y1 = std::min(y0 + 1, image.height - 1);
        const double wy = sy - y0;
        for (int x = 0; x < size; ++x) {
            const double sx = std::clamp((x + offset_x + 0.5) * image.width / resized_w - 0.5,
                                         0.0, static_cast<double>(image.width - 1));
            const int x0 = static_cast<int>(sx), x1 = std::min(x0 + 1, image.width - 1);
            const double wx = sx - x0;
            for (std::size_t c = 0; c < 3; ++c) {
                auto pixel = [&](int px, int py) {
                    return image.rgb[(static_cast<std::size_t>(py) *
                                      static_cast<std::size_t>(image.width) +
                                      static_cast<std::size_t>(px)) * 3 + c];
                };
                const double value = (1-wy)*((1-wx)*pixel(x0,y0)+wx*pixel(x1,y0)) +
                                           wy*((1-wx)*pixel(x0,y1)+wx*pixel(x1,y1));
                tensor[c * plane + static_cast<std::size_t>(y * size + x)] =
                    static_cast<float>((value / 255.0 - mean[c]) / stddev[c]);
            }
        }
    }
    return tensor;
}

std::vector<float> softmax(const std::vector<float>& logits) {
    if (logits.empty()) throw std::invalid_argument("Empty logits");
    for (float v : logits) if (!std::isfinite(v)) throw std::runtime_error("Non-finite model output");
    const double maximum = *std::max_element(logits.begin(), logits.end());
    std::vector<float> result(logits.size());
    double sum = 0;
    for (float value : logits) sum += std::exp(static_cast<double>(value) - maximum);
    for (std::size_t i = 0; i < logits.size(); ++i)
        result[i] = static_cast<float>(std::exp(static_cast<double>(logits[i]) - maximum) / sum);
    return result;
}

double percentile(std::vector<double> samples, double fraction) {
    if (samples.empty() || !std::isfinite(fraction) || fraction < 0 || fraction > 1)
        throw std::invalid_argument("Invalid percentile input");
    std::sort(samples.begin(), samples.end());
    const double index = fraction * static_cast<double>(samples.size() - 1);
    const auto lower = static_cast<std::size_t>(index);
    const auto upper = std::min(lower + 1, samples.size() - 1);
    return samples[lower] + (index - static_cast<double>(lower)) * (samples[upper] - samples[lower]);
}
}  // namespace vision
