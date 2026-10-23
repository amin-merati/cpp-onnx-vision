#include "vision/classifier.hpp"
#include "vision/image.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>

namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
int number(const std::string& text, int minimum, int maximum) {
    std::size_t used = 0;
    const int value = std::stoi(text, &used);
    if (used != text.size() || value < minimum || value > maximum)
        throw std::invalid_argument("Numeric option out of range: " + text);
    return value;
}
void help() {
    std::cout << "vision_infer --model MODEL.onnx --image IMAGE.ppm [options]\n"
              << "  --runs 100 --warmup 10 --threads 1 --provider cpu|cuda\n"
              << "  --output result.json  --dump-input tensor.f32\n"
              << "P6 RGB input; static float32 [1,3,S,S] classifier. JSON to stdout by default.\n";
}
}

int main(int argc, char** argv) {
    try {
        std::map<std::string, std::string> args{{"--runs","100"},{"--warmup","10"},
                                               {"--threads","1"},{"--provider","cpu"}};
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--help") { help(); return 0; }
            if (key != "--model" && key != "--image" && key != "--runs" && key != "--warmup" &&
                key != "--threads" && key != "--provider" && key != "--output" && key != "--dump-input")
                throw std::invalid_argument("Unknown option: " + key);
            if (++i >= argc) throw std::invalid_argument("Missing value for " + key);
            args[key] = argv[i];
        }
        if (!args.count("--model") || !args.count("--image")) { help(); return 2; }
        const int runs = number(args.at("--runs"), 1, 1000000);
        const int warmup = number(args.at("--warmup"), 0, 1000000);
        const int threads = number(args.at("--threads"), 1, 1024);
        const auto load_start = Clock::now();
        vision::Classifier classifier(args.at("--model"), threads, args.at("--provider"));
        const double load_ms = elapsed(load_start);
        const auto prep_start = Clock::now();
        std::ifstream image_file(args.at("--image"), std::ios::binary);
        if (!image_file) throw std::runtime_error("Cannot open image");
        auto input = vision::preprocess(vision::read_ppm(image_file), classifier.image_size());
        const double preprocess_ms = elapsed(prep_start);
        if (args.count("--dump-input")) {
            std::ofstream dump(args.at("--dump-input"), std::ios::binary);
            dump.write(reinterpret_cast<const char*>(input.data()),
                       static_cast<std::streamsize>(input.size() * sizeof(float)));
            if (!dump) throw std::runtime_error("Cannot write input tensor");
        }
        for (int i = 0; i < warmup; ++i) classifier.run(input);
        std::vector<double> timings;
        timings.reserve(static_cast<std::size_t>(runs));
        std::vector<float> logits;
        for (int i = 0; i < runs; ++i) {
            const auto start = Clock::now();
            logits = classifier.run(input);
            timings.push_back(elapsed(start));
        }
        const auto probabilities = vision::softmax(logits);
        std::vector<std::size_t> order(logits.size());
        std::iota(order.begin(), order.end(), 0);
        const auto top_count = std::min<std::size_t>(5, order.size());
        std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(top_count), order.end(),
            [&](auto a, auto b) { return logits[a] == logits[b] ? a < b : logits[a] > logits[b]; });
        std::ofstream file;
        if (args.count("--output")) { file.open(args.at("--output")); if (!file) throw std::runtime_error("Cannot open output"); }
        std::ostream& out = args.count("--output") ? file : std::cout;
        out << std::setprecision(9) << "{\n  \"ort_version\": \"" << Ort::GetVersionString()
            << "\",\n  \"requested_provider\": \"" << args.at("--provider") << "\",\n  \"threads\": " << threads
            << ",\n  \"runs\": " << runs << ",\n  \"warmup\": " << warmup
            << ",\n  \"image_size\": " << classifier.image_size()
            << ",\n  \"session_load_ms\": " << load_ms << ",\n  \"preprocess_ms\": " << preprocess_ms
            << ",\n  \"inference_median_ms\": " << vision::percentile(timings, .5)
            << ",\n  \"inference_p95_ms\": " << vision::percentile(timings, .95)
            << ",\n  \"top5\": [";
        for (std::size_t i = 0; i < top_count; ++i) {
            if (i) out << ',';
            out << "{\"class_id\":" << order[i] << ",\"probability\":" << probabilities[order[i]] << '}';
        }
        out << "],\n  \"logits\": [";
        for (std::size_t i = 0; i < logits.size(); ++i) { if (i) out << ','; out << logits[i]; }
        out << "]\n}\n";
        if (!out) throw std::runtime_error("Output write failed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
