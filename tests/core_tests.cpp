#include "vision/image.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {
void check(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
template<class Function> void rejects(Function fn) {
    bool threw = false;
    try { fn(); } catch (const std::exception&) { threw = true; }
    check(threw, "Expected invalid input to be rejected");
}
}
int main() {
    try {
        // The first raster byte is newline: never consume it with std::ws.
        std::string bytes = "P6\n# comment\n1 1\n255\n";
        bytes += std::string{"\n\x20\xff", 3};
        std::istringstream stream(bytes);
        const auto image = vision::read_ppm(stream);
        check(image.rgb[0] == 10 && image.rgb[1] == 32 && image.rgb[2] == 255, "PPM raster damaged");
        const auto input = vision::preprocess({1,1,{255,0,0}}, 2);
        check(input.size() == 12, "Tensor shape");
        check(std::abs(input[0] - (1.f-.485f)/.229f) < 1e-5f, "Red normalisation");
        check(std::abs(input[4] - (-.456f)/.224f) < 1e-5f, "Green plane");
        check(std::abs(input[8] - (-.406f)/.225f) < 1e-5f, "Blue plane");
        const auto p = vision::softmax({10000,10001,10002});
        check(std::abs(p[0]+p[1]+p[2]-1.f) < 1e-6f && p[2]>p[1], "Stable softmax");
        check(vision::percentile({9,1,5}, .5) == 5, "Median");
        check(std::abs(vision::percentile({1,2,3,4,5}, .95)-4.8) < 1e-12, "Interpolated p95");
        rejects([] { std::istringstream s("P6\n2 2\n255\nx"); vision::read_ppm(s); });
        rejects([] { std::istringstream s("P6\n-1 2\n255\n"); vision::read_ppm(s); });
        rejects([] { std::istringstream s("P6\n99999 2\n255\n"); vision::read_ppm(s); });
        rejects([] { std::istringstream s("P3\n1 1\n255\n0 0 0"); vision::read_ppm(s); });
        rejects([] { vision::preprocess({2,2,{0}}, 2); });
        rejects([] { vision::softmax({std::numeric_limits<float>::infinity()}); });
        rejects([] { vision::softmax({}); });
        rejects([] { vision::percentile({}, .5); });
        std::cout << "core checks passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
