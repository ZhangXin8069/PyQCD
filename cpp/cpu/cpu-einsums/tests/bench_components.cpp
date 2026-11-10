#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string_view>
#include <utility>
#include <vector>

#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/ComputeGraph/Operations.hpp>

namespace {

template <typename T, int R, std::size_t... I>
auto make_impl(const char* name, const std::size_t* shape,
               std::index_sequence<I...>) {
    return einsums::create_tensor<T>(true, name, shape[I]...);
}

template <typename T, int R>
auto make(const char* name, const std::size_t* shape) {
    return make_impl<T, R>(name, shape, std::make_index_sequence<R>{});
}

template <typename Fn>
double median_ms(Fn fn, int warmup, int reps) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    std::vector<double> samples;
    samples.reserve(reps);
    for (int i = 0; i < reps; ++i) {
        auto start = std::chrono::steady_clock::now();
        fn();
        samples.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

template <typename FnEngine, typename FnAlloc>
void report(const char* label, FnEngine engine, FnAlloc alloc,
            double flops = 0.0) {
    const double t_engine = median_ms(engine, 2, 5);
    const double t_alloc = median_ms(alloc, 1, 3);
    const double total = t_engine + t_alloc;
    std::printf("%-20s engine=%8.2f ms  alloc=%8.2f ms  compute=%5.1f%%"
                "  total=%8.2f ms",
                label, t_engine, t_alloc, 100.0 * t_engine / total, total);
    if (flops > 0) {
        std::printf("  (%7.1f GFLOP/s)", flops / t_engine / 1e6);
    }
    std::printf("\n");
}

void gemm_4096() {
    const std::size_t n = 4096;
    const std::size_t shape[2] = {n, n};
    auto a = make<double, 2>("A", shape);
    auto b = make<double, 2>("B", shape);
    auto c = make<double, 2>("C", shape);
    const std::string_view spec = "ij <- ik ; kj";
    report(
        "gemm 4096",
        [&] { einsums::compute_graph::einsum(
                  einsums::compute_graph::EinsumFormatString(spec), &c, a, b); },
        [&] {
            auto x = make<double, 2>("X", shape);
            auto y = make<double, 2>("Y", shape);
            auto z = make<double, 2>("Z", shape);
            (void)x; (void)y; (void)z;
        },
        2.0 * n * n * n);
}

void elementwise_12000() {
    const std::size_t n = 12000;
    const std::size_t shape[2] = {n, n};
    auto a = make<double, 2>("A", shape);
    auto b = make<double, 2>("B", shape);
    auto c = make<double, 2>("C", shape);
    const std::string_view spec = "ij <- ij ; ij";
    report(
        "elementwise 12000",
        [&] { einsums::compute_graph::einsum(
                  einsums::compute_graph::EinsumFormatString(spec), &c, a, b); },
        [&] {
            auto x = make<double, 2>("X", shape);
            auto y = make<double, 2>("Y", shape);
            auto z = make<double, 2>("Z", shape);
            (void)x; (void)y; (void)z;
        },
        2.0 * n * n);
}

void outer_8000() {
    const std::size_t n = 8000;
    const std::size_t shape[1] = {n};
    const std::size_t out_shape[2] = {n, n};
    auto x = make<double, 1>("x", shape);
    auto y = make<double, 1>("y", shape);
    auto c = make<double, 2>("C", out_shape);
    const std::string_view spec = "ij <- i ; j";
    report(
        "outer 8000",
        [&] { einsums::compute_graph::einsum(
                  einsums::compute_graph::EinsumFormatString(spec), &c, x, y); },
        [&] {
            auto p = make<double, 1>("p", shape);
            auto q = make<double, 1>("q", shape);
            auto r = make<double, 2>("r", out_shape);
            (void)p; (void)q; (void)r;
        },
        2.0 * n * n);
}

void batched_16x1024() {
    const std::size_t shape[3] = {16, 1024, 1024};
    auto a = make<double, 3>("A", shape);
    auto b = make<double, 3>("B", shape);
    auto c = make<double, 3>("C", shape);
    const std::string_view spec = "bik <- bij ; bjk";
    report(
        "batched 16x1024",
        [&] { einsums::compute_graph::einsum(
                  einsums::compute_graph::EinsumFormatString(spec), &c, a, b); },
        [&] {
            auto x = make<double, 3>("X", shape);
            auto y = make<double, 3>("Y", shape);
            auto z = make<double, 3>("Z", shape);
            (void)x; (void)y; (void)z;
        },
        2.0 * 16 * 1024 * 1024 * 1024);
}

}  // namespace

int main() {
    std::printf("Einsums engine-only vs allocation (row-major, float64)\n");
    gemm_4096();
    elementwise_12000();
    outer_8000();
    batched_16x1024();
    return 0;
}
