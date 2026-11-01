// Standalone test for string_length. Builds a string column with a
// mix of 1-, 2-, and 3-byte UTF-8 code points, launches the kernel,
// and checks the result against a CPU reference, for
// n in {0, 1, 1000, 1000003}.

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

extern "C" __global__ void kernel(const int *offsets, const unsigned char *chars,
                                   int *lengths, long long n);

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                     \
                    cudaGetErrorString(err), __FILE__, __LINE__);            \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

const int BLOCK = 256;

static void append_utf8(std::vector<unsigned char> &out, unsigned int cp) {
    if (cp < 0x80) {
        out.push_back((unsigned char)cp);
    } else if (cp < 0x800) {
        out.push_back((unsigned char)(0xC0 | (cp >> 6)));
        out.push_back((unsigned char)(0x80 | (cp & 0x3F)));
    } else {
        out.push_back((unsigned char)(0xE0 | (cp >> 12)));
        out.push_back((unsigned char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((unsigned char)(0x80 | (cp & 0x3F)));
    }
}

static bool run_case(long long n, std::mt19937_64 &rng) {
    std::uniform_int_distribution<int> len_dist(0, 5);
    std::uniform_int_distribution<int> kind_dist(0, 2);
    std::uniform_int_distribution<unsigned int> ascii_dist(0x20, 0x7E);
    std::uniform_int_distribution<unsigned int> two_byte_dist(0x80, 0x7FF);
    std::uniform_int_distribution<unsigned int> three_byte_dist(0x800, 0xFFFF);

    std::vector<int> ref_lengths(n > 0 ? n : 1, 0);
    std::vector<int> offsets(n + 1, 0);
    std::vector<unsigned char> chars;

    for (long long i = 0; i < n; ++i) {
        int num_chars = len_dist(rng);
        ref_lengths[i] = num_chars;
        for (int c = 0; c < num_chars; ++c) {
            int kind = kind_dist(rng);
            unsigned int cp;
            if (kind == 0) {
                cp = ascii_dist(rng);
            } else if (kind == 1) {
                cp = two_byte_dist(rng);
            } else {
                do {
                    cp = three_byte_dist(rng);
                } while (cp >= 0xD800 && cp <= 0xDFFF);
            }
            append_utf8(chars, cp);
        }
        offsets[i + 1] = (int)chars.size();
    }

    int *d_offsets, *d_lengths;
    unsigned char *d_chars;
    CUDA_CHECK(cudaMalloc(&d_offsets, sizeof(int) * (n + 1)));
    CUDA_CHECK(cudaMalloc(&d_chars, sizeof(unsigned char) * (chars.size() > 0 ? chars.size() : 1)));
    CUDA_CHECK(cudaMalloc(&d_lengths, sizeof(int) * (n > 0 ? n : 1)));
    CUDA_CHECK(cudaMemcpy(d_offsets, offsets.data(), sizeof(int) * (n + 1),
                          cudaMemcpyHostToDevice));
    if (!chars.empty()) {
        CUDA_CHECK(cudaMemcpy(d_chars, chars.data(), sizeof(unsigned char) * chars.size(),
                              cudaMemcpyHostToDevice));
    }

    long long blocks = (n + BLOCK - 1) / BLOCK;
    if (blocks > 0) {
        kernel<<<blocks, BLOCK>>>(d_offsets, d_chars, d_lengths, n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> got(n > 0 ? n : 1);
    CUDA_CHECK(cudaMemcpy(got.data(), d_lengths, sizeof(int) * n,
                          cudaMemcpyDeviceToHost));

    bool ok = true;
    for (long long i = 0; i < n; ++i) {
        if (got[i] != ref_lengths[i]) {
            ok = false;
            break;
        }
    }

    printf("n=%lld: %s (bytes=%zu)\n", n, ok ? "PASS" : "FAIL", chars.size());

    cudaFree(d_offsets);
    cudaFree(d_chars);
    cudaFree(d_lengths);
    return ok;
}

int main() {
    std::mt19937_64 rng(0);
    bool all_ok = true;
    for (long long n : {0LL, 1LL, 1000LL, 1000003LL}) {
        all_ok &= run_case(n, rng);
    }
    return all_ok ? 0 : 1;
}
