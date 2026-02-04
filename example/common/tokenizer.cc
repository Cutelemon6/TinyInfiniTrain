#include "example/common/tokenizer.h"

#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "glog/logging.h"

namespace infini_train {

constexpr uint32_t kGpt2Eot = 50256;
constexpr uint32_t kLLaMA3Eot = 128001;
constexpr uint64_t kRandomU32Multiplier = 0x2545F4914F6CDD1Dull;
constexpr float kF32Divisor = 16777216.0f; // 2^24
constexpr uint64_t kRngState = 1337;

using Version = Tokenizer::Version;

const std::unordered_map<uint32_t, uint32_t> kEotMap = {
    {20240328, kGpt2Eot},   // GPT-2
    {20240801, kLLaMA3Eot}, // LLaMA-3
};

const std::unordered_map<uint32_t, std::vector<uint32_t>> kPromptMap = {
    // e.g. "The meaning of life is"
    // ref: https://tiktokenizer.vercel.app/
    {20240328, std::vector<uint32_t>{464, 3616, 286, 1204, 318}}, // GPT-2
    {20240801, std::vector<uint32_t>{791, 7438, 315, 2324, 374}}, // LLaMA-3
};

std::vector<uint8_t> ReadSeveralBytesFromIfstream(size_t num_bytes, std::ifstream *ifs) {
    std::vector<uint8_t> result(num_bytes);
    ifs->read(reinterpret_cast<char *>(result.data()), num_bytes);
    return result;
}

template <typename T> T BytesToType(const std::vector<uint8_t> &bytes, size_t offset) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable.");
    T value;
    std::memcpy(&value, &bytes[offset], sizeof(T));
    return value;
}

unsigned int RandomU32(uint64_t &state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return (state * kRandomU32Multiplier) >> 32;
}

float RandomF32(uint64_t &state) { // random float32 in [0,1)
    return (RandomU32(state) >> 8) / kF32Divisor;
}

int SampleMult(float *probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from RandomF32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

Tokenizer::Tokenizer(const std::string &filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open file: " << filepath;

    auto header_bytes = ReadSeveralBytesFromIfstream(1024, &ifs);
    uint32_t magic = BytesToType<uint32_t>(header_bytes, 0);
    uint32_t version = BytesToType<uint32_t>(header_bytes, 4);
    uint32_t vocab_size = BytesToType<uint32_t>(header_bytes, 8);

    magic_number_ = magic;
    vocab_size_ = vocab_size;
    eot_token_ = kEotMap.at(magic);

    token_table_.resize(vocab_size);
    for (uint32_t i = 0; i < vocab_size; ++i) {
        uint8_t len;
        ifs.read(reinterpret_cast<char *>(&len), 1);
        std::vector<char> buf(len);
        ifs.read(buf.data(), len);
        token_table_[i] = std::string(buf.begin(), buf.end());
    }
    ifs.close();
}

std::string Tokenizer::Decode(uint32_t token_id) const {
    if (token_id < token_table_.size()) {
        return token_table_[token_id];
    }
    return "";
}

void Tokenizer::GenerateText(infini_train::nn::Module &model, uint32_t batch_size, uint32_t sequence_length,
                             uint32_t text_length, Device device) const {
    std::vector<int64_t> dims;
    dims.assign({batch_size, sequence_length});
    // x_tensor (FLAGS_batch_size, FLAGS_sequence_length) eq:(4, 64)
    infini_train::Tensor x_tensor = infini_train::Tensor(dims, DataType::kINT64);
    int64_t *x_buff = static_cast<int64_t *>(x_tensor.DataPtr());
    for (int i = 0; i < batch_size * sequence_length; ++i) {
        x_buff[i] = eot_token_;
    }

    // Give some contexts: "The meaning of life is "
    auto prompt = kPromptMap.at(magic_number_);
    auto prompt_len = prompt.size();
    for (int i = 0; i < prompt_len; ++i) {
        x_buff[i] = prompt[i];
    }
    std::cout << "The meaning of life is";

    auto x = std::make_shared<infini_train::Tensor>(x_tensor.To(device));
    uint64_t kRngState = infini_train::kRngState;
    LOG(INFO) << "start generate text:";
    for (int t = prompt_len; t < text_length; t++) {
        auto results = model.Forward({x});
        auto logits = results[0];

        int last_time_idx = (t < sequence_length) ? (t - 1) : (sequence_length - 1);
        auto last_logits = logits->Slice(1, last_time_idx, last_time_idx + 1, 1)->Squeeze(1);
        auto probs = nn::function::Softmax(last_logits, -1)->To(Device(DeviceType::kCPU, 0));
        float *probs_ptr = static_cast<float *>(probs.DataPtr());

        for (uint32_t b = 0; b < batch_size; ++b) {
            float coin_val = RandomF32(kRngState);
            int next_token = SampleMult(probs_ptr + b * vocab_size_, vocab_size_, coin_val);

            if (b == 0) {
                std::cout << Decode(next_token);
                std::cout.flush();
            }

            if (t < sequence_length) {
                x_buff[b * sequence_length + t] = next_token;
            } else {
                for (int i = 0; i < sequence_length - 1; ++i) {
                    x_buff[b * sequence_length + i] = x_buff[b * sequence_length + i + 1];
                }
                x_buff[b * sequence_length + sequence_length - 1] = next_token;
            }
        }
        x = std::make_shared<infini_train::Tensor>(x_tensor.To(device));
    }
    std::cout << std::endl;
}
} // namespace infini_train
