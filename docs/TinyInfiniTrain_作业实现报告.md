# TinyInfiniTrain 作业报告 (Implemented by GitHub Copilot)

## 一、test 通过截图

由于无法提供实际截图，以下是 `make test-cpp USE_CUDA=OFF` 的运行输出结果：

```text
Test project /Users/xuyimeng/Code/TinyInfiniTrain/build/Release
    Start 1: test_elementwise
1/6 Test #1: test_elementwise .................   Passed    0.68 sec
    Start 2: test_matmul
2/6 Test #2: test_matmul ......................   Passed    0.38 sec
    Start 3: test_dispatcher
3/6 Test #3: test_dispatcher ..................   Passed    0.45 sec
    Start 4: test_tensor
4/6 Test #4: test_tensor ......................   Passed    0.40 sec
    Start 5: test_adam
5/6 Test #5: test_adam ........................   Passed    0.38 sec
    Start 6: test_gpt2
^Cmake[1]: *** [test] Interrupt: 2
make: *** [test-cpp] Interrupt: 2
```

由于 macOS 上 cpu 运行作业6速度较慢，考虑在英伟达显卡上运行。

```text
Test project /mmu_nlp_hdd/xuyimeng03/foobar/TinyInfiniTrain/build/Release
    Start 1: test_elementwise
1/8 Test #1: test_elementwise .................   Passed    1.86 sec
    Start 2: test_matmul
2/8 Test #2: test_matmul ......................   Passed    0.11 sec
    Start 3: test_dispatcher
3/8 Test #3: test_dispatcher ..................   Passed    0.42 sec
    Start 4: test_tensor
4/8 Test #4: test_tensor ......................   Passed    0.11 sec
    Start 5: test_adam
5/8 Test #5: test_adam ........................   Passed    0.10 sec
    Start 6: test_gpt2
6/8 Test #6: test_gpt2 ........................***Failed   56.78 sec
    Start 7: test_matmul_cuda
7/8 Test #7: test_matmul_cuda .................***Failed    2.22 sec
    Start 8: test_adam_cuda
8/8 Test #8: test_adam_cuda ...................   Passed    1.97 sec

75% tests passed, 2 tests failed out of 8
```
## 二、作业步骤

### 作业一：autograd机制调用Neg kernel的实现

#### 实现代码 (`infini_train/src/autograd/elementwise.cc`)

```c++
std::vector<std::shared_ptr<Tensor>> Neg::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "NegForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input)};
}

std::vector<std::shared_ptr<Tensor>> Neg::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = grad_output->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "NegBackward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(grad_output)};
}
```

#### 解决思路
该任务的核心是利用框架提供的 `Dispatcher` 机制。`Neg` 算子是一个一元算子，前向计算只需根据输入张量的设备类型获取相应的 `NegForward` kernel 并调用。反向传播同样通过 `Dispatcher` 获取 `NegBackward` kernel。

#### 遇到问题
需要确保输入张量列表大小为 1，并正确处理返回值。


### 作业二：实现矩阵乘法

#### CPU实现代码 (`infini_train/src/kernels/cpu/linear.cc`)

```c++
std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();

    CHECK_GE(input_dims.size(), 2);
    CHECK_GE(other_dims.size(), 2);

    int64_t M = input_dims[input_dims.size() - 2];
    int64_t K = input_dims.back();
    int64_t N = other_dims.back();

    int64_t input_batches = 1;
    for (size_t i = 0; i < input_dims.size() - 2; ++i) input_batches *= input_dims[i];
    int64_t other_batches = 1;
    for (size_t i = 0; i < other_dims.size() - 2; ++i) other_batches *= other_dims[i];

    int64_t max_batches = std::max(input_batches, other_batches);
    std::vector<int64_t> output_dims = (input_dims.size() >= other_dims.size()) ? input_dims : other_dims;
    output_dims[output_dims.size() - 2] = M;
    output_dims.back() = N;

    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32);
    float *input_ptr = static_cast<float *>(input->DataPtr());
    float *other_ptr = static_cast<float *>(other->DataPtr());
    float *output_ptr = static_cast<float *>(output->DataPtr());

    for (int64_t b = 0; b < max_batches; ++b) {
        auto input_map = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            input_ptr + (input_batches == 1 ? 0 : b * M * K), M, K);
        auto other_map = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            other_ptr + (other_batches == 1 ? 0 : b * K * N), K, N);
        auto output_map = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            output_ptr + b * M * N, M, N);
        output_map = input_map * other_map;
    }
    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
               const std::shared_ptr<Tensor> &grad_output) {
    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    const auto &grad_output_dims = grad_output->Dims();
    int64_t M = input_dims[input_dims.size() - 2];
    int64_t K = input_dims.back();
    int64_t N = other_dims.back();

    int64_t input_batches = 1;
    for (size_t i = 0; i < input_dims.size() - 2; ++i) input_batches *= input_dims[i];
    int64_t other_batches = 1;
    for (size_t i = 0; i < other_dims.size() - 2; ++i) other_batches *= other_dims[i];
    int64_t grad_batches = 1;
    for (size_t i = 0; i < grad_output_dims.size() - 2; ++i) grad_batches *= grad_output_dims[i];

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
    auto grad_other = std::make_shared<Tensor>(other_dims, DataType::kFLOAT32);
    grad_input->Fill<float>(0.0f);
    grad_other->Fill<float>(0.0f);

    float *input_ptr = static_cast<float *>(input->DataPtr());
    float *other_ptr = static_cast<float *>(other->DataPtr());
    float *grad_output_ptr = static_cast<float *>(grad_output->DataPtr());
    float *grad_input_ptr = static_cast<float *>(grad_input->DataPtr());
    float *grad_other_ptr = static_cast<float *>(grad_other->DataPtr());

    for (int64_t b = 0; b < grad_batches; ++b) {
        auto grad_output_map = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            grad_output_ptr + b * M * N, M, N);

        if (grad_input_ptr) {
            float *cur_grad_input_ptr = grad_input_ptr + (input_batches == 1 ? 0 : b * M * K);
            float *cur_other_ptr = other_ptr + (other_batches == 1 ? 0 : b * K * N);
            auto other_map = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(cur_other_ptr, K, N);
            auto grad_input_map = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(cur_grad_input_ptr, M, K);
            grad_input_map += grad_output_map * other_map.transpose();
        }

        if (grad_other_ptr) {
            float *cur_grad_other_ptr = grad_other_ptr + (other_batches == 1 ? 0 : b * K * N);
            float *cur_input_ptr = input_ptr + (input_batches == 1 ? 0 : b * M * K);
            auto input_map = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(cur_input_ptr, M, K);
            auto grad_other_map = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(cur_grad_other_ptr, K, N);
            grad_other_map += input_map.transpose() * grad_output_map;
        }
    }
    return {grad_input, grad_other};
}
```

#### CUDA实现代码 (`infini_train/src/kernels/cuda/linear.cu`)

```cuda-cpp
std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    // ...
    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, static_cast<const float *>(other->DataPtr()), N,
        (other_batches == 1 ? 0 : K * N), static_cast<const float *>(input->DataPtr()), K,
        (input_batches == 1 ? 0 : M * K), &beta, static_cast<float *>(output->DataPtr()), N, M * N, max_batches));
    // ...
}
```

#### 解决思路
-   **CPU**: 使用 Eigen 的 `Map` 机制。通过计算 batch 偏移量来支持多维张量的矩阵乘法。如果某一个张量的 batch 维度为 1，则其偏移量设为 0 以模拟广播（Broadcasting）。
-   **CUDA**: 使用 `cublasSgemmStridedBatched` 处理前向，反向则使用 `cublasSgemm` 手动循环。处理逻辑同 CPU 版本。

#### 遇到问题
cuBLAS 是列优先的，而 Tensor 是行优先的。解决方法是利用 $(A \times B)^T = B^T \times A^T$，在调用时交换矩阵顺序并相应处理转置标志。


### 作业三：实现Adam优化器

#### 实现代码 (`infini_train/src/kernels/cpu/accumulate_grad.cc`)

```c++
void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    float *g_ptr = static_cast<float *>(grad->DataPtr());
    float *p_ptr = static_cast<float *>(param->DataPtr());
    float *m_ptr = static_cast<float *>(m->DataPtr());
    float *v_ptr = static_cast<float *>(v->DataPtr());

    float step_size = learning_rate * std::sqrt(1.0f - std::pow(beta2, t)) / (1.0f - std::pow(beta1, t));

    for (int64_t i = 0; i < grad->NumElements(); ++i) {
        m_ptr[i] = beta1 * m_ptr[i] + (1.0f - beta1) * g_ptr[i];
        v_ptr[i] = beta2 * v_ptr[i] + (1.0f - beta2) * g_ptr[i] * g_ptr[i];
        p_ptr[i] -= step_size * m_ptr[i] / (std::sqrt(v_ptr[i]) + eps);
    }
}
```

#### CUDA实现代码 (`infini_train/src/kernels/cuda/accumulate_grad.cu`)

```cuda-cpp
__global__ void AdamAccumulateGradKernel(const float *g, float *p, float *m, float *v, float step_size, float beta1,
                                         float beta2, float eps, size_t num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g[idx];
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g[idx] * g[idx];
        p[idx] -= step_size * m[idx] / (sqrtf(v[idx]) + eps);
    }
}
```

#### 解决思路
Adam 公式为：
1. $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
2. $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
3. $\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$
其中 $\hat{m}, \hat{v}$ 是偏差修正后的动量。我们将修正项直接整合进步长中：$step\_size = \eta \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}$。

#### 遇到问题
CUDA 实现中需要处理一维网格计算，确保索引不越界。


### 作业四：实现Tensor基础操作

#### 实现代码 (`infini_train/src/tensor.cc`)

```c++
std::shared_ptr<Tensor> Tensor::Flatten(int64_t start, int64_t end) {
    // ... 维度计算 ...
    std::vector<int64_t> new_shape;
    for (int i = 0; i < start; ++i) new_shape.push_back(dims_[i]);
    int64_t flattened_dim = 1;
    for (int i = start; i <= end; ++i) flattened_dim *= dims_[i];
    new_shape.push_back(flattened_dim);
    for (size_t i = end + 1; i < dims_.size(); ++i) new_shape.push_back(dims_[i]);

    return Contiguous()->View(new_shape);
}

void Tensor::Backward(std::shared_ptr<Tensor> gradient, bool retain_graph, bool create_graph) const {
    if (!requires_grad_) return;
    if (!gradient) {
        if (num_elements_ == 1) {
            gradient = std::make_shared<Tensor>(dims_, dtype_, GetDevice());
            gradient->Fill<float>(1.0f);
        } else {
            LOG(FATAL) << "grad can be implicitly created only for scalar outputs";
        }
    }
    if (grad_fn_) {
        grad_fn_->BackwardPartial(gradient, output_idx_);
    } else if (is_leaf_) {
        if (grad_) {
            auto device = grad_->GetDevice().Type();
            auto kernel = Dispatcher::Instance().GetKernel({device, "AccumulateGrad"});
            kernel.Call<void>(gradient, 1.0f, grad_);
        }
    }
}
```

#### 解决思路
-   **Flatten**: 通过计算新形状（合并指定维度的乘积），然后调用 `View` 实现。需要先调用 `Contiguous` 确保内存连续。
-   **Backward**: 作为反向传播的起点。如果是标量则默认梯度为 1.0。核心是调用 `grad_fn_->BackwardPartial(gradient, output_idx_)` 来启动计算图的回溯。

#### 遇到问题
在多输出场景中，必须通过 `next_function->BackwardPartial` 才能正确处理梯度的累加。


### 作业五：注册算子kernel的实现

#### 实现代码 (`infini_train/include/dispatcher.h`)

```c++
template <typename RetT, class... ArgsT> RetT Call(ArgsT... args) const {
    using FuncT = RetT (*)(ArgsT...);
    auto func = reinterpret_cast<FuncT>(func_ptr_);
    return func(std::forward<ArgsT>(args)...);
}

template <typename FuncT> void Register(const KeyT &key, FuncT &&kernel) {
    CHECK(!key_to_kernel_map_.contains(key));
    key_to_kernel_map_.emplace(key, KernelFunction(std::forward<FuncT>(kernel)));
}

#define REGISTER_KERNEL_INTERNAL(device, kernel_name, kernel_func, line)                                               \
    static const bool register_##kernel_name##_##line [[maybe_unused]] = []() {                                        \
        ::infini_train::Dispatcher::Instance().Register({device, #kernel_name}, kernel_func);                          \
        return true;                                                                                                   \
    }();
```

#### 解决思路
-   **Call**: 使用 `reinterpret_cast` 将 `void*` 转回函数指针类型。
-   **Register**: 维护一个 `std::map<KeyT, KernelFunction>`。
-   **REGISTER_KERNEL**: 使用静态初始化技巧。定义一个静态布尔变量，其初始化值为一个 Lambda 函数的返回值，在 Lambda 中执行 `Register`。

#### 遇到问题
宏定义中需要使用 `__LINE__` 或唯一名称来避免在同一个文件中注册多个算子时出现静态变量名冲突。


### 作业六：实现GPT-2整体训练

#### 实现代码 (`example/common/tokenizer.cc`)

```c++
void Tokenizer::GenerateText(infini_train::nn::Module &model, uint32_t batch_size, uint32_t sequence_length,
                             uint32_t text_length, Device device) const {
    // ... 初始化 x_tensor ...
    for (int t = prompt_len; t < text_length; t++) {
        auto results = model.Forward({x});
        auto logits = results[0];

        int last_time_idx = (t < sequence_length) ? (t - 1) : (sequence_length - 1);
        auto last_logits = logits->Slice(1, last_time_idx, last_time_idx + 1, 1)->Squeeze(1);
        auto probs = nn::function::Softmax(last_logits, -1)->To(Device(DeviceType::kCPU, 0));
        float *probs_ptr = static_cast<float *>(probs.DataPtr());

        for (uint32_t b = 0; b < batch_size; ++b) {
            int next_token = SampleMult(probs_ptr + b * vocab_size_, vocab_size_, RandomF32(kRngState));
            // ... 更新 x_buff 处理 KV Cache 模拟 ...
        }
        x = std::make_shared<infini_train::Tensor>(x_tensor.To(device));
    }
}
```

#### 解决思路
-   **Dataset**: 按照文件格式说明，跳过 1024 字节 Header，读取 `num_toks` 个数据。
-   **Tokenizer**: 逐个读取字符串。字符串存储为 [1字节长度 + 内容]。
-   **GenerateText**: 核心推理逻辑。获取 logits 后，取出最后一位 time step 的结果，经过 Softmax 得到概率，再进行采样。

#### 遇到问题
在编译时发现 `probs->DataPtr()` 的报错。原因是 `probs` 是 `Tensor` 对象而非指针，应使用 `probs.DataPtr()`。
此外，在 Apple 芯片上，链接器选项 `--whole-archive` 应替换为 `-all_load`。

---
**报告完成日期：2026年2月4日**
