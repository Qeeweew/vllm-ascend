// SPDX-License-Identifier: Apache-2.0
// Fake runtime detects released storage; no Torch, CANN, or NPU is used.
#include <cassert>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

struct aclOpExecutor {};
using aclrtStream = void*;
using InitHugeMemThreadLocal = void (*)(void*, bool);
using UnInitHugeMemThreadLocal = void (*)(void*, bool);
using ReleaseHugeMem = void (*)(void*, bool);
constexpr int kByte = 0;
std::weak_ptr<std::vector<unsigned char>> allocated;
std::function<int()> queued;
bool defer_command = false;
int api_calls = 0;

namespace at {
struct TensorOptions {
  explicit TensorOptions(int) {}
  TensorOptions& dtype(int) { return *this; }
};
struct Storage {
  std::shared_ptr<std::vector<unsigned char>> owner;
  void* data() const { return owner->data(); }
};
struct Tensor {
  std::shared_ptr<std::vector<unsigned char>> owner;
  Storage storage() const { return {owner}; }
  void reset() { owner.reset(); }
};
Tensor empty(std::initializer_list<uint64_t> shape, const TensorOptions&) {
  Tensor result{std::make_shared<std::vector<unsigned char>>(*shape.begin(), 0x5a)};
  allocated = result.owner;
  return result;
}
}  // namespace at
namespace torch_npu::utils {
int get_npu_device_type() { return 0; }
}  // namespace torch_npu::utils
namespace c10_npu {
struct Stream {
  aclrtStream stream(bool) { return nullptr; }
};
Stream getCurrentNPUStream() { return {}; }
}  // namespace c10_npu
namespace at_npu::native {
struct OpCommand {
  std::function<int()> handler;
  void Name(const char*) {}
  void SetCustomHandler(std::function<int()> value) { handler = std::move(value); }
  void Run() {
    if (defer_command) {
      queued = handler;
    } else {
      handler();
    }
  }
};
}  // namespace at_npu::native
int get_workspace(uint64_t requested, uint64_t* bytes, aclOpExecutor** executor) {
  *bytes = requested;
  *executor = nullptr;
  return 0;
}
int launch(void* pointer, uint64_t bytes, aclOpExecutor*, aclrtStream) {
  ++api_calls;
  if (bytes == 0) {
    assert(pointer == nullptr);
    return 0;
  }
  auto live = allocated.lock();
  if (!live || pointer != live->data() || live->size() != bytes) {
    throw std::runtime_error("workspace released before handler submission");
  }
  assert((*live)[bytes - 1] == 0x5a);
  return 0;
}
void* GetOpApiFuncAddr(const char* name) {
  if (std::string(name) == "FakeGetWorkspaceSize") {
    return reinterpret_cast<void*>(&get_workspace);
  }
  if (std::string(name) == "Fake") {
    return reinterpret_cast<void*>(&launch);
  }
  return nullptr;
}
const char* GetOpApiLibName() { return "cpu_fixture"; }
const char* aclGetRecentErrMsg() { return "cpu_fixture"; }
#define TORCH_CHECK(condition, ...)                                     \
  do {                                                                  \
    if (!(condition)) throw std::runtime_error("runtime check failed"); \
  } while (false)
template <typename... Args>
auto ConvertTypes(Args&... args) {
  return std::make_tuple(args...);
}
template <typename Tuple>
auto ConvertToOpApiFunc(const Tuple&, void* function) {
  return reinterpret_cast<int (*)(uint64_t, uint64_t*, aclOpExecutor**)>(function);
}
template <typename Function, typename Tuple>
auto call(Function function, Tuple args) {
  return std::apply(function, args);
}
template <typename Tuple>
void ReleaseConvertTypes(Tuple&) {}

// PRODUCTION_MACRO

int main(int argc, char** argv) {
  assert(argc == 3);
  defer_command = std::stoi(argv[1]) != 0;
  uint64_t requested = std::stoull(argv[2]);
  EXEC_NPU_CMD(Fake, requested);
  if (defer_command) {
    assert(api_calls == 0);
    if (requested) assert(!allocated.expired());
    auto retained_queue_slot = queued;
    queued();
    // A task queue can retain handler copies after their one submission.
    // Completed scratch must not stay allocated until that slot is reused.
    assert(allocated.expired());
    queued = {};
  }
  assert(api_calls == 1);
  assert(allocated.expired());
  return 0;
}
