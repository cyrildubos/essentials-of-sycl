#include <sycl/sycl.hpp>

constexpr std::size_t size = 42;

int main() {
  sycl::queue queue;

  std::array<int, size> array;

  int* data = sycl::malloc_device<int>(size, queue);

  for (auto index = 0; index < size; ++index)
    array[index] = index;

  queue
      .submit([&](auto& handler) {
        handler.memcpy(data, &array[0], size * sizeof(int));
      })
      .wait();

  queue
      .submit([&](auto& handler) {
        handler.parallel_for(size,
                             [=](auto identifier) { data[identifier] *= 2; });
      })
      .wait();

  queue
      .submit([&](auto& handler) {
        handler.memcpy(&array[0], data, size * sizeof(int));
      })
      .wait();

  sycl::free(data, queue);

  return 0;
}