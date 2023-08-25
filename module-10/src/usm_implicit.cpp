#include <sycl/sycl.hpp>

constexpr std::size_t size = 42;

int main() {
  sycl::queue queue;

  int* host_data = sycl::malloc_host<int>(size, queue);
  int* shared_data = sycl::malloc_shared<int>(size, queue);

  for (auto index = 0; index < size; ++index)
    host_data[index] = index;

  queue
      .submit([&](auto& handler) {
        handler.parallel_for(size, [=](auto identifier) {
          shared_data[identifier] = host_data[identifier] + 1;
        });
      })
      .wait();

  for (auto index = 0; index < size; ++index)
    host_data[index] = shared_data[index];

  sycl::free(host_data, queue);
  sycl::free(shared_data, queue);

  return 0;
}