#include <sycl/sycl.hpp>

constexpr std::size_t size = 42;

int main() {
  sycl::queue queue;

  sycl::buffer<int> buffer(size);

  queue.submit([&](auto& handler) {
    sycl::accessor accessor(buffer, handler);

    handler.parallel_for(size,
                         [=](auto identifier) { accessor[identifier] = 1; });
  });

  queue.submit([&](auto& handler) {
    sycl::accessor accessor(buffer, handler);

    handler.single_task([=]() {
      for (auto index = 1; index < size; ++index)
        accessor[0] += accessor[index];
    });
  });

  sycl::host_accessor accessor(buffer);

  std::cout << accessor[0] << '\n';

  return 0;
}