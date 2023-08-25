#include <sycl/sycl.hpp>

constexpr std::size_t size = 16;

int main() {
  sycl::range<1> range(size);

  std::vector<int> vector(size, 10);

  sycl::buffer buffer(vector);

  sycl::queue queue;

  queue.submit([&](auto& handler) {
    sycl::accessor accessor(buffer, handler, sycl::write_only);

    handler.parallel_for(range,
                         [=](auto identifier) { accessor[identifier] -= 2; });
  });

  sycl::host_accessor accessor(buffer, sycl::read_only);

  for (auto index = 0; index < size; ++index)
    std::cout << accessor[index] << '\n';

  return 0;
}