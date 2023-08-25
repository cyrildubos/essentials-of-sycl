#include <sycl/sycl.hpp>

constexpr std::size_t size = 256;

int main() {
  std::vector<int> vector_a(size, 10);
  std::vector<int> vector_b(size, 20);

  sycl::buffer buffer_a(vector_a);
  sycl::buffer buffer_b(vector_b);

  sycl::queue queue;

  queue.submit([&](auto& handler) {
    sycl::accessor accessor_a(buffer_a, handler, sycl::read_write);
    sycl::accessor accessor_b(buffer_b, handler, sycl::read_only);

    handler.parallel_for(sycl::range<1>(size), [=](auto identifier) {
      accessor_a[identifier] += 1 + accessor_b[identifier];
    });
  });

  sycl::host_accessor accessor_a(buffer_a, sycl::read_only);

  for (auto index = 0; index < size; ++index)
    std::cout << accessor_a[index] << '\n';
}