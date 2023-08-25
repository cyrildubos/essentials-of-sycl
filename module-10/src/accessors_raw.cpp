#include <sycl/sycl.hpp>

constexpr std::size_t size = 42;

int main() {
  std::array<int, size> array_a;
  std::array<int, size> array_b;
  std::array<int, size> array_c;

  for (auto index = 0; index < size; ++index) {
    array_a[index] = 0;
    array_b[index] = 0;
    array_c[index] = 0;
  }

  sycl::queue queue;

  sycl::buffer buffer_a(array_a);
  sycl::buffer buffer_b(array_b);
  sycl::buffer buffer_c(array_c);

  queue.submit([&](auto& handler) {
    sycl::accessor accessor_a(buffer_a, handler, sycl::read_only);
    sycl::accessor accessor_b(buffer_b, handler, sycl::write_only);

    handler.parallel_for(size, [=](auto identifier) {
      accessor_b[identifier] = accessor_a[identifier] + 1;
    });
  });

  queue.submit([&](auto& handler) {
    sycl::accessor accessor_a(buffer_a, handler, sycl::read_only);

    handler.parallel_for(
        size, [=](auto identifier) { int data = accessor_a[identifier]; });
  });

  queue.submit([&](auto& handler) {
    sycl::accessor accessor_b(buffer_b, handler, sycl::read_only);
    sycl::accessor accessor_c(buffer_c, handler, sycl::write_only);

    handler.parallel_for(size, [=](auto identifier) {
      accessor_c[identifier] = accessor_b[identifier] + 2;
    });
  });

  sycl::host_accessor accessor_c(buffer_c, sycl::read_only);

  for (auto index = 0; index < size; ++index)
    std::cout << accessor_c[index] << '\n';

  return 0;
}