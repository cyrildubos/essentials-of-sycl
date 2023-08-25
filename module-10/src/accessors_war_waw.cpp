#include <sycl/sycl.hpp>

constexpr std::size_t size = 42;

int main() {
  std::array<int, size> array_a;
  std::array<int, size> array_b;

  for (auto index = 0; index < size; ++index) {
    array_a[index] = 0;
    array_b[index] = 0;
  }

  sycl::queue queue;

  sycl::buffer buffer_a(array_a);
  sycl::buffer buffer_b(array_b);

  queue.submit([&](auto& handler) {
    sycl::accessor accessor_a(buffer_a, handler, sycl::read_only);
    sycl::accessor accessor_b(buffer_b, handler, sycl::write_only);

    handler.parallel_for(size, [=](auto identifier) {
      accessor_b[identifier] = accessor_a[identifier] + 1;
    });
  });

  queue.submit([&](auto& handler) {
    sycl::accessor accessor_a(buffer_a, handler, sycl::write_only);

    handler.parallel_for(size,
                         [=](auto identifier) { accessor_a[identifier] = 42; });
  });

  queue.submit([&](auto& handler) {
    sycl::accessor accessor_b(buffer_b, handler, sycl::write_only);

    handler.parallel_for(size,
                         [=](auto identifier) { accessor_b[identifier] = 42; });
  });

  sycl::host_accessor accessor_a(buffer_a, sycl::read_only);
  sycl::host_accessor accessor_b(buffer_b, sycl::read_only);

  for (auto index = 0; index < size; ++index)
    std::cout << accessor_a[index] << ' ' << accessor_b[index] << '\n';

  return 0;
}