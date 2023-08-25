#include <sycl/sycl.hpp>

constexpr std::size_t size = 16;

int main() {
  sycl::range<1> range(size);

  sycl::buffer<int> buffer_a(range);
  sycl::buffer<int> buffer_b(range);

  sycl::queue queue;

  queue.submit([&](auto& handler) {
    sycl::accessor accessor_a(buffer_a, handler, sycl::write_only);

    handler.parallel_for(range,
                         [=](auto identifier) { accessor_a[identifier] = 5; });
  });

  queue.submit([&](auto& handler) {
    sycl::accessor accessor_a(buffer_a, handler, sycl::write_only);

    handler.parallel_for(range, [=](auto identifier) {
      accessor_a[identifier] += accessor_a[0];
    });
  });

  queue.submit([&](auto& handler) {
    sycl::accessor accessor_b(buffer_b, handler, sycl::write_only);

    handler.parallel_for(range,
                         [=](auto identifier) { accessor_b[identifier] = 2; });
  });

  queue.submit([&](auto& handler) {
    sycl::accessor accessor_a(buffer_a, handler, sycl::read_only);
    sycl::accessor accessor_b(buffer_b, handler, sycl::read_write);

    handler.parallel_for(range, [=](auto identifier) {
      accessor_b[identifier] *= accessor_a[identifier];
    });
  });

  sycl::host_accessor accessor_b(buffer_b, sycl::read_only);

  for (auto index = 0; index < size; ++index)
    std::cout << accessor_b[index] << '\n';

  return 0;
}