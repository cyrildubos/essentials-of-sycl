#include <sycl/sycl.hpp>

constexpr std::size_t size = 42;

int main() {
  sycl::queue queue;

  sycl::range<1> range(size);

  sycl::buffer<int> buffer_a(range);
  sycl::buffer<int> buffer_b(range);
  sycl::buffer<int> buffer_c(range);

  queue.submit([&](auto& handler) {
    sycl::accessor accessor_a(buffer_a, handler, sycl::write_only,
                              sycl::no_init);
    sycl::accessor accessor_b(buffer_b, handler, sycl::write_only,
                              sycl::no_init);
    sycl::accessor accessor_c(buffer_c, handler, sycl::write_only,
                              sycl::no_init);

    handler.parallel_for(sycl::range<1>(size), [=](auto identifier) {
      accessor_a[identifier] = 1;
      accessor_b[identifier] = 40;
      accessor_c[identifier] = 0;
    });
  });

  queue.submit([&](auto& handler) {
    sycl::accessor accessor_a(buffer_a, handler, sycl::read_only);
    sycl::accessor accessor_b(buffer_b, handler, sycl::read_only);
    sycl::accessor accessor_c(buffer_c, handler, sycl::read_write);

    handler.parallel_for(sycl::range<1>(size), [=](auto identifier) {
      accessor_c[identifier] += accessor_a[identifier] + accessor_b[identifier];
    });
  });

  sycl::host_accessor accessor_c(buffer_c, sycl::read_only);

  for (auto index = 0; index < size; ++index)
    std::cout << accessor_c[index] << '\n';

  return 0;
}