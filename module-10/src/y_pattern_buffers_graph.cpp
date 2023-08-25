#include <sycl/sycl.hpp>

constexpr std::size_t size = 42;

int main() {
  sycl::queue queue;

  sycl::buffer<int> buffer_1(size);
  sycl::buffer<int> buffer_2(size);

  queue.submit([&](auto& handler) {
    sycl::accessor accessor_1(buffer_1, handler, sycl::write_only);

    handler.parallel_for(size,
                         [=](auto identifier) { accessor_1[identifier] = 1; });
  });

  queue.submit([&](auto& handler) {
    sycl::accessor accessor_2(buffer_2, handler, sycl::write_only);

    handler.parallel_for(size,
                         [=](auto identifier) { accessor_2[identifier] = 2; });
  });

  queue.submit([&](auto& handler) {
    sycl::accessor accessor_1(buffer_1, handler, sycl::write_only);
    sycl::accessor accessor_2(buffer_2, handler, sycl::write_only);

    handler.parallel_for(size, [=](auto identifier) {
      accessor_1[identifier] += accessor_2[identifier];
    });
  });

  queue.submit([&](auto& handler) {
    sycl::accessor accessor_1(buffer_1, handler, sycl::read_write);

    handler.single_task([=]() {
      for (auto index = 1; index < size; ++index)
        accessor_1[0] += accessor_1[index];

      accessor_1[0] /= 3;
    });
  });

  sycl::host_accessor accessor_1(buffer_1);

  std::cout << accessor_1[0] << '\n';

  return 0;
}