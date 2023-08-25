#include <sycl/sycl.hpp>

constexpr std::size_t size = 256;

int main() {
  sycl::queue queue;

  std::vector<int> vector_1(size, 1);
  std::vector<int> vector_2(size, 2);
  std::vector<int> vector_3(size, 3);

  sycl::buffer<int> buffer_1(vector_1);
  sycl::buffer<int> buffer_2(vector_2);
  sycl::buffer<int> buffer_3(vector_3);

  queue.submit([&](auto& handler) {
    sycl::accessor accessor_1(buffer_1, handler, sycl::read_write);
    sycl::accessor accessor_3(buffer_3, handler, sycl::read_only);

    handler.parallel_for(size, [=](auto identifier) {
      accessor_1[identifier] += accessor_3[identifier];
    });
  });

  queue.submit([&](auto& handler) {
    sycl::accessor accessor_2(buffer_2, handler, sycl::read_write);

    handler.parallel_for(size,
                         [=](auto identifier) { accessor_2[identifier] *= 2; });
  });

  queue.submit([&](auto& handler) {
    sycl::accessor accessor_1(buffer_1, handler, sycl::read_only);
    sycl::accessor accessor_2(buffer_2, handler, sycl::read_only);
    sycl::accessor accessor_3(buffer_3, handler, sycl::write_only);

    handler.parallel_for(size, [=](auto identifier) {
      accessor_3[identifier] = accessor_1[identifier] + accessor_2[identifier];
    });
  });

  sycl::host_accessor accessor_3(buffer_3);

  for (auto index = 0; index < size; ++index)
    std::cout << accessor_3[index] << '\n';

  return 0;
}