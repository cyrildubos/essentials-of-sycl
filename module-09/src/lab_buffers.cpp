#include <sycl/sycl.hpp>

constexpr std::size_t size = 256;

int main() {
  int data[size];

  for (auto index = 0; index < size; ++index)
    data[index] = index;

  for (auto index = 0; index < size; ++index)
    std::cout << data[index] << '\n';

  sycl::buffer<int> buffer(data, sycl::range<1>(size));

  sycl::buffer<int> buffer_1(buffer, 0, sycl::range<1>(64));
  sycl::buffer<int> buffer_2(buffer, 64, sycl::range<1>(128));
  sycl::buffer<int> buffer_3(buffer, 192, sycl::range<1>(64));

  sycl::queue queue;

  queue.submit([&](auto& handler) {
    sycl::accessor accessor_1(buffer_1, handler);

    handler.parallel_for(sycl::range<1>(64),
                         [=](auto identifier) { accessor_1[identifier] *= 2; });
  });

  queue.submit([&](auto& handler) {
    sycl::accessor accessor_2(buffer_2, handler);

    handler.parallel_for(sycl::range<1>(128),
                         [=](auto identifier) { accessor_2[identifier] *= 3; });
  });

  queue.submit([&](auto& handler) {
    sycl::accessor accessor_3(buffer_3, handler);

    handler.parallel_for(sycl::range<1>(64),
                         [=](auto identifier) { accessor_3[identifier] *= 2; });
  });

  sycl::host_accessor accessor(buffer);

  for (auto index = 0; index < size; ++index)
    std::cout << data[index] << '\n';

  return 0;
}