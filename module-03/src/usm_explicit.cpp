#include <sycl/sycl.hpp>

constexpr std::size_t size = 16;

int main() {
  sycl::queue queue;

  auto data = static_cast<int*>(malloc(size * sizeof(int)));

  for (auto index = 0; index < size; ++index)
    data[index] = index;

  auto data_device = sycl::malloc_device<int>(size, queue);

  queue.memcpy(data_device, data, size * sizeof(int)).wait();

  queue
      .parallel_for(sycl::range<1>(size),
                    [=](auto identifier) { data_device[identifier] *= 2; })
      .wait();

  queue.memcpy(data, data_device, size * sizeof(int)).wait();

  for (auto index = 0; index < size; ++index)
    std::cout << data[index] << '\n';

  sycl::free(data_device, queue);

  free(data);

  return 0;
}