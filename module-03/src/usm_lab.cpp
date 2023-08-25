#include <sycl/sycl.hpp>

constexpr std::size_t size = 1'024;

int main() {
  sycl::queue queue;

  auto data_1 = static_cast<int*>(malloc(size * sizeof(int)));
  auto data_2 = static_cast<int*>(malloc(size * sizeof(int)));

  for (auto index = 0; index < size; ++index) {
    data_1[index] = 25;
    data_2[index] = 49;
  }

  auto data_1_device = sycl::malloc_device<int>(size, queue);
  auto data_2_device = sycl::malloc_device<int>(size, queue);

  queue.memcpy(data_1_device, data_1, size * sizeof(int));
  queue.memcpy(data_2_device, data_2, size * sizeof(int));

  queue.wait();

  sycl::range<1> range(size);

  queue.parallel_for(range, [=](auto identifier) {
    data_1_device[identifier] = std::sqrt(data_1_device[identifier]);
  });
  queue.parallel_for(range, [=](auto identifier) {
    data_2_device[identifier] = std::sqrt(data_2_device[identifier]);
  });

  queue.wait();

  queue
      .parallel_for(range,
                    [=](auto identifier) {
                      data_1_device[identifier] += data_2_device[identifier];
                    })
      .wait();

  queue.memcpy(data_1, data_1_device, size * sizeof(int)).wait();

  for (auto index = 0; index < size; ++index)
    std::cout << data_1[index] << '\n';

  sycl::free(data_1_device, queue);
  sycl::free(data_2_device, queue);

  free(data_1);
  free(data_2);

  return 0;
}