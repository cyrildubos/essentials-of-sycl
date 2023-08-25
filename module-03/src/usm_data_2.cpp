#include <sycl/sycl.hpp>

constexpr std::size_t size = 16;

int main() {
  sycl::queue queue;

  auto data_1 = sycl::malloc_shared<int>(size, queue);
  auto data_2 = sycl::malloc_shared<int>(size, queue);

  for (auto index = 0; index < size; ++index) {
    data_1[index] = 10;
    data_2[index] = 10;
  }

  sycl::range<1> range(size);

  queue.parallel_for(range, [=](auto identifier) { data_1[identifier] += 2; });
  queue.parallel_for(range, [=](auto identifier) { data_2[identifier] += 3; });

  queue
      .parallel_for(
          range,
          [=](auto identifier) { data_1[identifier] += data_2[identifier]; })
      .wait();

  for (auto index = 0; index < size; ++index)
    std::cout << data_1[index] << '\n';

  sycl::free(data_1, queue);
  sycl::free(data_2, queue);

  return 0;
}