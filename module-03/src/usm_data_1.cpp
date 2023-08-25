#include <sycl/sycl.hpp>

constexpr std::size_t size = 16;

int main() {
  sycl::queue queue;

  auto data = sycl::malloc_shared<int>(size, queue);

  for (auto index = 0; index < size; ++index)
    data[index] = 10;

  sycl::range<1> range(size);

  queue.parallel_for(range, [=](auto identifier) { data[identifier] += 2; });
  queue.parallel_for(range, [=](auto identifier) { data[identifier] += 3; });
  queue.parallel_for(range, [=](auto identifier) { data[identifier] += 5; });

  queue.wait();

  for (auto index = 0; index < size; ++index)
    std::cout << data[index] << '\n';

  sycl::free(data, queue);

  return 0;
}