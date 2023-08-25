#include <sycl/sycl.hpp>

constexpr std::size_t size = 16;

int main() {
  sycl::queue queue;

  auto data = sycl::malloc_shared<int>(size, queue);

  for (auto index = 0; index < size; ++index)
    data[index] = index;

  queue
      .parallel_for(sycl::range<1>(size),
                    [=](auto identifier) { data[identifier] *= 2; })
      .wait();

  for (auto index = 0; index < size; ++index)
    std::cout << data[index] << '\n';

  sycl::free(data, queue);

  return 0;
}