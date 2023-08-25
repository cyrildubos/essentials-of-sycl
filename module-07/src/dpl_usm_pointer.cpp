#include <sycl/sycl.hpp>

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

constexpr std::size_t size = 4;

int main() {
  sycl::queue queue;

  auto data = sycl::malloc_shared<int>(size, queue);

  oneapi::dpl::fill(oneapi::dpl::execution::make_device_policy(queue), data,
                    data + size, 20);

  for (auto index = 0; index < size; ++index)
    std::cout << data[index] << '\n';

  sycl::free(data, queue);

  return 0;
}