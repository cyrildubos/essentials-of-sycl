#include <sycl/sycl.hpp>

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

constexpr std::size_t size = 4;

int main() {
  sycl::queue queue;

  sycl::usm_allocator<int, sycl::usm::alloc::shared> allocator(queue);

  std::vector<int, decltype(allocator)> vector(size, allocator);

  oneapi::dpl::fill(oneapi::dpl::execution::make_device_policy(queue),
                    vector.begin(), vector.end(), 20);

  for (auto index = 0; index < size; ++index)
    std::cout << vector[index] << '\n';

  return 0;
}