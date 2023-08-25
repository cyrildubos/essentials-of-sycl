#include <sycl/sycl.hpp>

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

constexpr std::size_t size = 4;

using namespace oneapi::dpl::execution;

int main() {
  sycl::queue queue;

  std::vector<int> vector(size);

  oneapi::dpl::fill(oneapi::dpl::execution::make_device_policy(queue),
                    vector.begin(), vector.end(), 20);

  for (auto index = 0; index < size; ++index)
    std::cout << vector[index] << '\n';

  return 0;
}