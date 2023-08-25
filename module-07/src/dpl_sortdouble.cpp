#include <sycl/sycl.hpp>

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

int main() {
  sycl::queue queue;

  std::vector<int> vector({2, 3, 1, 4});

  oneapi::dpl::for_each(oneapi::dpl::execution::make_device_policy(queue),
                        vector.begin(), vector.end(),
                        [](auto& element) { element *= 2; });

  oneapi::dpl::sort(oneapi::dpl::execution::make_device_policy(queue),
                    vector.begin(), vector.end());

  for (auto index = 0; index < vector.size(); ++index)
    std::cout << vector[index] << '\n';

  return 0;
}