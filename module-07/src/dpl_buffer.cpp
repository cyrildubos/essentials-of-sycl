#include <sycl/sycl.hpp>

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

int main() {
  sycl::queue queue;

  std::vector<int> vector({2, 3, 1, 4});

  sycl::buffer buffer(vector);

  auto buffer_begin = oneapi::dpl::begin(buffer);
  auto buffer_end = oneapi::dpl::end(buffer);

  oneapi::dpl::for_each(oneapi::dpl::execution::make_device_policy(queue),
                        buffer_begin, buffer_end,
                        [](auto& element) { element *= 3; });

  oneapi::dpl::sort(oneapi::dpl::execution::make_device_policy(queue),
                    buffer_begin, buffer_end);

  for (auto index = 0; index < vector.size(); ++index)
    std::cout << vector[index] << '\n';

  return 0;
}