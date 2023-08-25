#include <sycl/sycl.hpp>

constexpr std::size_t global_size = 256;
constexpr std::size_t local_size = 64;

int main() {
  sycl::queue queue;

  auto data = sycl::malloc_shared<int>(global_size, queue);

  for (auto index = 0; index < global_size; ++index)
    data[index] = index;

  for (auto index = 0; index < global_size; ++index)
    std::cout << data[index] << '\n';

  queue
      .parallel_for(sycl::nd_range<1>(global_size, local_size),
                    [=](auto item) {
                      auto sub_group = item.get_sub_group();
                      auto identifier = item.get_global_id();

                      auto result = sycl::reduce_over_group(
                          sub_group, data[identifier], sycl::plus<int>());

                      data[identifier] =
                          sub_group.get_local_id()[0] == 0 ? result : 0;
                    })
      .wait();

  for (auto index = 0; index < global_size; ++index)
    std::cout << data[index] << '\n';

  sycl::free(data, queue);

  return 0;
}