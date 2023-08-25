#include <sycl/sycl.hpp>

constexpr std::size_t global_size = 1'024;
constexpr std::size_t group_size = 256;
constexpr std::size_t sub_group_size = 32;

int main() {
  sycl::queue queue;

  auto data = sycl::malloc_shared<int>(global_size, queue);
  auto sub_group_data =
      sycl::malloc_shared<int>(global_size / sub_group_size, queue);

  for (auto index = 0; index < global_size; ++index)
    data[index] = index;

  // for (auto index = 0; index < global_size; ++index)
  //   std::cout << data[index] << '\n';

  queue
      .parallel_for(sycl::nd_range<1>(global_size, group_size),
                    [=](auto item)
                        [[intel::reqd_sub_group_size(sub_group_size)]] {
                          auto sub_group = item.get_sub_group();
                          auto identifier = item.get_global_id(0);

                          auto sum = sycl::reduce_over_group(
                              sub_group, data[identifier], sycl::plus<int>());

                          sub_group_data[identifier / sub_group_size] = sum;
                        })
      .wait();

  for (auto index = 0; index < global_size / sub_group_size; ++index)
    std::cout << sub_group_data[index] << '\n';

  auto sum = 0;

  for (auto index = 0; index < global_size / sub_group_size; ++index)
    sum += sub_group_data[index];

  std::cout << "sum = " << sum << '\n';

  sycl::free(data, queue);
  sycl::free(sub_group_data, queue);

  return 0;
}