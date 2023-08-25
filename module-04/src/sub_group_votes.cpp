#include <sycl/sycl.hpp>

constexpr std::size_t global_size = 32;
constexpr std::size_t group_size = 16;

int main() {
  sycl::queue queue;

  auto data = sycl::malloc_shared<int>(global_size, queue);
  auto all = sycl::malloc_shared<int>(global_size, queue);
  auto any = sycl::malloc_shared<int>(global_size, queue);
  auto none = sycl::malloc_shared<int>(global_size, queue);

  for (auto index = 0; index < global_size; ++index)
    data[index] = index < 10 ? 0 : index;

  for (auto index = 0; index < global_size; ++index)
    std::cout << data[index] << '\n';

  queue
      .parallel_for(
          sycl::nd_range<1>(global_size, group_size),
          [=](auto item) {
            auto sub_group = item.get_sub_group();
            auto identifier = item.get_global_id(0);

            all[identifier] = sycl::all_of_group(sub_group, data[identifier]);
            any[identifier] = sycl::any_of_group(sub_group, data[identifier]);
            none[identifier] = sycl::none_of_group(sub_group, data[identifier]);
          })
      .wait();

  std::cout << "all_of_group:\n";

  for (auto index = 0; index < global_size; ++index)
    std::cout << all[index] << '\n';

  std::cout << "any_of_group:\n";

  for (auto index = 0; index < global_size; ++index)
    std::cout << any[index] << '\n';

  std::cout << "none_of_group:\n";

  for (auto index = 0; index < global_size; ++index)
    std::cout << none[index] << '\n';

  sycl::free(data, queue);
  sycl::free(all, queue);
  sycl::free(any, queue);
  sycl::free(none, queue);

  return 0;
}