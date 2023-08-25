#include <sycl/sycl.hpp>

constexpr std::size_t global_size = 1'024;
constexpr std::size_t group_size = 128;

int main() {
  sycl::queue queue;

  auto data = sycl::malloc_shared<int>(global_size, queue);

  for (auto index = 0; index < global_size; ++index)
    data[index] = index;

  queue.parallel_for(
      sycl::nd_range<1>(global_size, group_size), [=](auto item) {
        auto group = item.get_group();
        auto identifier = item.get_global_id(0);

        auto sum =
            sycl::reduce_over_group(group, data[identifier], sycl::plus<int>());

        data[identifier] = item.get_local_id(0) == 0 ? sum : 0;
      });

  queue
      .single_task([=]() {
        auto sum = 0;

        for (auto index = 0; index < global_size; index += group_size)
          sum += data[index];

        data[0] = sum;
      })
      .wait();

  std::cout << "sum = " << data[0] << '\n';

  sycl::free(data, queue);

  return 0;
}