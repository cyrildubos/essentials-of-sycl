#include <sycl/sycl.hpp>

constexpr std::size_t global_size = 1'024;
constexpr std::size_t group_size = 128;

int main() {
  sycl::queue queue;

  auto data = sycl::malloc_shared<int>(global_size, queue);

  for (auto index = 0; index < global_size; ++index)
    data[index] = index;

  auto min = sycl::malloc_shared<int>(1, queue);
  auto max = sycl::malloc_shared<int>(1, queue);

  min[0] = 0;
  max[0] = 0;

  auto reduction_min = sycl::reduction(min, sycl::minimum<int>());
  auto reduction_max = sycl::reduction(min, sycl::maximum<int>());

  queue
      .submit([&](auto& handler) {
        handler.parallel_for(sycl::nd_range<1>(global_size, group_size),
                             reduction_min, reduction_max,
                             [=](auto item, auto& min, auto& max) {
                               auto identifier = item.get_global_id();

                               min.combine(data[identifier]);
                               max.combine(data[identifier]);
                             });
      })
      .wait();

  auto mid_range = (min[0] + max[0]) / 2;

  std::cout << "mid_range = " << mid_range << '\n';

  sycl::free(data, queue);

  sycl::free(min, queue);
  sycl::free(max, queue);

  return 0;
}