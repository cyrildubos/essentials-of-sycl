#include <sycl/sycl.hpp>

constexpr std::size_t global_size = 1'024;
constexpr std::size_t group_size = 128;

int main() {
  sycl::queue queue;

  auto data = static_cast<int*>(malloc(global_size * sizeof(int)));

  for (auto index = 0; index < global_size; ++index)
    data[index] = index;

  auto sum = 0;
  auto min = 0;
  auto max = 0;

  sycl::buffer buffer_data(data, sycl::range(global_size));

  sycl::buffer buffer_sum(&sum, sycl::range(1));
  sycl::buffer buffer_min(&min, sycl::range(1));
  sycl::buffer buffer_max(&max, sycl::range(1));

  queue
      .submit([&](auto& handler) {
        sycl::accessor accessor_data(buffer_data, handler, sycl::read_only);

        auto reduction_sum =
            sycl::reduction(buffer_sum, handler, sycl::plus<int>());
        auto reduction_min =
            sycl::reduction(buffer_min, handler, sycl::minimum<int>());
        auto reduction_max =
            sycl::reduction(buffer_max, handler, sycl::maximum<int>());

        handler.parallel_for(sycl::nd_range<1>(global_size, group_size),
                             reduction_sum, reduction_min, reduction_max,
                             [=](auto item, auto& sum, auto& min, auto& max) {
                               auto identifier = item.get_global_id();

                               sum.combine(accessor_data[identifier]);
                               min.combine(accessor_data[identifier]);
                               max.combine(accessor_data[identifier]);
                             });
      })
      .wait();

  std::cout << "sum = " << sum << '\n';
  std::cout << "min = " << min << '\n';
  std::cout << "max = " << max << '\n';

  return 0;
}
