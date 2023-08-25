#include <sycl/sycl.hpp>

constexpr std::size_t global_size = 1'024;
constexpr std::size_t group_size = 128;

int main() {
  sycl::queue queue;

  auto data = static_cast<int*>(malloc(global_size * sizeof(int)));

  for (auto index = 0; index < global_size; ++index)
    data[index] = index;

  auto sum = 0;

  sycl::buffer buffer_data(data, sycl::range(global_size));

  sycl::buffer buffer_sum(&sum, sycl::range(1));

  queue.submit([&](auto& handler) {
    sycl::accessor accessor_data(buffer_data, handler, sycl::read_only);

    // TODO: sycl::reduction
    auto reduction_sum = sycl::reduction(buffer_sum, sycl::plus<int>());

    handler.parallel_for(sycl::nd_range<1>(global_size, group_size),
                         reduction_sum, [=](auto item, auto& sum) {
                           auto identifier = item.get_global_id(0);

                           sum.combine(accessor_data[identifier]);
                         });
  });

  std::cout << "sum = " << sum << '\n';

  return 0;
}