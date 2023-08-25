#include <sycl/sycl.hpp>

constexpr std::size_t global_size = 1'024;
constexpr std::size_t group_size = 128;

int main() {
  sycl::queue queue;

  auto data = sycl::malloc_shared<int>(global_size, queue);
  auto sum = sycl::malloc_shared<int>(1, queue);

  for (auto index = 0; index < global_size; ++index)
    data[index] = index;

  sum[0] = 0;

  queue
      .parallel_for(sycl::nd_range<1>(global_size, group_size),
                    sycl::reduction(sum, sycl::plus<int>()),
                    [=](auto item, auto& sum) {
                      auto identifier = item.get_global_id(0);

                      sum.combine(data[identifier]);
                    })
      .wait();

  std::cout << "sum = " << sum[0] << '\n';

  sycl::free(data, queue);
  sycl::free(sum, queue);

  return 0;
}