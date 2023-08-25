#include <sycl/sycl.hpp>

constexpr std::size_t size = 1'024;

int main() {
  sycl::queue queue;

  auto data = sycl::malloc_shared<int>(size, queue);

  for (auto index = 0; index < size; ++index)
    data[index] = index;

  auto sum = sycl::malloc_shared<int>(1, queue);

  sum[0] = 0;

  queue
      .parallel_for(
          size,
          [=](auto identifier) {
            auto atomic =
                sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>(
                    sum[0]);

            atomic += data[identifier];
          })
      .wait();

  std::cout << "sum = " << sum[0] << '\n';

  return 0;
}