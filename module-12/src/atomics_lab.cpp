#include <sycl/sycl.hpp>

constexpr std::size_t size = 1'024;

int main() {
  sycl::queue queue;

  auto data = sycl::malloc_shared<int>(size, queue);

  for (auto index = 0; index < size; ++index)
    data[index] = index;

  auto minimum = sycl::malloc_shared<int>(1, queue);
  auto maximum = sycl::malloc_shared<int>(1, queue);

  minimum[0] = 0;
  maximum[0] = 0;

  queue
      .parallel_for(
          size,
          [=](auto identifier) {
            auto atomic_minimum =
                sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>(
                    minimum[0]);

            auto atomic_maximum =
                sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>(
                    maximum[0]);

            if (data[identifier] < atomic_minimum)
              atomic_minimum = data[identifier];

            if (data[identifier] > atomic_maximum)
              atomic_maximum = data[identifier];
          })
      .wait();

  auto mid_range = (maximum[0] - minimum[0]) / 2;

  std::cout << "minimum   = " << minimum[0] << '\n';
  std::cout << "maximum   = " << maximum[0] << '\n';
  std::cout << "mid_range = " << mid_range << '\n';

  return 0;
}