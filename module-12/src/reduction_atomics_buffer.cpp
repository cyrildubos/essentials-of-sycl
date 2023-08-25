#include <sycl/sycl.hpp>

constexpr std::size_t size = 1'024;

int main() {
  sycl::queue queue;

  std::vector<int> data(size);

  for (auto index = 0; index < size; ++index)
    data[index] = index;

  auto sum = 0;

  sycl::buffer buffer_data(data);
  sycl::buffer buffer_sum(&sum, sycl::range(1));

  queue
      .submit([&](auto& handler) {
        sycl::accessor accessor_data(buffer_data, handler, sycl::read_only);
        sycl::accessor accessor_sum(buffer_sum, handler);

        handler.parallel_for(size, [=](auto identifier) {
          auto atomic =
              sycl::atomic_ref<int, sycl::memory_order::relaxed,
                               sycl::memory_scope::device,
                               sycl::access::address_space::global_space>(
                  accessor_sum[0]);

          atomic += accessor_data[identifier];
        });
      })
      .wait();

  std::cout << "sum = " << sum << '\n';

  return 0;
}