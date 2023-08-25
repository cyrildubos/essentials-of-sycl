#include <sycl/sycl.hpp>

constexpr std::size_t size = 42;

int main() {
  sycl::queue queue(sycl::accelerator_selector_v,
                    sycl::property::queue::in_order());

  int* data_1 = sycl::malloc_shared<int>(size, queue);
  int* data_2 = sycl::malloc_shared<int>(size, queue);

  queue.parallel_for(size, [=](auto identifier) { data_1[identifier] = 1; });
  queue.parallel_for(size, [=](auto identifier) { data_2[identifier] = 2; });

  queue.parallel_for(
      size, [=](auto identifier) { data_1[identifier] += data_2[identifier]; });

  queue
      .single_task([=]() {
        for (auto index = 1; index < size; ++index)
          data_1[0] += data_1[index];

        data_1[0] /= 3;
      })
      .wait();

  std::cout << data_1[0] << '\n';

  return 0;
}