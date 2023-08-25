#include <sycl/sycl.hpp>

constexpr std::size_t size = 1'024;

int main() {
  sycl::queue queue;

  auto data = sycl::malloc_shared<int>(size, queue);

  for (auto index = 0; index < size; ++index)
    data[index] = index;

  for (auto index = 0; index < size; ++index)
    std::cout << data[index] << '\n';

  queue
      .single_task([=]() {
        auto sum = 0;

        for (auto index = 0; index < size; ++index)
          sum += data[index];

        data[0] = sum;
      })
      .wait();

  std::cout << "sum = " << data[0] << '\n';

  sycl::free(data, queue);

  return 0;
}