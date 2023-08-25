#include <sycl/sycl.hpp>

constexpr std::size_t size = 42;

int main() {
  sycl::queue queue(sycl::accelerator_selector_v,
                    sycl::property::queue::in_order());

  int* data = sycl::malloc_shared<int>(size, queue);

  auto event =
      queue.parallel_for(size, [=](auto identifier) { data[identifier] = 1; });

  queue
      .submit([&](auto& handler) {
        handler.depends_on(event);

        handler.single_task([=](auto identifier) {
          for (auto index = 1; index < size; ++index)
            data[0] += data[index];
        });
      })
      .wait();

  std::cout << data[0] << '\n';

  return 0;
}