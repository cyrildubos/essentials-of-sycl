#include <sycl/sycl.hpp>

constexpr std::size_t size = 42;

int main() {
  std::array<int, size> array;

  for (auto index = 0; index < size; ++index)
    array[index] = index;

  {
    sycl::queue queue;

    sycl::buffer buffer(array);

    // TODO: does not work
    buffer.set_write_back(false);

    queue.submit([&](auto& handler) {
      sycl::accessor accessor(buffer, handler);

      handler.parallel_for(size,
                           [=](auto identifier) { accessor[identifier] *= 2; });
    });
  }

  for (auto index = 0; index < size; ++index)
    std::cout << array[index] << '\n';

  return 0;
}