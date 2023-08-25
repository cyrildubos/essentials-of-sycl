#include <sycl/sycl.hpp>

constexpr std::size_t size = 42;

int main() {
  std::array<int, size> array;

  for (auto index = 0; index < size; ++index)
    array[index] = index;

  auto pointer = std::make_shared<std::array<int, size>>();

  {
    sycl::queue queue;

    sycl::buffer buffer(array);

    buffer.set_final_data(pointer);

    queue.submit([&](auto& handler) {
      sycl::accessor accessor(buffer, handler);

      handler.parallel_for(size,
                           [=](auto identifier) { accessor[identifier] *= 2; });
    });
  }

  for (auto index = 0; index < size; ++index)
    std::cout << array[index] << '\n';

  for (auto index = 0; index < size; ++index)
    std::cout << (*pointer)[index] << '\n';

  return 0;
}