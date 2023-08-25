#include <sycl/sycl.hpp>

constexpr std::size_t size = 16;

void execute(std::vector<int>& vector, sycl::queue& queue) {
  sycl::range<1> range(size);

  sycl::buffer buffer(vector);

  queue.submit([&](auto& handler) {
    sycl::accessor accessor(buffer, handler, sycl::write_only);

    handler.parallel_for(range,
                         [=](auto identifier) { accessor[identifier] -= 2; });
  });
}

int main() {
  sycl::queue queue;

  std::vector<int> vector(size, 20);

  execute(vector, queue);

  for (auto index = 0; index < size; ++index)
    std::cout << vector[index] << '\n';

  return 0;
}