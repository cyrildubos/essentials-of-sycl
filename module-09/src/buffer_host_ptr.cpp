#include <sycl/sycl.hpp>

#include <mutex>

constexpr std::size_t size = 20;

int main() {
  sycl::queue queue;

  // int data[42];

  std::vector<float> vector_a(size, 10.0f);
  std::vector<float> vector_b(size, 20.0f);

  // TODO
  sycl::buffer buffer_a(vector_a, {sycl::property::buffer::use_host_ptr()});
  sycl::buffer buffer_b(vector_b, {sycl::property::buffer::use_host_ptr()});

  queue.submit([&](auto& handler) {
    sycl::accessor accessor_a(buffer_a, handler);
    sycl::accessor accessor_b(buffer_b, handler);

    handler.parallel_for(sycl::range<1>(size), [=](auto identifier) {
      accessor_a[identifier] += accessor_b[1];
    });
  });

  for (auto index = 0; index < size; ++index)
    std::cout << vector_a[index] << '\n';

  return 0;
}