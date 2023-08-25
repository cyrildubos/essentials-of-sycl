#include <sycl/sycl.hpp>

constexpr std::size_t global_size = 64;
constexpr std::size_t local_size = 64;

int main() {
  sycl::queue queue;

  queue
      .submit([&](auto& handler) {
        sycl::stream stream(1'024, 768, handler);

        handler.parallel_for(
            sycl::nd_range<1>(global_size, local_size), [=](auto item) {
              auto sub_group = item.get_sub_group();

              if (sub_group.get_local_id()[0] == 0) {
                stream << "sub_group id: " << sub_group.get_group_id()[0]
                       << " of " << sub_group.get_group_range()[0]
                       << ", size=" << sub_group.get_local_range()[0] << '\n';
              }
            });
      })
      .wait();

  return 0;
}