#include <sycl/sycl.hpp>

constexpr std::size_t global_size = 1'024;
constexpr std::size_t group_size = 128;

template <typename T> struct entry {
  bool operator<(const entry& e) const {
    return value <= e.value || (value == e.value && index <= e.index);
  }

  int index;
  T value;
};

int main() {
  // sycl::queue queue;

  // auto data = sycl::malloc_shared<int>(global_size, queue);

  // for (auto index = 0; index < global_size; ++index)
  //   data[index] = index;

  // for (auto index = 0; index < global_size; ++index)
  //   std::cout << data[index] << '\n';

  // auto result = sycl::malloc_shared<entry<int>>(1, queue);

  // entry<int> identity(0, 1);

  // result[0] = identity;

  // auto reduction =
  //     sycl::reduction(result, identity, sycl::minimum<std::pair<int,
  //     int>>());

  // queue
  //     .parallel_for(sycl::nd_range<1>(global_size, group_size), reduction,
  //                   [=](auto item, auto& result) {
  //                     auto identifier = item.get_global_id();

  //                     result.combine(entry<int>(identifier,
  //                     data[identifier]));
  //                   })
  //     .wait();

  // std::cout << "index = " << result->index << '\n';
  // std::cout << "value = " << result->value << '\n';

  return 0;
}