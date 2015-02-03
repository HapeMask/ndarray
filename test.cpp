#include <iostream>
#include <tuple>
#include <typeinfo>

template <typename T, size_t S>
std::ostream& operator<<(std::ostream& out, const std::array<T, S>& arr) {
    out << "( ";
    for (const auto& i : arr) { out << i << " "; }
    out << ")";
    return out;
}

#include "ndarray.hpp"
#include "ndmath.hpp"

using nda::range;
using nda::slice;
using nda::ndarray;

int main(int argc, char** args) {
    ndarray<float, 3, 3> x;

    for(int i=0; i<3; ++i) {
    for(int j=0; j<3; ++j) {
        x(i, j) = 3*i + j;
    }}

    auto sl = x(all(), slice(0,3,2));

    std::cerr << sl.shape() << std::endl;
    std::cerr << sl.size() << std::endl;

    ndarray<float, 3, 2> z = 2*(-exp(sl+2));
    std::cerr << x << std::endl;
    std::cerr << std::endl;
    std::cerr << z << std::endl;

    return 0;
}
