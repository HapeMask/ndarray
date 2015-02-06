#include <iostream>
#include <tuple>
#include <typeinfo>
#include <chrono>

template <typename T, size_t S>
std::ostream& operator<<(std::ostream& out, const std::array<T, S>& arr) {
    out << "( ";
    for (const auto& i : arr) { out << i << " "; }
    out << ")";
    return out;
}

#include "ndarray.hpp"

using nda::range;
using nda::slice;
using nda::ndarray;

int main(int argc, char** args) {
    ndarray<float, 3, 3, 3> x;
    for(int i=0; i<3; ++i){
    for(int j=0; j<3; ++j){
    for(int k=0; k<3; ++k){
        x(i, j, k) = 9*i+3*j+k;
    }}}

    nda::nda_impl<float, universal_shape_pack<3>> y = x;
    std::cerr << y << std::endl;
    return 0;
}
