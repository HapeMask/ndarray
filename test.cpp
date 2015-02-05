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
    return 0;
}
