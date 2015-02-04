#pragma once

#include <iostream>
#include <cstring>
#include <array>
#include <cassert>
#include <type_traits>

#include "type_tools.hpp"
#include "slicing.hpp"

//////////////////////
// RANGE
//////////////////////

namespace nda {

/*
 * Ensures instances of any two types with shape() methods have matching
 * shapes.
 */
template <typename Arr1, typename Arr2>
bool shape_match(const Arr1& s1, const Arr2& s2) {
    assert(s1.size() == s2.size());
    bool match = true;
    for(size_t i=0; i < s1.size(); ++i) {
        match = match && (s1[i] == s2[i]);
    }
    return match;
}

//////////////////////
// NDARRAY
//////////////////////

/*
 * Base n-dimensional array class. Shape is defined by a static ShapePack-like
 * struct encapsulating the variadic shapes.
 */
template <typename T, typename ShapePack>
class nda_impl {
    public:
        using value_type = T;
        using shape_type = ShapePack;
        static constexpr bool _is_array = true;

        static constexpr size_t ndim = ShapePack::len;
        size_t _size;
        std::array<size_t, ndim> _strides;
        std::array<size_t, ndim> _shape;

        const std::array<size_t, ndim>& shape() const { return _shape; }
        const std::array<size_t, ndim>& strides() const { return _strides; }
        const size_t& size() const { return _size; }

        T* data = nullptr;

        ////////////////////////
        //  Default Constructor
        ////////////////////////
        nda_impl() {
            static_assert(n_dynamic_dims == 0, "Must specify all dynamic dimensions for construction.");

            fill_array<ShapePack>(_shape);
            _compute_size();
            _compute_basic_strides();
            _alloc_data();
        }

        //////////////////////////////////
        //  Initializer-list Constructor
        //////////////////////////////////
        nda_impl(const std::initializer_list<T>& items) {
            static_assert(n_dynamic_dims < 2,
                    "Cannot construct arrays with multiple dynamic dimensions from an initializer list (yet).");

            fill_array<ShapePack>(_shape);

            size_t sz = 1;
            size_t dyn_dim = 0;
            for(size_t i=0; i<ndim; ++i) {
                if(_shape[i] == 0) { dyn_dim = i; } else { sz *= _shape[i]; }
            }

            if (n_dynamic_dims == 1) {
                _shape[dyn_dim] = items.size() / sz;
            }

            _compute_size();
            assert(_size == items.size());
            _compute_basic_strides();
            _alloc_data(items);
        }

        //////////////////////
        //  Copy Constructors
        //////////////////////
        /*
         * Two copy constructors are needed, one "true" copy constructor for
         * other instances of nda_impl with the same (possibly dynamic) shape,
         * and one for instances with other (compatible) shapes. The same holds
         * for move constructors and assignment operators.
         */
        nda_impl(const nda_impl& other) : _size(other._size), _strides(other._strides), _shape(other._shape)
        {
            if(!has_fixed_shape) {
                assert(shape_match(_shape, other._shape));
            }

            _alloc_data();
            memcpy(data, other.data, _size*sizeof(T));
        }
        template <typename OShape>
        nda_impl(const nda_impl<T, OShape>& other) : _size(other._size), _strides(other._strides), _shape(other._shape)
        {
            static_assert(elementwise_compatible<ShapePack, OShape>::value,
                    "Shapes must match for all fixed dimensions.");

            if(!has_fixed_shape) {
                assert(shape_match(_shape, other._shape));
            }

            _alloc_data();
            memcpy(data, other.data, _size*sizeof(T));
        }

        //////////////////////
        //  Move Constructors
        //////////////////////
        nda_impl(nda_impl&& other) : _size(other._size),
                                     _strides(other._strides),
                                     _shape(other._shape),
                                     data(other.data)
        {
            if(!has_fixed_shape) {
                assert(shape_match(_shape, other._shape));
            }

            other.data = nullptr;
        }

        template <typename OShape>
        nda_impl(nda_impl<T, OShape>&& other) : _size(other._size),
                                                _strides(other._strides),
                                                _shape(other._shape),
                                                data(other.data)
        {
            static_assert(elementwise_compatible<ShapePack, OShape>::value,
                    "Shapes must match for all known dimensions.");

            if(!has_fixed_shape) {
                assert(shape_match(_shape, other._shape));
            }

            other.data = nullptr;
        }

        ///////////////////////////////
        //  Dynamic Shape Constructor
        ///////////////////////////////
        template <typename... DynShapes, typename = std::enable_if_t<all_integral<DynShapes...>::value>>
        nda_impl(const DynShapes&... dynamic_shapes) {
            static_assert(n_dynamic_dims > 0, "Tried to call dynamic shape constructor with no dynamic dims.");
            static_assert(sizeof...(DynShapes) == n_dynamic_dims, "Must specify all dynamic dimensions for construction.");

            fill_array<ShapePack>(_shape);
            _fill_dynamic_shapes(dynamic_shapes...);
            _compute_size();
            _compute_basic_strides();
            _alloc_data();
        }

        ////////////////////////////////////
        //  Expression Template Constructor
        ////////////////////////////////////
        template <typename Expr,
                 enable_if_expression(Expr),
                 enable_if_compatible(ShapePack, typename Expr::shape_type)>
        nda_impl(const Expr& ex) {
            fill_array<ShapePack>(_shape);
            for(int i=0; i<ndim; ++i) {
                if (_shape[i] == DYNAMIC_SHAPE) { _shape[i] = ex.shape()[i]; }
            }
            _compute_size();

            assert(shape_match(_shape, ex.shape()));
            assert(_size == ex.size());

            _compute_basic_strides();
            _alloc_data();

            auto it = begin();
            for(auto x : ex){
                (*it) = x;
                ++it;
            }

        }

        ~nda_impl() {
            if (data) {
                delete[] data;
            }
        }

        ///////////////////////////////////
        // Copy/Move Assignment Operators
        ///////////////////////////////////
        nda_impl& operator=(const nda_impl& other) {
            if(!has_fixed_shape) {
                assert(shape_match(_shape, other._shape));
            }

            memcpy(data, other.data, _size*sizeof(T));
            return *this;
        }
        template <typename OShape>
        nda_impl& operator=(const nda_impl<T, OShape>& other) {
            static_assert(elementwise_compatible<ShapePack, OShape>::value,
                    "Shapes must match for all known dimensions.");

            if(!has_fixed_shape) {
                assert(shape_match(_shape, other._shape));
            }

            memcpy(data, other.data, _size*sizeof(T));
            return *this;
        }

        nda_impl& operator=(nda_impl&& other) {
            if(!has_fixed_shape) {
                assert(shape_match(_shape, other._shape));
            }

            if(this != &other) {
                data = other.data;
                other.data = nullptr;
            }
            return *this;
        }
        template <typename OShape>
        nda_impl& operator=(nda_impl<T, OShape>&& other) {
            static_assert(elementwise_compatible<ShapePack, OShape>::value,
                    "Shapes must match for all known dimensions.");

            if(!has_fixed_shape) {
                assert(shape_match(_shape, other._shape));
            }

            if(this != &other) {
                data = other.data;
                other.data = nullptr;
            }
            return *this;
        }

        ////////////////////////////////////
        //  Expression Template Assignment
        ////////////////////////////////////
        template <typename Expr, enable_if_expression(Expr)>
        nda_impl& operator=(const Expr& ex) {
            fill_array<ShapePack>(_shape);
            for(int i=0; i<ndim; ++i) {
                if (_shape[i] == DYNAMIC_SHAPE) { _shape[i] = ex.shape()[i]; }
            }
            _compute_size();
            assert(shape_match(_shape, ex.shape()));
            assert(_size == ex.size());

            _compute_basic_strides();
            _alloc_data();

            for(size_t i=0; i < _size; ++i) {
                data[i] = ex(i);
            }
            return *this;
        }

        ///////////////////////
        //  Scalar Assignment
        ///////////////////////
        nda_impl& operator=(const T& s) {
            for(int i=0; i<_size; ++i) {
                data[i] = s;
            }
            return *this;
        }

        /////////////////////
        //  Access Operators
        /////////////////////
        template <typename... Indices, typename = std::enable_if_t<sizeof...(Indices) == ndim && all_integral<Indices...>::value>>
        const T& operator()(const Indices&... indices) const {
            static_assert(sizeof...(Indices) == ndim, "Incorrect number of indices.");
            return data[_compute_index(indices...)];
        }
        template <typename... Indices, typename = std::enable_if_t<sizeof...(Indices) == ndim && all_integral<Indices...>::value>>
        T& operator()(const Indices&... indices) {
            return data[_compute_index(indices...)];
        }
        const T& operator()(const std::array<size_t, ndim>& ind_arr) const {
            return data[_compute_index(ind_arr)];
        }
        T& operator()(const std::array<size_t, ndim>& ind_arr) {
            return data[_compute_index(ind_arr)];
        }

        /*
         * Slice access operator (returns a slice expression).
         */
        template <typename... Slices, typename = std::enable_if_t<!all_integral<Slices...>::value || (sizeof...(Slices) < ndim && all_integral_or_slice<Slices...>::value)>>
        slice_expr<const nda_impl<T, ShapePack>, Slices...> operator()(const Slices&... slices) const {
            return {*this, slices...};
        }
        template <typename... Slices, typename = std::enable_if_t<!all_integral<Slices...>::value || (sizeof...(Slices) < ndim && all_integral_or_slice<Slices...>::value)>>
        slice_expr<nda_impl<T, ShapePack>, Slices...> operator()(const Slices&... slices) {
            return {*this, slices...};
        }

        //////////////////
        //  Iterators
        //////////////////
        using const_iterator = const T*;

        /*
         * The raw data pointer works fine as an iterator as long as the array
         * has not been sliced.
         */
        const_iterator begin() const { return data; }
        const_iterator end() const { return data + _size; }

        T* begin() { return data; }
        T* end() { return data + _size; }

        template <typename OShape>
        nda_impl& operator+=(const nda_impl<T, OShape>& rhs) {
            auto it = begin();
            auto r_it = rhs.begin();
            for(; it != end() && r_it != rhs.end(); ++it, ++r_it) {
                (*it) += (*r_it);
            }
            return (*this);
        }

    private:
        static constexpr size_t n_dynamic_dims = count_in<DYNAMIC_SHAPE, ShapePack>::value;
        static constexpr size_t n_static_dims = ndim - n_dynamic_dims;
        static constexpr bool has_fixed_shape = (n_dynamic_dims == 0);

        /*
         * Given a list of runtime shapes, fill in the missing dynamic values
         * in the _shapes array.
         */
        template <typename... DynShapes>
        void _fill_dynamic_shapes(const DynShapes&... dynamic_shapes) {
            static_assert(sizeof...(dynamic_shapes) == n_dynamic_dims, "Invalid number of dynamic dimensions.");
            __fill_dynamic_shapes<0>(dynamic_shapes...);
        }
        template <size_t si, typename... DynShapes>
        void __fill_dynamic_shapes(const size_t& s0, const DynShapes&... dynamic_shapes) {
            __fill_dynamic_shapes<si>(s0);
            __fill_dynamic_shapes<si+1>(dynamic_shapes...);
        }
        template <size_t si>
        void __fill_dynamic_shapes(const size_t& s0) {
            _shape[nth_match_index_in_pack<si, DYNAMIC_SHAPE, shape_type>::value] = s0;
        }

        void _alloc_data() {
            if(data) { delete[] data; }
            data = new T[_size];
        }

        void _alloc_data(const std::initializer_list<T>& items) {
            if(data) { delete[] data; }
            data = new T[items.size()];
            std::copy(items.begin(), items.end(), data);
        }

        void _compute_size() {
            _size = 1;
            for (const auto& si : _shape) { _size *= si; }
        }

        /*
         * Compute strides assuming the array has not been sliced.
         */
        void _compute_basic_strides() {
            _strides[ndim-1] = 1;
            for(size_t i=2; i<=ndim; ++i) {
                _strides[ndim-i] = _strides[ndim-i+1] * _shape[ndim-i+1];
            }
        }

        /*
         * Compute an offset into the data array given a list of indices.
         */
        size_t _compute_index(const std::array<size_t, ndim>& indices) const {
            int ind = 0;
            for(int i=0; i < ndim; ++i) {
                ind += _strides[i] * indices[i];
            }
            return ind;
        }
        template <typename... Indices>
        size_t _compute_index(const size_t& i0, const Indices&... indices) const {
            const size_t ind = i0 * _strides[ndim - sizeof...(Indices) - 1] + _compute_index(indices...);
            assert(ind < _size && "Index out of range.");
            return ind;
        }

        size_t _compute_index() const { return 0; }
};

template <typename T, size_t... Shape>
using ndarray = nda_impl<T, shape_pack<Shape...>>;

template <typename ArrExpr>
using array_like = nda_impl<typename ArrExpr::value_type, typename ArrExpr::shape_type>;
} // END namespace nda

template <typename Expr>
typename std::enable_if<is_array_or_expr(Expr) && Expr::ndim == 1,
std::ostream&>::type
operator<<(std::ostream& out, const Expr& expr) {
    out << "[";
    for(size_t i=0; i < expr.shape()[0]; ++i) {
        out << expr(i);
        if(i < expr.shape()[0]-1) { out << "    "; }
    }
    out << "]";
    return out;
}

template <typename Expr>
typename std::enable_if<is_array_or_expr(Expr) && Expr::ndim == 2,
std::ostream&>::type
operator<<(std::ostream& out, const Expr& expr) {
    out << "[";
    for(size_t i=0; i < expr.shape()[0]; ++i) {
        if(i > 0) {out << " ";}
        out << expr(i);
        if(i < expr.shape()[0]-1) { out << std::endl; }
    }
    out << "]";
    return out;
}

template <typename Expr>
typename std::enable_if<is_array_or_expr(Expr) && (Expr::ndim > 2),
std::ostream&>::type
operator<<(std::ostream& out, const Expr& expr) {
    for(size_t i=0; i < expr.shape()[0]; ++i) {
        out << "[";
        out << expr(i);
        out << "]";
        if(i < expr.shape()[0]-1) { out << std::endl; }
    }
    return out;
}
