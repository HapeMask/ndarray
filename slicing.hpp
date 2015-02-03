#pragma once

namespace nda {
/*
 * Range object, used for sliced indexing and index generation in for loops.
 */
struct range {
    int start, stop, step;

    range() : start(-1), stop(-1), step(-1) {}
    range(const all& a) : start(-1), stop(-1), step(-1) {}
    explicit range(int sto) : start(0), stop(sto), step(1) {}
    range(int sta, int sto, int ste=1) : start(sta), stop(sto), step(ste) {}

    struct rng_iter {
        const range& rng;
        int cur;

        rng_iter(const range& r) : rng(r), cur(r.start) {}
        rng_iter(const range& r, int init) : rng(r), cur(init) {}

        const int& operator*() const { return cur; }

        rng_iter& operator++() { cur += rng.step; return *this; }
        rng_iter operator++(int) { rng_iter tmp(*this); cur += rng.step; return tmp; }

        rng_iter& operator+=(int x) { cur += x*rng.step; return *this; }
        rng_iter operator+(int x) const { return rng_iter(*this) += x; }
        rng_iter& operator-=(int x) { cur -= x*rng.step; return *this; }
        rng_iter operator-(int x) const { return rng_iter(*this) -= x; }

        bool operator==(const rng_iter& other) const { return other.cur == cur && other.rng == rng; }
        bool operator!=(const rng_iter& other) const { return !(this->operator==(other)); }
    };

    rng_iter begin() const { return rng_iter(*this); }
    rng_iter end() const { return rng_iter(*this, start + len()*step); }

    int len() const { return (stop - start) / step + (((abs(stop-start)%abs(step)) == 0) ? 0 : 1); }
    bool operator==(const range& other) const { return other.start == start && other.stop == stop && other.step == step; }
    bool operator!=(const range& other) const { return !(this->operator==(other)); }
};

using slice = range;

template <typename T>
struct is_slice {
    static constexpr bool value = std::is_same<typename std::decay<T>::type, slice>::value ||
                                  std::is_same<typename std::decay<T>::type, all>::value;
};
template <typename T, typename... Ts>
struct all_integral_or_slice {
    static constexpr bool value = (std::is_integral<T>::value || is_slice<T>::value) && all_integral_or_slice<Ts...>::value;
};
template <typename T>
struct all_integral_or_slice<T> {
    static constexpr bool value = (std::is_integral<T>::value || is_slice<T>::value);
};

/*
 * Convert a list of parameters, which may be indices or slices, to a
 * list of slices.
 */
template<typename I0>
slice _index_to_slice(const I0& i0, typename std::enable_if<std::is_integral<I0>::value, void*>::type = 0) {
    return slice(i0, i0+1);
}
template<typename I0>
slice _index_to_slice(const I0& i0, typename std::enable_if<is_slice<I0>::value, void*>::type = 0) {
    return i0;
}
template <typename T, typename I0, typename... Indices>
void _indices_to_slices(T& slices, int n, const I0& i0, const Indices&... inds) {
    slices[n] = _index_to_slice(i0);
    _indices_to_slices(slices, n+1, inds...);
}
template <typename T, typename I0>
void _indices_to_slices(T& slices, int n, const I0& i0) {
    slices[n] = _index_to_slice(i0);
}

template <typename Arr, typename... Slices>
struct slice_expr {
    using value_type = typename Arr::value_type;
    using shape_type = sliced_shape_pack<typename Arr::shape_type, Slices...>;
    using array_ref = typename std::conditional<std::is_const<Arr>::value, const Arr&, Arr&>::type;
    static constexpr bool _is_expression = true;
    static constexpr bool _is_array = true;
    static constexpr size_t ndim = Arr::ndim;

    array_ref arr;
    std::array<slice, ndim> _slices;
    std::array<size_t, ndim> _shape;
    size_t _size;

    const size_t& size() const { return _size; }
    const std::array<size_t, ndim>& shape() const { return _shape; }

    template<enable_if_array(Arr)>
    slice_expr(array_ref a, const Slices&... slices) : arr(a) {
        const auto& a_shp = arr.shape();
        std::array<slice, Arr::ndim> s;
        _indices_to_slices(s, 0, slices...);

        _size = 1;
        for(int i=0; i < ndim; ++i) {
            _slices[i] = (is_all(s[i]) || s[i]==slice()) ? slice(a_shp[i]) : s[i];
            _shape[i] = _slices[i].len();
            _size *= _shape[i];
        }
    }

    template <typename... Is, typename = std::enable_if_t<all_integral<Is...>::value>>
    const value_type& operator()(Is... i) const {
        const std::array<int, ndim> slice_inds{i...};
        std::array<size_t, ndim> inds;
        for(int j=0; j < ndim; ++j) {
            assert(slice_inds[j] < _shape[j] && "Invalid index.");
            inds[j] = *(_slices[j].begin() + slice_inds[j]);
        }
        return arr(inds);
    }
    template <typename... Is, typename = std::enable_if_t<all_integral<Is...>::value>>
    value_type& operator()(Is... i) {
        const std::array<int, ndim> slice_inds{i...};
        std::array<size_t, ndim> inds;
        for(int j=0; j<ndim; ++j) {
            assert(slice_inds[j] < _shape[j] && "Invalid index.");
            inds[j] = *(_slices[j].begin() + slice_inds[j]);
        }
        return arr(inds);
    }
    const value_type& operator()(const std::array<size_t, ndim>& slice_inds) const {
        std::array<size_t, ndim> inds;
        for(int j=0; j<ndim; ++j) {
            assert(slice_inds[j] < _shape[j] && "Invalid index.");
            inds[j] = *(_slices[j].begin() + slice_inds[j]);
        }
        return arr(inds);
    }
    value_type& operator()(const std::array<size_t, ndim>& slice_inds) {
        std::array<size_t, ndim> inds;
        for(int j=0; j<ndim; ++j) {
            assert(slice_inds[j] < _shape[j] && "Invalid index.");
            inds[j] = *(_slices[j].begin() + slice_inds[j]);
        }
        return arr(inds);
    }

    slice_expr& operator=(const value_type& v) {
        for(auto it = begin(); it != end(); ++it) { (*it) = v; }
        return *this;
    }

    struct iterator {
        const slice_expr& expr;

        // The {} is required to value-initialize the array with zeroes.
        std::array<size_t, ndim> cur_pos{};

        iterator(const slice_expr& e) : expr(e) { }
        iterator(const slice_expr& e, const std::array<size_t, ndim>& p) : expr(e), cur_pos(p) {}

        const value_type& operator*() const { return expr(cur_pos); }

        bool operator==(const iterator& other) const {
            assert(other.expr.arr.data == expr.arr.data && "Cannot compare iterators from different array instances.");
            bool equal = true;
            for(int i=0; i<ndim; ++i) {
                equal = equal && (cur_pos[i] == other.cur_pos[i]);
            }
            return equal;
        }
        bool operator!=(const iterator& other) const { return !(this->operator==(other)); }

        iterator& operator++() {
            int carries = 0;
            for(int i=ndim-1; i>=0; --i) {
                if(cur_pos[i] < expr.shape()[i]-1) {
                    ++cur_pos[i];
                    break;
                } else {
                    cur_pos[i] = 0;
                    ++carries;
                }
            }

            if (carries == ndim) {
                cur_pos = expr.shape();
            }
            return *this;
        }

        iterator& operator--() {
            int zeros = 0;
            for(int i=ndim-1; i>=0; --i) {
                if(cur_pos[i] > 0) {
                    --cur_pos[i];
                    break;
                } else {
                    cur_pos[i] = expr.shape()[i]-1;
                    ++zeros;
                }
            }

            if (zeros == ndim) {
                cur_pos = expr.shape();
            }
            return *this;
        }

        iterator operator++(int) {
            iterator copy(*this);
            ++(*this);
            return copy;
        }
        iterator operator--(int) {
            iterator copy(*this);
            --(*this);
            return copy;
        }
    };

    struct mutable_iterator : public iterator {
        slice_expr& expr;

        mutable_iterator(slice_expr& e) : expr(e) { }
        mutable_iterator(slice_expr& e, const std::array<size_t, ndim>& p) : expr(e) { this->cur_pos = p; }

        value_type& operator*() { return expr(this->cur_pos); }
    };

    using const_iterator = iterator;

    const_iterator begin() const { return const_iterator(*this); }
    const_iterator end() const { return const_iterator(*this, _shape); }
    mutable_iterator begin() { return mutable_iterator(*this); }
    mutable_iterator end() { return mutable_iterator(*this, _shape); }
};

} // END namespace nda
