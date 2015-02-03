#pragma once

namespace nda {
/*
 * Range object, used for sliced indexing and index generation in for loops.
 */
struct range {
    static constexpr bool _is_slice = true;
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
template <size_t S, typename I0, typename... Indices>
void _convert_indices(std::array<slice, S>& slices, std::array<bool, S>& ind_types, int n, const I0& i0, const Indices&... inds) {
    slices[n] = _index_to_slice(i0);
    ind_types[n] = is_slice<I0>::value;
    _convert_indices(slices, ind_types, n+1, inds...);
}
template <size_t S, typename I0>
void _convert_indices(std::array<slice, S>& slices, std::array<bool, S>& ind_types, int n, const I0& i0) {
    slices[n] = _index_to_slice(i0);
    ind_types[n] = is_slice<I0>::value;
}

template <typename Expr, typename... Slices>
struct slice_expr {
    using value_type = typename Expr::value_type;
    using shape_type = sliced_shape_pack<typename Expr::shape_type, Slices...>;
    using expr_ref = typename std::conditional<std::is_const<Expr>::value, const Expr&, Expr&>::type;
    static constexpr bool _is_expression = true;
    static constexpr bool _is_array = true;
    static constexpr size_t ndim = Expr::ndim - count_integral<Slices...>::value;
    static constexpr size_t fulldim = Expr::ndim;

    expr_ref expr;
    std::array<slice, fulldim> _slices = {};
    std::array<bool, fulldim> dim_preserved = {};
    std::array<size_t, ndim> sliced_dims = {};
    std::array<size_t, ndim> _shape = {};
    size_t _size;

    const size_t& size() const { return _size; }
    const std::array<size_t, ndim>& shape() const { return _shape; }

    template<enable_if_array_or_expr(Expr)>
    slice_expr(expr_ref e, const Slices&... slices) : expr(e) {
        const auto& a_shp = expr.shape();
        std::array<slice, fulldim> converted_slices;
        for(int i=0; i<fulldim; ++i) { dim_preserved[i] = true; }
        _convert_indices(converted_slices, dim_preserved, 0, slices...);

        _size = 1;
        int si = 0;
        for(int i=0; i<fulldim; ++i) {
            const auto& sl = converted_slices[i];
            if( dim_preserved[i] ) {
                _slices[i] = (sl == slice()) ? slice(a_shp[i]) : sl;
                _shape[si] = _slices[i].len();
                _size *= _shape[si];
                sliced_dims[si] = i;
                ++si;
            } else {
                _slices[i] = sl;
            }
        }
    }

    template <typename... Is, typename = std::enable_if_t<all_integral<Is...>::value>>
    const value_type& operator()(Is... is) const {
        return (*this)({static_cast<size_t>(is)...});
    }
    template <typename... Is, typename = std::enable_if_t<all_integral<Is...>::value>>
    value_type& operator()(Is... is) {
        static_assert(sizeof...(Is) == ndim, "Insufficient indices.");

        return (*this)({static_cast<size_t>(is)...});
    }
    const value_type& operator()(const std::array<size_t, ndim>& slice_inds) const {
        std::array<size_t, fulldim> inds;
        int si = 0;
        for(int i=0; i < fulldim; ++i) {
            if( dim_preserved[i] ) {
                assert(slice_inds[si] < _shape[i] && "Invalid index.");
                inds[i] = *(_slices[i].begin() + slice_inds[si]);
                ++si;
            } else {
                inds[i] = _slices[i].start;
            }
        }
        return expr(inds);
    }
    value_type& operator()(const std::array<size_t, ndim>& slice_inds) {
        std::array<size_t, fulldim> inds;
        int si = 0;
        for(int i=0; i < fulldim; ++i) {
            if( dim_preserved[i] ) {
                assert(slice_inds[si] < _shape[i] && "Invalid index.");
                inds[i] = *(_slices[i].begin() + slice_inds[si]);
                ++si;
            } else {
                inds[i] = _slices[i].start;
            }
        }
        return expr(inds);
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
