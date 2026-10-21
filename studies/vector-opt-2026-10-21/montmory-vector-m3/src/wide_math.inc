// Shared unsigned arithmetic for four independent words: C++ SIMD and Metal.
// Backend supplies V, splat, mask_lt, mask_ne and high_product.
struct Wide64 { V lo, hi; };
inline V choose(V mask, V yes, V no) { return (mask & yes) | (~mask & no); }
inline Wide64 wide_add(Wide64 a, Wide64 b) {
    V lo=a.lo+b.lo;
    return {lo,a.hi+b.hi+(mask_lt(lo,a.lo)&splat(1))};
}
inline V wide_f(V x, Wide64 key, unsigned ro, unsigned re) {
    V odd=mask_ne(x&splat(1),splat(0));
    V factor=splat(1)+((x&splat(1))<<1);
    V low=x*factor, high=high_product(x,factor);
    V sum=low+key.lo;
    high+=key.hi+(mask_lt(sum,low)&splat(1));
    V r=choose(odd,splat(ro),splat(re));
    V swap=mask_ne(r&splat(32),splat(0));
    V a=choose(swap,high,sum),b=choose(swap,sum,high);
    V shift=r&splat(31); // Every seed rotation is odd: shift in 1..31.
    return ((a<<shift)|(b>>(splat(32)-shift))) ^ choose(odd,x>>1,x*splat(5));
}
inline Wide64 wide_round(Wide64 x, Wide64 key, unsigned ro, unsigned re) {
    return {x.hi^wide_f(x.lo,key,ro,re),x.lo};
}
inline Wide64 wide_xor_right(Wide64 x,unsigned shift) {
    return {x.lo^((x.lo>>shift)|(x.hi<<(32-shift))),x.hi^(x.hi>>shift)};
}
inline Wide64 wide_multiply(Wide64 x,unsigned lo,unsigned hi) {
    return {x.lo*splat(lo),high_product(x.lo,splat(lo))+x.hi*splat(lo)+x.lo*splat(hi)};
}
inline Wide64 wide_mix(Wide64 x) {
    V r=x.hi>>27;
    Wide64 w={x.lo^x.hi,x.hi};
    w=wide_xor_right(w,25);w=wide_multiply(w,0x96EE9B5Fu,0xA24BAED4u);
    w=wide_xor_right(w,28);w=wide_multiply(w,0x1E98DF25u,0x9FB21C65u);
    w=wide_xor_right(w,27);
    V complement=(splat(32)-r)&splat(31),nonzero=mask_ne(r,splat(0));
    return {(w.lo>>r)|((w.hi<<complement)&nonzero),
            (w.hi>>r)|((w.lo<<complement)&nonzero)};
}
