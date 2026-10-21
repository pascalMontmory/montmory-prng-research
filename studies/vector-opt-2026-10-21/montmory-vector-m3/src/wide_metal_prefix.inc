namespace montmory_wide_metal {
typedef uint4 V;
inline V splat(uint x) {return V(x);}
inline V mask_lt(V a,V b) {return select(V(0),V(0xffffffffu),a<b);}
inline V mask_ne(V a,V b) {return select(V(0),V(0xffffffffu),a!=b);}
inline V high_product(V a,V b) {return metal::mulhi(a,b);}
