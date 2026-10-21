// Generated from the shipped sources; Random123 license: vendor/LICENSE.Random123.
#include <metal_stdlib>
using namespace metal;
#include <metal_integer>
inline int haveAESNI(){
    return 0;
}
struct r123array1x32{ uint32_t v[1]; };
struct r123array2x32{ uint32_t v[2]; };
struct r123array4x32{ uint32_t v[4]; };
struct r123array8x32{ uint32_t v[8]; };
struct r123array16x8{ uint8_t v[16]; };
enum r123_enum_threefry64x4 {
    R_64x4_0_0=14, R_64x4_0_1=16,
    R_64x4_1_0=52, R_64x4_1_1=57,
    R_64x4_2_0=23, R_64x4_2_1=40,
    R_64x4_3_0= 5, R_64x4_3_1=37,
    R_64x4_4_0=25, R_64x4_4_1=33,
    R_64x4_5_0=46, R_64x4_5_1=12,
    R_64x4_6_0=58, R_64x4_6_1=22,
    R_64x4_7_0=32, R_64x4_7_1=32
};
enum r123_enum_threefry64x2 {
    R_64x2_0_0=16,
    R_64x2_1_0=42,
    R_64x2_2_0=12,
    R_64x2_3_0=31,
    R_64x2_4_0=16,
    R_64x2_5_0=32,
    R_64x2_6_0=24,
    R_64x2_7_0=21
};
enum r123_enum_threefry32x4 {
    R_32x4_0_0=10, R_32x4_0_1=26,
    R_32x4_1_0=11, R_32x4_1_1=21,
    R_32x4_2_0=13, R_32x4_2_1=27,
    R_32x4_3_0=23, R_32x4_3_1= 5,
    R_32x4_4_0= 6, R_32x4_4_1=20,
    R_32x4_5_0=17, R_32x4_5_1=11,
    R_32x4_6_0=25, R_32x4_6_1=10,
    R_32x4_7_0=18, R_32x4_7_1=20
};
enum r123_enum_threefry32x2 {
    R_32x2_0_0=13,
    R_32x2_1_0=15,
    R_32x2_2_0=26,
    R_32x2_3_0= 6,
    R_32x2_4_0=17,
    R_32x2_5_0=29,
    R_32x2_6_0=16,
    R_32x2_7_0=24
    };
enum r123_enum_threefry_wcnt {
    WCNT2=2,
    WCNT4=4
};
 inline uint32_t RotL_32(uint32_t x, unsigned int N) __attribute__((always_inline));
 inline uint32_t RotL_32(uint32_t x, unsigned int N)
{
    return (x << (N & 31)) | (x >> ((32-N) & 31));
}
typedef struct r123array2x32 threefry2x32_ctr_t; typedef struct r123array2x32 threefry2x32_key_t; typedef struct r123array2x32 threefry2x32_ukey_t; inline threefry2x32_key_t threefry2x32keyinit(threefry2x32_ukey_t uk) { return uk; } inline threefry2x32_ctr_t threefry2x32_R(unsigned int Nrounds, threefry2x32_ctr_t in, threefry2x32_key_t k) __attribute__((always_inline)); inline threefry2x32_ctr_t threefry2x32_R(unsigned int Nrounds, threefry2x32_ctr_t in, threefry2x32_key_t k){ uint32_t X0,X1; uint32_t ks0, ks1, ks2; ; ks2 = 0x1BD11BDA; ks0 = k.v[0]; X0 = in.v[0] + ks0; ks2 ^= ks0; ks1 = k.v[1]; X1 = in.v[1] + ks1; ks2 ^= ks1; if(Nrounds>0){ X0 += X1; X1 = RotL_32(X1,R_32x2_0_0); X1 ^= X0; } if(Nrounds>1){ X0 += X1; X1 = RotL_32(X1,R_32x2_1_0); X1 ^= X0; } if(Nrounds>2){ X0 += X1; X1 = RotL_32(X1,R_32x2_2_0); X1 ^= X0; } if(Nrounds>3){ X0 += X1; X1 = RotL_32(X1,R_32x2_3_0); X1 ^= X0; } if(Nrounds>3){ X0 += ks1; X1 += ks2; X1 += 1; } if(Nrounds>4){ X0 += X1; X1 = RotL_32(X1,R_32x2_4_0); X1 ^= X0; } if(Nrounds>5){ X0 += X1; X1 = RotL_32(X1,R_32x2_5_0); X1 ^= X0; } if(Nrounds>6){ X0 += X1; X1 = RotL_32(X1,R_32x2_6_0); X1 ^= X0; } if(Nrounds>7){ X0 += X1; X1 = RotL_32(X1,R_32x2_7_0); X1 ^= X0; } if(Nrounds>7){ X0 += ks2; X1 += ks0; X1 += 2; } if(Nrounds>8){ X0 += X1; X1 = RotL_32(X1,R_32x2_0_0); X1 ^= X0; } if(Nrounds>9){ X0 += X1; X1 = RotL_32(X1,R_32x2_1_0); X1 ^= X0; } if(Nrounds>10){ X0 += X1; X1 = RotL_32(X1,R_32x2_2_0); X1 ^= X0; } if(Nrounds>11){ X0 += X1; X1 = RotL_32(X1,R_32x2_3_0); X1 ^= X0; } if(Nrounds>11){ X0 += ks0; X1 += ks1; X1 += 3; } if(Nrounds>12){ X0 += X1; X1 = RotL_32(X1,R_32x2_4_0); X1 ^= X0; } if(Nrounds>13){ X0 += X1; X1 = RotL_32(X1,R_32x2_5_0); X1 ^= X0; } if(Nrounds>14){ X0 += X1; X1 = RotL_32(X1,R_32x2_6_0); X1 ^= X0; } if(Nrounds>15){ X0 += X1; X1 = RotL_32(X1,R_32x2_7_0); X1 ^= X0; } if(Nrounds>15){ X0 += ks1; X1 += ks2; X1 += 4; } if(Nrounds>16){ X0 += X1; X1 = RotL_32(X1,R_32x2_0_0); X1 ^= X0; } if(Nrounds>17){ X0 += X1; X1 = RotL_32(X1,R_32x2_1_0); X1 ^= X0; } if(Nrounds>18){ X0 += X1; X1 = RotL_32(X1,R_32x2_2_0); X1 ^= X0; } if(Nrounds>19){ X0 += X1; X1 = RotL_32(X1,R_32x2_3_0); X1 ^= X0; } if(Nrounds>19){ X0 += ks2; X1 += ks0; X1 += 5; } if(Nrounds>20){ X0 += X1; X1 = RotL_32(X1,R_32x2_4_0); X1 ^= X0; } if(Nrounds>21){ X0 += X1; X1 = RotL_32(X1,R_32x2_5_0); X1 ^= X0; } if(Nrounds>22){ X0 += X1; X1 = RotL_32(X1,R_32x2_6_0); X1 ^= X0; } if(Nrounds>23){ X0 += X1; X1 = RotL_32(X1,R_32x2_7_0); X1 ^= X0; } if(Nrounds>23){ X0 += ks0; X1 += ks1; X1 += 6; } if(Nrounds>24){ X0 += X1; X1 = RotL_32(X1,R_32x2_0_0); X1 ^= X0; } if(Nrounds>25){ X0 += X1; X1 = RotL_32(X1,R_32x2_1_0); X1 ^= X0; } if(Nrounds>26){ X0 += X1; X1 = RotL_32(X1,R_32x2_2_0); X1 ^= X0; } if(Nrounds>27){ X0 += X1; X1 = RotL_32(X1,R_32x2_3_0); X1 ^= X0; } if(Nrounds>27){ X0 += ks1; X1 += ks2; X1 += 7; } if(Nrounds>28){ X0 += X1; X1 = RotL_32(X1,R_32x2_4_0); X1 ^= X0; } if(Nrounds>29){ X0 += X1; X1 = RotL_32(X1,R_32x2_5_0); X1 ^= X0; } if(Nrounds>30){ X0 += X1; X1 = RotL_32(X1,R_32x2_6_0); X1 ^= X0; } if(Nrounds>31){ X0 += X1; X1 = RotL_32(X1,R_32x2_7_0); X1 ^= X0; } if(Nrounds>31){ X0 += ks2; X1 += ks0; X1 += 8; } threefry2x32_ctr_t ret={{X0, X1}}; return ret; } enum r123_enum_threefry2x32 { threefry2x32_rounds = 20 }; inline threefry2x32_ctr_t threefry2x32(threefry2x32_ctr_t in, threefry2x32_key_t k) __attribute__((always_inline)); inline threefry2x32_ctr_t threefry2x32(threefry2x32_ctr_t in, threefry2x32_key_t k){ return threefry2x32_R(threefry2x32_rounds, in, k); }
typedef struct r123array4x32 threefry4x32_ctr_t; typedef struct r123array4x32 threefry4x32_key_t; typedef struct r123array4x32 threefry4x32_ukey_t; inline threefry4x32_key_t threefry4x32keyinit(threefry4x32_ukey_t uk) { return uk; } inline threefry4x32_ctr_t threefry4x32_R(unsigned int Nrounds, threefry4x32_ctr_t in, threefry4x32_key_t k) __attribute__((always_inline)); inline threefry4x32_ctr_t threefry4x32_R(unsigned int Nrounds, threefry4x32_ctr_t in, threefry4x32_key_t k){ uint32_t X0, X1, X2, X3; uint32_t ks0, ks1, ks2, ks3, ks4; ; ks4 = 0x1BD11BDA; ks0 = k.v[0]; X0 = in.v[0] + ks0; ks4 ^= ks0; ks1 = k.v[1]; X1 = in.v[1] + ks1; ks4 ^= ks1; ks2 = k.v[2]; X2 = in.v[2] + ks2; ks4 ^= ks2; ks3 = k.v[3]; X3 = in.v[3] + ks3; ks4 ^= ks3; if(Nrounds>0){ X0 += X1; X1 = RotL_32(X1,R_32x4_0_0); X1 ^= X0; X2 += X3; X3 = RotL_32(X3,R_32x4_0_1); X3 ^= X2; } if(Nrounds>1){ X0 += X3; X3 = RotL_32(X3,R_32x4_1_0); X3 ^= X0; X2 += X1; X1 = RotL_32(X1,R_32x4_1_1); X1 ^= X2; } if(Nrounds>2){ X0 += X1; X1 = RotL_32(X1,R_32x4_2_0); X1 ^= X0; X2 += X3; X3 = RotL_32(X3,R_32x4_2_1); X3 ^= X2; } if(Nrounds>3){ X0 += X3; X3 = RotL_32(X3,R_32x4_3_0); X3 ^= X0; X2 += X1; X1 = RotL_32(X1,R_32x4_3_1); X1 ^= X2; } if(Nrounds>3){ X0 += ks1; X1 += ks2; X2 += ks3; X3 += ks4; X3 += 1; } if(Nrounds>4){ X0 += X1; X1 = RotL_32(X1,R_32x4_4_0); X1 ^= X0; X2 += X3; X3 = RotL_32(X3,R_32x4_4_1); X3 ^= X2; } if(Nrounds>5){ X0 += X3; X3 = RotL_32(X3,R_32x4_5_0); X3 ^= X0; X2 += X1; X1 = RotL_32(X1,R_32x4_5_1); X1 ^= X2; } if(Nrounds>6){ X0 += X1; X1 = RotL_32(X1,R_32x4_6_0); X1 ^= X0; X2 += X3; X3 = RotL_32(X3,R_32x4_6_1); X3 ^= X2; } if(Nrounds>7){ X0 += X3; X3 = RotL_32(X3,R_32x4_7_0); X3 ^= X0; X2 += X1; X1 = RotL_32(X1,R_32x4_7_1); X1 ^= X2; } if(Nrounds>7){ X0 += ks2; X1 += ks3; X2 += ks4; X3 += ks0; X3 += 2; } if(Nrounds>8){ X0 += X1; X1 = RotL_32(X1,R_32x4_0_0); X1 ^= X0; X2 += X3; X3 = RotL_32(X3,R_32x4_0_1); X3 ^= X2; } if(Nrounds>9){ X0 += X3; X3 = RotL_32(X3,R_32x4_1_0); X3 ^= X0; X2 += X1; X1 = RotL_32(X1,R_32x4_1_1); X1 ^= X2; } if(Nrounds>10){ X0 += X1; X1 = RotL_32(X1,R_32x4_2_0); X1 ^= X0; X2 += X3; X3 = RotL_32(X3,R_32x4_2_1); X3 ^= X2; } if(Nrounds>11){ X0 += X3; X3 = RotL_32(X3,R_32x4_3_0); X3 ^= X0; X2 += X1; X1 = RotL_32(X1,R_32x4_3_1); X1 ^= X2; } if(Nrounds>11){ X0 += ks3; X1 += ks4; X2 += ks0; X3 += ks1; X3 += 3; } if(Nrounds>12){ X0 += X1; X1 = RotL_32(X1,R_32x4_4_0); X1 ^= X0; X2 += X3; X3 = RotL_32(X3,R_32x4_4_1); X3 ^= X2; } if(Nrounds>13){ X0 += X3; X3 = RotL_32(X3,R_32x4_5_0); X3 ^= X0; X2 += X1; X1 = RotL_32(X1,R_32x4_5_1); X1 ^= X2; } if(Nrounds>14){ X0 += X1; X1 = RotL_32(X1,R_32x4_6_0); X1 ^= X0; X2 += X3; X3 = RotL_32(X3,R_32x4_6_1); X3 ^= X2; } if(Nrounds>15){ X0 += X3; X3 = RotL_32(X3,R_32x4_7_0); X3 ^= X0; X2 += X1; X1 = RotL_32(X1,R_32x4_7_1); X1 ^= X2; } if(Nrounds>15){ X0 += ks4; X1 += ks0; X2 += ks1; X3 += ks2; X3 += 4; } if(Nrounds>16){ X0 += X1; X1 = RotL_32(X1,R_32x4_0_0); X1 ^= X0; X2 += X3; X3 = RotL_32(X3,R_32x4_0_1); X3 ^= X2; } if(Nrounds>17){ X0 += X3; X3 = RotL_32(X3,R_32x4_1_0); X3 ^= X0; X2 += X1; X1 = RotL_32(X1,R_32x4_1_1); X1 ^= X2; } if(Nrounds>18){ X0 += X1; X1 = RotL_32(X1,R_32x4_2_0); X1 ^= X0; X2 += X3; X3 = RotL_32(X3,R_32x4_2_1); X3 ^= X2; } if(Nrounds>19){ X0 += X3; X3 = RotL_32(X3,R_32x4_3_0); X3 ^= X0; X2 += X1; X1 = RotL_32(X1,R_32x4_3_1); X1 ^= X2; } if(Nrounds>19){ X0 += ks0; X1 += ks1; X2 += ks2; X3 += ks3; X3 += 5; } if(Nrounds>20){ X0 += X1; X1 = RotL_32(X1,R_32x4_4_0); X1 ^= X0; X2 += X3; X3 = RotL_32(X3,R_32x4_4_1); X3 ^= X2; } if(Nrounds>21){ X0 += X3; X3 = RotL_32(X3,R_32x4_5_0); X3 ^= X0; X2 += X1; X1 = RotL_32(X1,R_32x4_5_1); X1 ^= X2; } if(Nrounds>22){ X0 += X1; X1 = RotL_32(X1,R_32x4_6_0); X1 ^= X0; X2 += X3; X3 = RotL_32(X3,R_32x4_6_1); X3 ^= X2; } if(Nrounds>23){ X0 += X3; X3 = RotL_32(X3,R_32x4_7_0); X3 ^= X0; X2 += X1; X1 = RotL_32(X1,R_32x4_7_1); X1 ^= X2; } if(Nrounds>23){ X0 += ks1; X1 += ks2; X2 += ks3; X3 += ks4; X3 += 6; } if(Nrounds>24){ X0 += X1; X1 = RotL_32(X1,R_32x4_0_0); X1 ^= X0; X2 += X3; X3 = RotL_32(X3,R_32x4_0_1); X3 ^= X2; } if(Nrounds>25){ X0 += X3; X3 = RotL_32(X3,R_32x4_1_0); X3 ^= X0; X2 += X1; X1 = RotL_32(X1,R_32x4_1_1); X1 ^= X2; } if(Nrounds>26){ X0 += X1; X1 = RotL_32(X1,R_32x4_2_0); X1 ^= X0; X2 += X3; X3 = RotL_32(X3,R_32x4_2_1); X3 ^= X2; } if(Nrounds>27){ X0 += X3; X3 = RotL_32(X3,R_32x4_3_0); X3 ^= X0; X2 += X1; X1 = RotL_32(X1,R_32x4_3_1); X1 ^= X2; } if(Nrounds>27){ X0 += ks2; X1 += ks3; X2 += ks4; X3 += ks0; X3 += 7; } if(Nrounds>28){ X0 += X1; X1 = RotL_32(X1,R_32x4_4_0); X1 ^= X0; X2 += X3; X3 = RotL_32(X3,R_32x4_4_1); X3 ^= X2; } if(Nrounds>29){ X0 += X3; X3 = RotL_32(X3,R_32x4_5_0); X3 ^= X0; X2 += X1; X1 = RotL_32(X1,R_32x4_5_1); X1 ^= X2; } if(Nrounds>30){ X0 += X1; X1 = RotL_32(X1,R_32x4_6_0); X1 ^= X0; X2 += X3; X3 = RotL_32(X3,R_32x4_6_1); X3 ^= X2; } if(Nrounds>31){ X0 += X3; X3 = RotL_32(X3,R_32x4_7_0); X3 ^= X0; X2 += X1; X1 = RotL_32(X1,R_32x4_7_1); X1 ^= X2; } if(Nrounds>31){ X0 += ks3; X1 += ks4; X2 += ks0; X3 += ks1; X3 += 8; } if(Nrounds>32){ X0 += X1; X1 = RotL_32(X1,R_32x4_0_0); X1 ^= X0; X2 += X3; X3 = RotL_32(X3,R_32x4_0_1); X3 ^= X2; } if(Nrounds>33){ X0 += X3; X3 = RotL_32(X3,R_32x4_1_0); X3 ^= X0; X2 += X1; X1 = RotL_32(X1,R_32x4_1_1); X1 ^= X2; } if(Nrounds>34){ X0 += X1; X1 = RotL_32(X1,R_32x4_2_0); X1 ^= X0; X2 += X3; X3 = RotL_32(X3,R_32x4_2_1); X3 ^= X2; } if(Nrounds>35){ X0 += X3; X3 = RotL_32(X3,R_32x4_3_0); X3 ^= X0; X2 += X1; X1 = RotL_32(X1,R_32x4_3_1); X1 ^= X2; } if(Nrounds>35){ X0 += ks4; X1 += ks0; X2 += ks1; X3 += ks2; X3 += 9; } if(Nrounds>36){ X0 += X1; X1 = RotL_32(X1,R_32x4_4_0); X1 ^= X0; X2 += X3; X3 = RotL_32(X3,R_32x4_4_1); X3 ^= X2; } if(Nrounds>37){ X0 += X3; X3 = RotL_32(X3,R_32x4_5_0); X3 ^= X0; X2 += X1; X1 = RotL_32(X1,R_32x4_5_1); X1 ^= X2; } if(Nrounds>38){ X0 += X1; X1 = RotL_32(X1,R_32x4_6_0); X1 ^= X0; X2 += X3; X3 = RotL_32(X3,R_32x4_6_1); X3 ^= X2; } if(Nrounds>39){ X0 += X3; X3 = RotL_32(X3,R_32x4_7_0); X3 ^= X0; X2 += X1; X1 = RotL_32(X1,R_32x4_7_1); X1 ^= X2; } if(Nrounds>39){ X0 += ks0; X1 += ks1; X2 += ks2; X3 += ks3; X3 += 10; } if(Nrounds>40){ X0 += X1; X1 = RotL_32(X1,R_32x4_0_0); X1 ^= X0; X2 += X3; X3 = RotL_32(X3,R_32x4_0_1); X3 ^= X2; } if(Nrounds>41){ X0 += X3; X3 = RotL_32(X3,R_32x4_1_0); X3 ^= X0; X2 += X1; X1 = RotL_32(X1,R_32x4_1_1); X1 ^= X2; } if(Nrounds>42){ X0 += X1; X1 = RotL_32(X1,R_32x4_2_0); X1 ^= X0; X2 += X3; X3 = RotL_32(X3,R_32x4_2_1); X3 ^= X2; } if(Nrounds>43){ X0 += X3; X3 = RotL_32(X3,R_32x4_3_0); X3 ^= X0; X2 += X1; X1 = RotL_32(X1,R_32x4_3_1); X1 ^= X2; } if(Nrounds>43){ X0 += ks1; X1 += ks2; X2 += ks3; X3 += ks4; X3 += 11; } if(Nrounds>44){ X0 += X1; X1 = RotL_32(X1,R_32x4_4_0); X1 ^= X0; X2 += X3; X3 = RotL_32(X3,R_32x4_4_1); X3 ^= X2; } if(Nrounds>45){ X0 += X3; X3 = RotL_32(X3,R_32x4_5_0); X3 ^= X0; X2 += X1; X1 = RotL_32(X1,R_32x4_5_1); X1 ^= X2; } if(Nrounds>46){ X0 += X1; X1 = RotL_32(X1,R_32x4_6_0); X1 ^= X0; X2 += X3; X3 = RotL_32(X3,R_32x4_6_1); X3 ^= X2; } if(Nrounds>47){ X0 += X3; X3 = RotL_32(X3,R_32x4_7_0); X3 ^= X0; X2 += X1; X1 = RotL_32(X1,R_32x4_7_1); X1 ^= X2; } if(Nrounds>47){ X0 += ks2; X1 += ks3; X2 += ks4; X3 += ks0; X3 += 12; } if(Nrounds>48){ X0 += X1; X1 = RotL_32(X1,R_32x4_0_0); X1 ^= X0; X2 += X3; X3 = RotL_32(X3,R_32x4_0_1); X3 ^= X2; } if(Nrounds>49){ X0 += X3; X3 = RotL_32(X3,R_32x4_1_0); X3 ^= X0; X2 += X1; X1 = RotL_32(X1,R_32x4_1_1); X1 ^= X2; } if(Nrounds>50){ X0 += X1; X1 = RotL_32(X1,R_32x4_2_0); X1 ^= X0; X2 += X3; X3 = RotL_32(X3,R_32x4_2_1); X3 ^= X2; } if(Nrounds>51){ X0 += X3; X3 = RotL_32(X3,R_32x4_3_0); X3 ^= X0; X2 += X1; X1 = RotL_32(X1,R_32x4_3_1); X1 ^= X2; } if(Nrounds>51){ X0 += ks3; X1 += ks4; X2 += ks0; X3 += ks1; X3 += 13; } if(Nrounds>52){ X0 += X1; X1 = RotL_32(X1,R_32x4_4_0); X1 ^= X0; X2 += X3; X3 = RotL_32(X3,R_32x4_4_1); X3 ^= X2; } if(Nrounds>53){ X0 += X3; X3 = RotL_32(X3,R_32x4_5_0); X3 ^= X0; X2 += X1; X1 = RotL_32(X1,R_32x4_5_1); X1 ^= X2; } if(Nrounds>54){ X0 += X1; X1 = RotL_32(X1,R_32x4_6_0); X1 ^= X0; X2 += X3; X3 = RotL_32(X3,R_32x4_6_1); X3 ^= X2; } if(Nrounds>55){ X0 += X3; X3 = RotL_32(X3,R_32x4_7_0); X3 ^= X0; X2 += X1; X1 = RotL_32(X1,R_32x4_7_1); X1 ^= X2; } if(Nrounds>55){ X0 += ks4; X1 += ks0; X2 += ks1; X3 += ks2; X3 += 14; } if(Nrounds>56){ X0 += X1; X1 = RotL_32(X1,R_32x4_0_0); X1 ^= X0; X2 += X3; X3 = RotL_32(X3,R_32x4_0_1); X3 ^= X2; } if(Nrounds>57){ X0 += X3; X3 = RotL_32(X3,R_32x4_1_0); X3 ^= X0; X2 += X1; X1 = RotL_32(X1,R_32x4_1_1); X1 ^= X2; } if(Nrounds>58){ X0 += X1; X1 = RotL_32(X1,R_32x4_2_0); X1 ^= X0; X2 += X3; X3 = RotL_32(X3,R_32x4_2_1); X3 ^= X2; } if(Nrounds>59){ X0 += X3; X3 = RotL_32(X3,R_32x4_3_0); X3 ^= X0; X2 += X1; X1 = RotL_32(X1,R_32x4_3_1); X1 ^= X2; } if(Nrounds>59){ X0 += ks0; X1 += ks1; X2 += ks2; X3 += ks3; X3 += 15; } if(Nrounds>60){ X0 += X1; X1 = RotL_32(X1,R_32x4_4_0); X1 ^= X0; X2 += X3; X3 = RotL_32(X3,R_32x4_4_1); X3 ^= X2; } if(Nrounds>61){ X0 += X3; X3 = RotL_32(X3,R_32x4_5_0); X3 ^= X0; X2 += X1; X1 = RotL_32(X1,R_32x4_5_1); X1 ^= X2; } if(Nrounds>62){ X0 += X1; X1 = RotL_32(X1,R_32x4_6_0); X1 ^= X0; X2 += X3; X3 = RotL_32(X3,R_32x4_6_1); X3 ^= X2; } if(Nrounds>63){ X0 += X3; X3 = RotL_32(X3,R_32x4_7_0); X3 ^= X0; X2 += X1; X1 = RotL_32(X1,R_32x4_7_1); X1 ^= X2; } if(Nrounds>63){ X0 += ks1; X1 += ks2; X2 += ks3; X3 += ks4; X3 += 16; } if(Nrounds>64){ X0 += X1; X1 = RotL_32(X1,R_32x4_0_0); X1 ^= X0; X2 += X3; X3 = RotL_32(X3,R_32x4_0_1); X3 ^= X2; } if(Nrounds>65){ X0 += X3; X3 = RotL_32(X3,R_32x4_1_0); X3 ^= X0; X2 += X1; X1 = RotL_32(X1,R_32x4_1_1); X1 ^= X2; } if(Nrounds>66){ X0 += X1; X1 = RotL_32(X1,R_32x4_2_0); X1 ^= X0; X2 += X3; X3 = RotL_32(X3,R_32x4_2_1); X3 ^= X2; } if(Nrounds>67){ X0 += X3; X3 = RotL_32(X3,R_32x4_3_0); X3 ^= X0; X2 += X1; X1 = RotL_32(X1,R_32x4_3_1); X1 ^= X2; } if(Nrounds>67){ X0 += ks2; X1 += ks3; X2 += ks4; X3 += ks0; X3 += 17; } if(Nrounds>68){ X0 += X1; X1 = RotL_32(X1,R_32x4_4_0); X1 ^= X0; X2 += X3; X3 = RotL_32(X3,R_32x4_4_1); X3 ^= X2; } if(Nrounds>69){ X0 += X3; X3 = RotL_32(X3,R_32x4_5_0); X3 ^= X0; X2 += X1; X1 = RotL_32(X1,R_32x4_5_1); X1 ^= X2; } if(Nrounds>70){ X0 += X1; X1 = RotL_32(X1,R_32x4_6_0); X1 ^= X0; X2 += X3; X3 = RotL_32(X3,R_32x4_6_1); X3 ^= X2; } if(Nrounds>71){ X0 += X3; X3 = RotL_32(X3,R_32x4_7_0); X3 ^= X0; X2 += X1; X1 = RotL_32(X1,R_32x4_7_1); X1 ^= X2; } if(Nrounds>71){ X0 += ks3; X1 += ks4; X2 += ks0; X3 += ks1; X3 += 18; } threefry4x32_ctr_t ret = {{X0, X1, X2, X3}}; return ret; } enum r123_enum_threefry4x32 { threefry4x32_rounds = 20 }; inline threefry4x32_ctr_t threefry4x32(threefry4x32_ctr_t in, threefry4x32_key_t k) __attribute__((always_inline)); inline threefry4x32_ctr_t threefry4x32(threefry4x32_ctr_t in, threefry4x32_key_t k){ return threefry4x32_R(threefry4x32_rounds, in, k); }
namespace r123{ template<unsigned int ROUNDS> struct Threefry2x32_R{ typedef threefry2x32_ctr_t ctr_type; typedef threefry2x32_key_t key_type; typedef threefry2x32_key_t ukey_type; static const constant unsigned int rounds=ROUNDS; inline ctr_type operator()(ctr_type ctr, key_type key) __attribute__((always_inline)){ static_assert(ROUNDS<=72, "threefry is only unrolled up to 72 rounds\n"); return threefry2x32_R(ROUNDS, ctr, key); } }; typedef Threefry2x32_R<threefry2x32_rounds> Threefry2x32; }
namespace r123{ template<unsigned int ROUNDS> struct Threefry4x32_R{ typedef threefry4x32_ctr_t ctr_type; typedef threefry4x32_key_t key_type; typedef threefry4x32_key_t ukey_type; static const constant unsigned int rounds=ROUNDS; inline ctr_type operator()(ctr_type ctr, key_type key) __attribute__((always_inline)){ static_assert(ROUNDS<=72, "threefry is only unrolled up to 72 rounds\n"); return threefry4x32_R(ROUNDS, ctr, key); } }; typedef Threefry4x32_R<threefry4x32_rounds> Threefry4x32; }
 inline uint32_t mulhilo32(uint32_t a, uint32_t b, thread uint32_t* hip){ *hip = metal::mulhi(a, b); return a*b; }
 inline struct r123array1x32 _philox2x32bumpkey( struct r123array1x32 key) { key.v[0] += ((uint32_t)0x9E3779B9); return key; }
 inline struct r123array2x32 _philox4x32bumpkey( struct r123array2x32 key) { key.v[0] += ((uint32_t)0x9E3779B9); key.v[1] += ((uint32_t)0xBB67AE85); return key; }
 inline struct r123array2x32 _philox2x32round(struct r123array2x32 ctr, struct r123array1x32 key) __attribute__((always_inline)); inline struct r123array2x32 _philox2x32round(struct r123array2x32 ctr, struct r123array1x32 key){ uint32_t hi; uint32_t lo = mulhilo32(((uint32_t)0xd256d193), ctr.v[0], &hi); struct r123array2x32 out = {{hi^key.v[0]^ctr.v[1], lo}}; return out; }
 inline struct r123array4x32 _philox4x32round(struct r123array4x32 ctr, struct r123array2x32 key) __attribute__((always_inline)); inline struct r123array4x32 _philox4x32round(struct r123array4x32 ctr, struct r123array2x32 key){ uint32_t hi0; uint32_t hi1; uint32_t lo0 = mulhilo32(((uint32_t)0xD2511F53), ctr.v[0], &hi0); uint32_t lo1 = mulhilo32(((uint32_t)0xCD9E8D57), ctr.v[2], &hi1); struct r123array4x32 out = {{hi1^ctr.v[1]^key.v[0], lo1, hi0^ctr.v[3]^key.v[1], lo0}}; return out; }
enum r123_enum_philox2x32 { philox2x32_rounds = 10 }; typedef struct r123array2x32 philox2x32_ctr_t; typedef struct r123array1x32 philox2x32_key_t; typedef struct r123array1x32 philox2x32_ukey_t; inline philox2x32_key_t philox2x32keyinit(philox2x32_ukey_t uk) { return uk; } inline philox2x32_ctr_t philox2x32_R(unsigned int R, philox2x32_ctr_t ctr, philox2x32_key_t key) __attribute__((always_inline)); inline philox2x32_ctr_t philox2x32_R(unsigned int R, philox2x32_ctr_t ctr, philox2x32_key_t key) { ; if(R>0){ ctr = _philox2x32round(ctr, key); } if(R>1){ key = _philox2x32bumpkey(key); ctr = _philox2x32round(ctr, key); } if(R>2){ key = _philox2x32bumpkey(key); ctr = _philox2x32round(ctr, key); } if(R>3){ key = _philox2x32bumpkey(key); ctr = _philox2x32round(ctr, key); } if(R>4){ key = _philox2x32bumpkey(key); ctr = _philox2x32round(ctr, key); } if(R>5){ key = _philox2x32bumpkey(key); ctr = _philox2x32round(ctr, key); } if(R>6){ key = _philox2x32bumpkey(key); ctr = _philox2x32round(ctr, key); } if(R>7){ key = _philox2x32bumpkey(key); ctr = _philox2x32round(ctr, key); } if(R>8){ key = _philox2x32bumpkey(key); ctr = _philox2x32round(ctr, key); } if(R>9){ key = _philox2x32bumpkey(key); ctr = _philox2x32round(ctr, key); } if(R>10){ key = _philox2x32bumpkey(key); ctr = _philox2x32round(ctr, key); } if(R>11){ key = _philox2x32bumpkey(key); ctr = _philox2x32round(ctr, key); } if(R>12){ key = _philox2x32bumpkey(key); ctr = _philox2x32round(ctr, key); } if(R>13){ key = _philox2x32bumpkey(key); ctr = _philox2x32round(ctr, key); } if(R>14){ key = _philox2x32bumpkey(key); ctr = _philox2x32round(ctr, key); } if(R>15){ key = _philox2x32bumpkey(key); ctr = _philox2x32round(ctr, key); } return ctr; }
enum r123_enum_philox4x32 { philox4x32_rounds = 10 }; typedef struct r123array4x32 philox4x32_ctr_t; typedef struct r123array2x32 philox4x32_key_t; typedef struct r123array2x32 philox4x32_ukey_t; inline philox4x32_key_t philox4x32keyinit(philox4x32_ukey_t uk) { return uk; } inline philox4x32_ctr_t philox4x32_R(unsigned int R, philox4x32_ctr_t ctr, philox4x32_key_t key) __attribute__((always_inline)); inline philox4x32_ctr_t philox4x32_R(unsigned int R, philox4x32_ctr_t ctr, philox4x32_key_t key) { ; if(R>0){ ctr = _philox4x32round(ctr, key); } if(R>1){ key = _philox4x32bumpkey(key); ctr = _philox4x32round(ctr, key); } if(R>2){ key = _philox4x32bumpkey(key); ctr = _philox4x32round(ctr, key); } if(R>3){ key = _philox4x32bumpkey(key); ctr = _philox4x32round(ctr, key); } if(R>4){ key = _philox4x32bumpkey(key); ctr = _philox4x32round(ctr, key); } if(R>5){ key = _philox4x32bumpkey(key); ctr = _philox4x32round(ctr, key); } if(R>6){ key = _philox4x32bumpkey(key); ctr = _philox4x32round(ctr, key); } if(R>7){ key = _philox4x32bumpkey(key); ctr = _philox4x32round(ctr, key); } if(R>8){ key = _philox4x32bumpkey(key); ctr = _philox4x32round(ctr, key); } if(R>9){ key = _philox4x32bumpkey(key); ctr = _philox4x32round(ctr, key); } if(R>10){ key = _philox4x32bumpkey(key); ctr = _philox4x32round(ctr, key); } if(R>11){ key = _philox4x32bumpkey(key); ctr = _philox4x32round(ctr, key); } if(R>12){ key = _philox4x32bumpkey(key); ctr = _philox4x32round(ctr, key); } if(R>13){ key = _philox4x32bumpkey(key); ctr = _philox4x32round(ctr, key); } if(R>14){ key = _philox4x32bumpkey(key); ctr = _philox4x32round(ctr, key); } if(R>15){ key = _philox4x32bumpkey(key); ctr = _philox4x32round(ctr, key); } return ctr; }
namespace r123{ template<unsigned int ROUNDS> struct Philox2x32_R{ typedef r123array2x32 ctr_type; typedef r123array1x32 key_type; typedef r123array1x32 ukey_type; static const constant unsigned int rounds=ROUNDS; inline ctr_type operator()(ctr_type ctr, key_type key) const __attribute__((always_inline)){ static_assert(ROUNDS<=16, "philox is only unrolled up to 16 rounds\n"); return philox2x32_R(ROUNDS, ctr, key); } }; typedef Philox2x32_R<philox2x32_rounds> Philox2x32; }
namespace r123{ template<unsigned int ROUNDS> struct Philox4x32_R{ typedef r123array4x32 ctr_type; typedef r123array2x32 key_type; typedef r123array2x32 ukey_type; static const constant unsigned int rounds=ROUNDS; inline ctr_type operator()(ctr_type ctr, key_type key) const __attribute__((always_inline)){ static_assert(ROUNDS<=16, "philox is only unrolled up to 16 rounds\n"); return philox4x32_R(ROUNDS, ctr, key); } }; typedef Philox4x32_R<philox4x32_rounds> Philox4x32; }
typedef ulong BenchU64;
typedef uint BenchU32;
struct BenchParams {
    BenchU64 seed, base, words, k0, k1, k2;
    BenchU32 r0, r1, r2, key0, key1;
    BenchU32 logical_lanes, draws_shift, physical_threads, output_shift, reserved;
};
#include <metal_stdlib>
using namespace metal;
struct Params { ulong seed; ulong base; ulong draws; };
inline ulong rotl64(ulong x, uint r){ r&=63u; return (x<<r)|(x>>((64u-r)&63u)); }
inline ulong splitmix64(ulong x){
  x+=0x9E3779B97F4A7C15UL;
  x=(x^(x>>30))*0xBF58476D1CE4E5B9UL;
  x=(x^(x>>27))*0x94D049BB133111EBUL;
  return x^(x>>31);
}
inline ulong pcg_dxsm(ulong x){
  ulong w=(x>>32)^x, rot=x>>59;
  w^=w>>25; w*=0xA24BAED496EE9B5FUL;
  w^=w>>28; w*=0x9FB21C651E98DF25UL;
  w^=w>>27;
  return (w>>rot)|(w<<((-rot)&63));
}
inline uint mmc_f(ulong x, ulong k, uint r0, uint r1){
  ulong y = (x&1UL) ? (rotl64(x*3UL + k, r0) ^ (x>>1)) : (rotl64(x + k, r1) ^ (x*5UL));
  return (uint)(y & 0xFFFFFFFFUL);
}
struct MmcState{ ulong s,gamma,k0,k1,k2; uint r0,r1,r2; };
inline void mmc_init(thread MmcState& st, ulong seed, ulong lane){
  st.s=seed; st.gamma=(0x9E3779B97F4A7C15UL + (lane<<1)) | 1UL;
  st.k0=splitmix64(seed); st.k1=splitmix64(st.k0); st.k2=splitmix64(st.k1);
  ulong r=splitmix64(st.k2);
  st.r0=((uint)( r &63u))|1u; st.r1=((uint)((r>>8 )&63u))|1u; st.r2=((uint)((r>>16)&63u))|1u;
}
inline ulong perm3(ulong x, thread const MmcState& st){
  ulong L=(x>>32)&0xFFFFFFFFUL, R=x&0xFFFFFFFFUL, t; uint F;
  F=mmc_f(R, st.k0, st.r0, st.r1); L^=F; t=L; L=R; R=t;
  F=mmc_f(R, st.k1, st.r1, st.r2); L^=F; t=L; L=R; R=t;
  F=mmc_f(R, st.k2, st.r2, st.r0); L^=F; t=L; L=R; R=t;
  return ((L & 0xFFFFFFFFUL)<<32) | (R & 0xFFFFFFFFUL);
}
kernel void mmc_fill(device ulong* out [[buffer(0)]],
                     constant Params& P [[buffer(1)]],
                     uint tid [[thread_position_in_grid]])
{
  ulong lane=(ulong)tid; MmcState st; mmc_init(st, P.seed, lane);
  ulong ofs = lane * P.draws;
  ulong start = P.base + lane * P.draws;
  st.s = st.s + start * st.gamma;
  for (ulong i=0;i<P.draws;++i){
    st.s += st.gamma;
    ulong y = perm3(st.s, st);
    y = pcg_dxsm(y);
    ulong mant = (y >> 11) & ((1UL<<53)-1UL);
    out[ofs + i] = mant;
  }
}
struct MmcPair32 { uint lo, hi; };
inline uint mmc_rot_low32(uint lo,uint hi,uint r) {
    uint shift=r&31u;
    uint a=(r&32u)?hi:lo;
    uint b=(r&32u)?lo:hi;
    return (a<<shift)|(b>>(32u-shift));
}
inline uint mmc_f32(uint x,ulong key,uint r0,uint r1) {
    uint mask=0u-(x&1u);
    uint tlo=x+((x<<1)&mask);
    uint thi=((x>>31)+uint(tlo<x))&mask;
    uint lo=tlo+uint(key);
    uint hi=thi+uint(key>>32)+uint(lo<tlo);
    uint rotation=(r0&mask)|(r1&~mask);
    uint correction=((x>>1)&mask)|((x*5u)&~mask);
    return mmc_rot_low32(lo,hi,rotation)^correction;
}
inline MmcPair32 mmc_perm32(ulong x,thread const MmcState& st) {
    uint left=uint(x>>32),right=uint(x),next;
    next=left^mmc_f32(right,st.k0,st.r0,st.r1); left=right; right=next;
    next=left^mmc_f32(right,st.k1,st.r1,st.r2); left=right; right=next;
    next=left^mmc_f32(right,st.k2,st.r2,st.r0); left=right; right=next;
    return MmcPair32{right,left};
}
inline MmcPair32 mmc_xor_right32(MmcPair32 x,uint shift) {
    return MmcPair32{x.lo^((x.lo>>shift)|(x.hi<<(32u-shift))),x.hi^(x.hi>>shift)};
}
inline MmcPair32 mmc_mul32(MmcPair32 x,uint clo,uint chi) {
    return MmcPair32{x.lo*clo,metal::mulhi(x.lo,clo)+x.hi*clo+x.lo*chi};
}
inline ulong mmc_mix32(MmcPair32 x) {
    uint rotation=x.hi>>27;
    MmcPair32 w={x.lo^x.hi,x.hi};
    w=mmc_xor_right32(w,25);
    w=mmc_mul32(w,0x96EE9B5Fu,0xA24BAED4u);
    w=mmc_xor_right32(w,28);
    w=mmc_mul32(w,0x1E98DF25u,0x9FB21C65u);
    w=mmc_xor_right32(w,27);
    uint nonzero=0u-uint(rotation!=0);
    uint complementary=(32u-rotation)&31u;
    uint lo=(w.lo>>rotation)|((w.hi<<complementary)&nonzero);
    uint hi=(w.hi>>rotation)|((w.lo<<complementary)&nonzero);
    return ulong(lo)|(ulong(hi)<<32);
}
template<bool Split32>
inline ulong mmc_variant_word(ulong counter,thread const MmcState& state) {
    return Split32?mmc_mix32(mmc_perm32(counter,state)):pcg_dxsm(perm3(counter,state));
}
inline MmcState mmc_prepared_state(constant BenchParams& p) {
    MmcState state={};
    state.k0=p.k0; state.k1=p.k1; state.k2=p.k2;
    state.r0=p.r0; state.r1=p.r1; state.r2=p.r2;
    return state;
}
template<bool Split32>
inline void mmc_indexed_body(device ulong* out,constant BenchParams& p,uint tid) {
    MmcState state=mmc_prepared_state(p);
    for(ulong i=tid;i<p.words;i+=p.physical_threads) {
        ulong lane=(i>>p.draws_shift)&(p.logical_lanes-1);
        ulong gamma=0x9E3779B97F4A7C15UL+2*lane;
        ulong counter=p.seed+(p.base+i+1)*gamma;
        out[i]=mmc_variant_word<Split32>(counter,state)>>p.output_shift;
    }
}
template<bool Split32>
inline void mmc_tiled_body(device ulong* out,constant BenchParams& p,uint tid) {
    if(p.reserved==0xFFFFFFFFu) { mmc_indexed_body<Split32>(out,p,tid); return; }
    uint shift=p.reserved;
    uint width=1u<<shift;
    uint lane=tid>>shift;
    uint first=tid&(width-1u);
    ulong draws=1UL<<p.draws_shift;
    if(lane>=p.logical_lanes || first>=draws) return;
    ulong offset=ulong(lane)<<p.draws_shift;
    ulong gamma=0x9E3779B97F4A7C15UL+2*ulong(lane);
    ulong counter=p.seed+(p.base+offset+first+1)*gamma;
    ulong step=gamma<<shift;
    MmcState state=mmc_prepared_state(p);
    for(ulong i=first;i<draws;i+=width) {
        out[offset+i]=mmc_variant_word<Split32>(counter,state)>>p.output_shift;
        counter+=step;
    }
}
constant uint mmc_rot0 [[function_constant(0)]];
constant uint mmc_rot1 [[function_constant(1)]];
constant uint mmc_rot2 [[function_constant(2)]];
inline uint mmc_f_mulhi(uint x,ulong key,uint ro,uint re) {
    uint factor=1u+((x&1u)<<1);
    uint low=x*factor;
    uint high=metal::mulhi(x,factor);
    uint sum=low+uint(key);
    high+=uint(key>>32)+uint(sum<low);
    return mmc_rot_low32(sum,high,(x&1u)?ro:re)^((x&1u)?(x>>1):(x*5u));
}
inline uint mmc_f_dual(uint x,ulong key,uint ro,uint re) {
    uint lo3=x*3u, hi3=metal::mulhi(x,3u);
    uint oddlo=lo3+uint(key), evenlo=x+uint(key);
    uint oddhi=hi3+uint(key>>32)+uint(oddlo<lo3);
    uint evenhi=uint(key>>32)+uint(evenlo<x);
    uint odd=mmc_rot_low32(oddlo,oddhi,ro)^(x>>1);
    uint even=mmc_rot_low32(evenlo,evenhi,re)^(x*5u);
    return (x&1u)?odd:even;
}
template<uint Core>
inline uint mmc_round_variant(uint x,ulong key,uint ro,uint re) {
    if(Core==0) return mmc_f32(x,key,ro,re);
    if(Core==1) return mmc_f_mulhi(x,key,ro,re);
    if(Core==2) return mmc_f(ulong(x),key,ro,re);
    return mmc_f_dual(x,key,ro,re);
}
template<uint Core>
inline void mmc_batch_round(thread MmcPair32& x,ulong key,uint ro,uint re) {
    uint next=x.hi^mmc_round_variant<Core>(x.lo,key,ro,re);
    x.hi=x.lo; x.lo=next;
}
template<uint Core>
inline MmcState mmc_batch_state(constant BenchParams& p) { return mmc_prepared_state(p); }
template<>
inline MmcState mmc_batch_state<2>(constant BenchParams& p) {
    MmcState s=mmc_prepared_state(p);
    s.r0=mmc_rot0; s.r1=mmc_rot1; s.r2=mmc_rot2;
    return s;
}
template<>
inline MmcState mmc_batch_state<3>(constant BenchParams& p) { return mmc_batch_state<2>(p); }
template<uint Batch,uint Core,bool Reduce>
inline void mmc_batch_body(device ulong* out,constant BenchParams& p,uint tid) {
    MmcState state=mmc_batch_state<Core>(p);
    ulong checksum=0;
    if(p.reserved==0xFFFFFFFFu) {
        for(uint i=tid;i<uint(p.words);i+=p.physical_threads) {
            uint lane=(i>>p.draws_shift)&(p.logical_lanes-1);
            ulong counter=p.seed+(p.base+ulong(i)+1)*(0x9E3779B97F4A7C15UL+2*ulong(lane));
            MmcPair32 x={uint(counter),uint(counter>>32)};
            mmc_batch_round<Core>(x,state.k0,state.r0,state.r1);
            mmc_batch_round<Core>(x,state.k1,state.r1,state.r2);
            mmc_batch_round<Core>(x,state.k2,state.r2,state.r0);
            ulong value=mmc_mix32(x)>>p.output_shift;
            if(Reduce) checksum^=value; else out[i]=value;
        }
        if(Reduce) out[tid]=checksum;
        return;
    }
    uint width=1u<<p.reserved;
    uint lane=tid>>p.reserved;
    uint first=(tid&(width-1u))*Batch;
    uint draws=1u<<p.draws_shift;
    if(lane>=p.logical_lanes || first>=draws) {
        if(Reduce) out[tid]=0;
        return;
    }
    uint offset=lane<<p.draws_shift;
    ulong gamma=0x9E3779B97F4A7C15UL+2*ulong(lane);
    ulong counter=p.seed+(p.base+ulong(offset)+first+1)*gamma;
    uint stride=width*Batch;
    ulong step=gamma*ulong(stride);
    for(uint j=first;j<draws;j+=stride) {
        MmcPair32 x[Batch];
#pragma unroll
        for(uint b=0;b<Batch;++b) {
            ulong c=counter+ulong(b)*gamma;
            x[b]=MmcPair32{uint(c),uint(c>>32)};
        }
#pragma unroll
        for(uint b=0;b<Batch;++b) mmc_batch_round<Core>(x[b],state.k0,state.r0,state.r1);
#pragma unroll
        for(uint b=0;b<Batch;++b) mmc_batch_round<Core>(x[b],state.k1,state.r1,state.r2);
#pragma unroll
        for(uint b=0;b<Batch;++b) mmc_batch_round<Core>(x[b],state.k2,state.r2,state.r0);
#pragma unroll
        for(uint b=0;b<Batch;++b) if(j+b<draws) {
            ulong value=mmc_mix32(x[b])>>p.output_shift;
            if(Reduce) checksum^=value; else out[offset+j+b]=value;
        }
        counter+=step;
    }
    if(Reduce) out[tid]=checksum;
}
template<uint Engine,bool Reduce>
inline void reference_diagnostic_body(device ulong* out,constant BenchParams& p,uint tid) {
    ulong checksum=0;
    if(Engine==6 && p.reserved!=0xFFFFFFFFu) {
        uint width=1u<<p.reserved, lane=tid>>p.reserved, first=tid&(width-1u);
        ulong draws=1UL<<p.draws_shift, offset=ulong(lane)<<p.draws_shift;
        if(lane<p.logical_lanes && first<draws) {
            MmcState st=mmc_prepared_state(p);
            ulong gamma=0x9E3779B97F4A7C15UL+2*ulong(lane);
            ulong counter=p.seed+(p.base+offset+first+1)*gamma;
            for(ulong j=first;j<draws;j+=width) {
                ulong value=mmc_variant_word<true>(counter,st)>>p.output_shift;
                if(Reduce) checksum^=value; else out[offset+j]=value;
                counter+=gamma<<p.reserved;
            }
        }
    } else if(Engine==2) {
        philox4x32_key_t key={{p.key0,p.key1}};
        for(ulong i=tid;i<p.words/2;i+=p.physical_threads) {
            ulong index=p.base+i;
            philox4x32_ctr_t c={{uint(index),uint(index>>32),0,0}};
            auto v=philox4x32_R(10,c,key);
            ulong a=(ulong(v.v[0])|(ulong(v.v[1])<<32))>>p.output_shift;
            ulong b=(ulong(v.v[2])|(ulong(v.v[3])<<32))>>p.output_shift;
            if(Reduce) checksum^=a^b; else { out[2*i]=a; out[2*i+1]=b; }
        }
    } else {
        MmcState st=mmc_prepared_state(p);
        threefry2x32_key_t key={{p.key0,p.key1}};
        for(ulong i=tid;i<p.words;i+=p.physical_threads) {
            ulong value;
            if(Engine==1) {
                ulong index=p.base+i;
                threefry2x32_ctr_t c={{uint(index),uint(index>>32)}};
                auto v=threefry2x32_R(20,c,key);
                value=ulong(v.v[0])|(ulong(v.v[1])<<32);
            } else if(Engine==6) {
                ulong lane=(i>>p.draws_shift)&(p.logical_lanes-1);
                value=mmc_variant_word<true>(p.seed+(p.base+i+1)*(0x9E3779B97F4A7C15UL+2*lane),st);
            } else value=(p.base+i)^p.seed;
            value>>=p.output_shift;
            if(Reduce) checksum^=value; else out[i]=value;
        }
    }
    if(Reduce) out[tid]=checksum;
}
kernel void montmory_indexed(device ulong* out [[buffer(0)]],
                            constant BenchParams& p [[buffer(1)]],
                            uint tid [[thread_position_in_grid]]) {
    MmcState st;
    st.k0=p.k0; st.k1=p.k1; st.k2=p.k2;
    st.r0=p.r0; st.r1=p.r1; st.r2=p.r2;
    for (ulong i=tid; i<p.words; i+=p.physical_threads) {
        ulong lane=(i>>p.draws_shift)&(p.logical_lanes-1);
        ulong gamma=0x9E3779B97F4A7C15UL+2*lane;
        ulong counter=p.seed+(p.base+i+1)*gamma;
        out[i]=pcg_dxsm(perm3(counter,st))>>p.output_shift;
    }
}
kernel void threefry_bulk(device ulong* out [[buffer(0)]],
                         constant BenchParams& p [[buffer(1)]],
                         uint tid [[thread_position_in_grid]]) {
    threefry2x32_key_t key={{p.key0,p.key1}};
    for (ulong i=tid; i<p.words; i+=p.physical_threads) {
        ulong index=p.base+i;
        threefry2x32_ctr_t counter={{uint(index),uint(index>>32)}};
        threefry2x32_ctr_t value=threefry2x32_R(20,counter,key);
        out[i]=(ulong(value.v[0])|(ulong(value.v[1])<<32))>>p.output_shift;
    }
}
kernel void philox_bulk(device ulong* out [[buffer(0)]],
                       constant BenchParams& p [[buffer(1)]],
                       uint tid [[thread_position_in_grid]]) {
    philox4x32_key_t key={{p.key0,p.key1}};
    for (ulong i=tid; i<p.words/2; i+=p.physical_threads) {
        ulong index=p.base+i;
        philox4x32_ctr_t counter={{uint(index),uint(index>>32),0,0}};
        philox4x32_ctr_t value=philox4x32_R(10,counter,key);
        out[2*i]=(ulong(value.v[0])|(ulong(value.v[1])<<32))>>p.output_shift;
        out[2*i+1]=(ulong(value.v[2])|(ulong(value.v[3])<<32))>>p.output_shift;
    }
}
kernel void montmory_u32_indexed(device ulong* out [[buffer(0)]],
                                constant BenchParams& p [[buffer(1)]],
                                uint tid [[thread_position_in_grid]]) {
    mmc_indexed_body<true>(out,p,tid);
}
kernel void montmory_u64_tiled(device ulong* out [[buffer(0)]],
                              constant BenchParams& p [[buffer(1)]],
                              uint tid [[thread_position_in_grid]]) {
    mmc_tiled_body<false>(out,p,tid);
}
kernel void montmory_u32_tiled(device ulong* out [[buffer(0)]],
                              constant BenchParams& p [[buffer(1)]],
                              uint tid [[thread_position_in_grid]]) {
    mmc_tiled_body<true>(out,p,tid);
}
kernel void write_baseline(device ulong* out [[buffer(0)]],
                           constant BenchParams& p [[buffer(1)]],
                           uint tid [[thread_position_in_grid]]) {
    for(ulong i=tid;i<p.words;i+=p.physical_threads)
        out[i]=((p.base+i)^p.seed)>>p.output_shift;
}
kernel void reference_kat(device uint* out [[buffer(0)]],
                          const device uint* inputs [[buffer(1)]],
                          uint tid [[thread_position_in_grid]]) {
    if (tid>=6) return;
    uint offset=tid*6;
    if (tid<3) {
        threefry2x32_ctr_t c={{inputs[offset],inputs[offset+1]}};
        threefry2x32_key_t k={{inputs[offset+4],inputs[offset+5]}};
        threefry2x32_ctr_t r=threefry2x32_R(20,c,k);
        out[4*tid]=r.v[0]; out[4*tid+1]=r.v[1];
        out[4*tid+2]=0; out[4*tid+3]=0;
    } else {
        philox4x32_ctr_t c={{inputs[offset],inputs[offset+1],inputs[offset+2],inputs[offset+3]}};
        philox4x32_key_t k={{inputs[offset+4],inputs[offset+5]}};
        philox4x32_ctr_t r=philox4x32_R(10,c,k);
        for (uint j=0;j<4;++j) out[4*tid+j]=r.v[j];
    }
}
kernel void montmory_pair(device ulong* out [[buffer(0)]],constant BenchParams& p [[buffer(1)]],uint tid [[thread_position_in_grid]]) { mmc_batch_body<2,0,false>(out,p,tid); }
kernel void montmory_quad(device ulong* out [[buffer(0)]],constant BenchParams& p [[buffer(1)]],uint tid [[thread_position_in_grid]]) { mmc_batch_body<4,0,false>(out,p,tid); }
kernel void montmory_mulhi_pair(device ulong* out [[buffer(0)]],constant BenchParams& p [[buffer(1)]],uint tid [[thread_position_in_grid]]) { mmc_batch_body<2,1,false>(out,p,tid); }
kernel void montmory_spec_pair(device ulong* out [[buffer(0)]],constant BenchParams& p [[buffer(1)]],uint tid [[thread_position_in_grid]]) { mmc_batch_body<2,2,false>(out,p,tid); }
kernel void montmory_spec_dual(device ulong* out [[buffer(0)]],constant BenchParams& p [[buffer(1)]],uint tid [[thread_position_in_grid]]) { mmc_batch_body<2,3,false>(out,p,tid); }
kernel void montmory_pair_reduce(device ulong* out [[buffer(0)]],constant BenchParams& p [[buffer(1)]],uint tid [[thread_position_in_grid]]) { mmc_batch_body<2,0,true>(out,p,tid); }
kernel void montmory_quad_reduce(device ulong* out [[buffer(0)]],constant BenchParams& p [[buffer(1)]],uint tid [[thread_position_in_grid]]) { mmc_batch_body<4,0,true>(out,p,tid); }
kernel void montmory_mulhi_pair_reduce(device ulong* out [[buffer(0)]],constant BenchParams& p [[buffer(1)]],uint tid [[thread_position_in_grid]]) { mmc_batch_body<2,1,true>(out,p,tid); }
kernel void montmory_spec_pair_reduce(device ulong* out [[buffer(0)]],constant BenchParams& p [[buffer(1)]],uint tid [[thread_position_in_grid]]) { mmc_batch_body<2,2,true>(out,p,tid); }
kernel void montmory_spec_dual_reduce(device ulong* out [[buffer(0)]],constant BenchParams& p [[buffer(1)]],uint tid [[thread_position_in_grid]]) { mmc_batch_body<2,3,true>(out,p,tid); }
kernel void threefry_bulk_reduce(device ulong* out [[buffer(0)]],constant BenchParams& p [[buffer(1)]],uint tid [[thread_position_in_grid]]) { reference_diagnostic_body<1,true>(out,p,tid); }
kernel void philox_bulk_reduce(device ulong* out [[buffer(0)]],constant BenchParams& p [[buffer(1)]],uint tid [[thread_position_in_grid]]) { reference_diagnostic_body<2,true>(out,p,tid); }
kernel void montmory_u32_tiled_reduce(device ulong* out [[buffer(0)]],constant BenchParams& p [[buffer(1)]],uint tid [[thread_position_in_grid]]) { reference_diagnostic_body<6,true>(out,p,tid); }
kernel void write_baseline_reduce(device ulong* out [[buffer(0)]],constant BenchParams& p [[buffer(1)]],uint tid [[thread_position_in_grid]]) { reference_diagnostic_body<7,true>(out,p,tid); }

// Exact rewrites of the v3 Montmory generator. Append after generated/v3.metal.
template<bool Specialized>
inline uint opt_f(uint x,ulong key,uint ro,uint re) {
    if(!Specialized) return mmc_f_mulhi(x,key,ro,re);
    uint factor=1u+((x&1u)<<1);
    uint low=x*factor;
    uint high=metal::mulhi(x,factor);
    uint sum=low+uint(key);
    high+=uint(key>>32)+uint(sum<low);
    // Two rotations with seed-specific constants instead of a per-word shift.
    // The Metal compiler chooses whether to predicate or branch this expression.
    return (x&1u) ? (mmc_rot_low32(sum,high,ro)^(x>>1))
                  : (mmc_rot_low32(sum,high,re)^(x*5u));
}
template<bool Specialized>
inline void opt_round(thread MmcPair32& x,ulong key,uint ro,uint re) {
    uint next=x.hi^opt_f<Specialized>(x.lo,key,ro,re);
    x.hi=x.lo;x.lo=next;
}
inline MmcPair32 opt_add(MmcPair32 a,MmcPair32 b) {
    uint low=a.lo+b.lo;
    return MmcPair32{low,a.hi+b.hi+uint(low<a.lo)};
}
inline MmcPair32 opt_pair(ulong x) {return MmcPair32{uint(x),uint(x>>32)};}

template<uint Batch,bool Striped,bool Specialized,bool SplitCounter>
inline void opt_body(device ulong* out,constant BenchParams& p,uint tid) {
    MmcState state=mmc_batch_state<Specialized?2u:1u>(p);
    if(p.reserved==0xFFFFFFFFu) {
        for(uint i=tid;i<uint(p.words);i+=p.physical_threads) {
            uint lane=(i>>p.draws_shift)&(p.logical_lanes-1);
            ulong gamma=0x9E3779B97F4A7C15UL+2*ulong(lane);
            MmcPair32 x=opt_pair(p.seed+(p.base+ulong(i)+1)*gamma);
            opt_round<Specialized>(x,state.k0,state.r0,state.r1);
            opt_round<Specialized>(x,state.k1,state.r1,state.r2);
            opt_round<Specialized>(x,state.k2,state.r2,state.r0);
            out[i]=mmc_mix32(x)>>p.output_shift;
        }
        return;
    }
    uint width=1u<<p.reserved,lane=tid>>p.reserved;
    uint slot=tid&(width-1u);
    uint first=Striped?slot:slot*Batch;
    uint draws=1u<<p.draws_shift;
    if(lane>=p.logical_lanes || first>=draws)return;
    uint offset=lane<<p.draws_shift;
    uint spacing=Striped?width:1u;
    uint stride=width*Batch;
    ulong gamma=0x9E3779B97F4A7C15UL+2*ulong(lane);
    ulong counter=p.seed+(p.base+ulong(offset)+first+1)*gamma;
    ulong nextWord=gamma*ulong(spacing),nextBatch=gamma*ulong(stride);
    MmcPair32 counter32=opt_pair(counter),word32=opt_pair(nextWord),batch32=opt_pair(nextBatch);
    for(uint j=first;j<draws;j+=stride) {
        MmcPair32 x[Batch];
        if(SplitCounter) {
            MmcPair32 c=counter32;
#pragma unroll
            for(uint b=0;b<Batch;++b) {x[b]=c;c=opt_add(c,word32);}
        } else {
#pragma unroll
            for(uint b=0;b<Batch;++b)x[b]=opt_pair(counter+ulong(b)*nextWord);
        }
#pragma unroll
        for(uint b=0;b<Batch;++b)opt_round<Specialized>(x[b],state.k0,state.r0,state.r1);
#pragma unroll
        for(uint b=0;b<Batch;++b)opt_round<Specialized>(x[b],state.k1,state.r1,state.r2);
#pragma unroll
        for(uint b=0;b<Batch;++b)opt_round<Specialized>(x[b],state.k2,state.r2,state.r0);
#pragma unroll
        for(uint b=0;b<Batch;++b)if(j+b*spacing<draws)
            out[offset+j+b*spacing]=mmc_mix32(x[b])>>p.output_shift;
        if(SplitCounter)counter32=opt_add(counter32,batch32);else counter+=nextBatch;
    }
}
kernel void montmory_mulhi_quad(device ulong* out [[buffer(0)]],constant BenchParams& p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {opt_body<4,false,false,false>(out,p,tid);}
kernel void montmory_stripe_pair(device ulong* out [[buffer(0)]],constant BenchParams& p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {opt_body<2,true,false,false>(out,p,tid);}
kernel void montmory_stripe_quad(device ulong* out [[buffer(0)]],constant BenchParams& p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {opt_body<4,true,false,false>(out,p,tid);}
kernel void montmory_const_pair(device ulong* out [[buffer(0)]],constant BenchParams& p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {opt_body<2,false,true,false>(out,p,tid);}
kernel void montmory_const_stripe_pair(device ulong* out [[buffer(0)]],constant BenchParams& p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {opt_body<2,true,true,false>(out,p,tid);}
kernel void montmory_split_stripe_pair(device ulong* out [[buffer(0)]],constant BenchParams& p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {opt_body<2,true,false,true>(out,p,tid);}
kernel void montmory_const_stripe_quad(device ulong* out [[buffer(0)]],constant BenchParams& p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {opt_body<4,true,true,false>(out,p,tid);}

namespace montmory_wide_metal {
typedef uint4 V;
inline V splat(uint x) {return V(x);}
inline V mask_lt(V a,V b) {return select(V(0),V(0xffffffffu),a<b);}
inline V mask_ne(V a,V b) {return select(V(0),V(0xffffffffu),a!=b);}
inline V high_product(V a,V b) {return metal::mulhi(a,b);}

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

inline Wide64 key(ulong k) {return {splat(uint(k)),splat(uint(k>>32))};}
inline Wide64 counters(ulong c,ulong step) {
    return {V(uint(c),uint(c+step),uint(c+2*step),uint(c+3*step)),
            V(uint(c>>32),uint((c+step)>>32),uint((c+2*step)>>32),uint((c+3*step)>>32))};
}
template<bool Striped>
inline void body(device ulong* out,constant BenchParams& p,uint tid) {
    // Irregular worker counts keep the exact scalar indexing fallback.
    if(p.reserved==0xffffffffu) {
        MmcState s=mmc_prepared_state(p);
        for(uint i=tid;i<uint(p.words);i+=p.physical_threads) {
            uint lane=(i>>p.draws_shift)&(p.logical_lanes-1);
            ulong c=p.seed+(p.base+ulong(i)+1)*(0x9E3779B97F4A7C15UL+2*ulong(lane));
            out[i]=mmc_mix32(mmc_perm32(c,s))>>p.output_shift;
        }
        return;
    }
    uint width=1u<<p.reserved,lane=tid>>p.reserved,slot=tid&(width-1u);
    uint first=Striped?slot:4*slot,spacing=Striped?width:1u;
    uint draws=1u<<p.draws_shift;
    if(lane>=p.logical_lanes || first>=draws)return;
    uint offset=lane<<p.draws_shift,stride=4*width;
    ulong gamma=0x9E3779B97F4A7C15UL+2*ulong(lane);
    Wide64 c=counters(p.seed+(p.base+ulong(offset)+first+1)*gamma,ulong(spacing)*gamma);
    Wide64 step=key(ulong(stride)*gamma),k0=key(p.k0),k1=key(p.k1),k2=key(p.k2);
    for(uint j=first;j<draws;j+=stride) {
        Wide64 x=wide_round(c,k0,p.r0,p.r1);
        x=wide_round(x,k1,p.r1,p.r2);x=wide_round(x,k2,p.r2,p.r0);
        x=wide_mix(x);
#pragma unroll
        for(uint b=0;b<4;++b)if(j+b*spacing<draws)
            out[offset+j+b*spacing]=(ulong(x.lo[b])|(ulong(x.hi[b])<<32))>>p.output_shift;
        c=wide_add(c,step);
    }
}
}
kernel void montmory_vector4(device ulong* out [[buffer(0)]],constant BenchParams& p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
    montmory_wide_metal::body<false>(out,p,tid);
}
kernel void montmory_vector4_striped(device ulong* out [[buffer(0)]],constant BenchParams& p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
    montmory_wide_metal::body<true>(out,p,tid);
}
