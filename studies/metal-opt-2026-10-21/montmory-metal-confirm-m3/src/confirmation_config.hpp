// Frozen from the seven-seed search; no tuning on confirmation results.
#pragma once
#include <array>
struct ConfirmJob { unsigned engine, workers, group, shift; };
inline constexpr std::array<ConfirmJob, 22> confirmJobs = {{
    {1u,131072u,128u,0u}, // threefry2x32_20 raw64
    {1u,65536u,64u,11u}, // threefry2x32_20 mantissa53
    {2u,131072u,64u,0u}, // philox4x32_10 raw64
    {2u,131072u,128u,11u}, // philox4x32_10 mantissa53
    {7u,8192u,256u,0u}, // write_baseline_NOT_PRNG raw64
    {7u,8192u,128u,11u}, // write_baseline_NOT_PRNG mantissa53
    {10u,131072u,64u,0u}, // montmory_mulhi_pair raw64
    {10u,131072u,64u,11u}, // montmory_mulhi_pair mantissa53
    {13u,131072u,512u,0u}, // montmory_mulhi_quad raw64
    {13u,131072u,512u,11u}, // montmory_mulhi_quad mantissa53
    {14u,131072u,64u,0u}, // montmory_stripe_pair raw64
    {14u,131072u,256u,11u}, // montmory_stripe_pair mantissa53
    {15u,131072u,256u,0u}, // montmory_stripe_quad raw64
    {15u,131072u,256u,11u}, // montmory_stripe_quad mantissa53
    {16u,131072u,64u,0u}, // montmory_const_pair raw64
    {16u,131072u,128u,11u}, // montmory_const_pair mantissa53
    {17u,131072u,128u,0u}, // montmory_const_stripe_pair raw64
    {17u,131072u,128u,11u}, // montmory_const_stripe_pair mantissa53
    {18u,131072u,256u,0u}, // montmory_split_stripe_pair raw64
    {18u,131072u,512u,11u}, // montmory_split_stripe_pair mantissa53
    {19u,131072u,256u,0u}, // montmory_const_stripe_quad raw64
    {19u,131072u,64u,11u}, // montmory_const_stripe_quad mantissa53
}};
