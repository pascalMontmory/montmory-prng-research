#pragma once
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

inline double confirmMedian(std::vector<double> values) {
    if (values.empty()) throw std::runtime_error("Empty timing series");
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
}

inline std::vector<double> confirmSeedMedians(const std::vector<std::vector<double>>& samples,
                                             std::size_t expectedSeeds, std::size_t observationsPerSeed) {
    if (samples.size() != expectedSeeds) throw std::runtime_error("Incomplete seeds");
    std::vector<double> result;
    for (const auto& seed : samples) {
        if (seed.size() != observationsPerSeed) throw std::runtime_error("Incomplete per-seed measurements");
        result.push_back(confirmMedian(seed));
    }
    return result;
}
