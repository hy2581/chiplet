// configuration.hpp
#pragma once

#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include "phase_config.hpp"

class configuration {
 private:
    std::vector<phase_config> phase1;
    std::vector<phase_config> phase2;
    std::string bench_file {"./bench.txt"};
    std::string delayinfo_file {"./delayInfo.txt"};

 public:
    void addPhase1Config(const phase_config &config) {
        phase1.push_back(config);
    }

    void addPhase2Config(const phase_config &config) {
        phase2.push_back(config);
    }

    [[nodiscard]] std::string result() const {
        std::stringstream ss;

        // Phase 1
        ss << "# Phase 1 configuration.\nphase1:\n";
        for (size_t i = 0; i < phase1.size(); ++i) {
            ss << "  # Process " << i << "\n"
               << phase1[i].result();
        }

        // Phase 2
        ss << "\n# Phase 2 configuration.\nphase2:\n";
        for (size_t i = 0; i < phase2.size(); ++i) {
            ss << "  # Process " << i << "\n"
               << phase2[i].result();
        }

        // File configuration
        ss << "\n# File configuration. (Not used yet)\n"
           << "bench_file: \"" << bench_file << "\"\n"
           << "delayinfo_file: \"" << delayinfo_file << "\"\n";

        return ss.str();
    }

    // Setter 和 Getter 方法
    void setBenchFile(const std::string &value) { bench_file = value; }
    void setDelayinfoFile(const std::string &value) { delayinfo_file = value; }
};
