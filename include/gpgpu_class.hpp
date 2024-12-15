#pragma once
#include <algorithm>
#include <string>
#include <sstream>
#include <iostream>
#include <utility>


class gpgpu_solo {
 private:
    std::string cmd;      // 仿真什么程序（这个也交给我们设置）
    std::string gpu_type; // 存储 GPU 类型

    std::string chip_location; // 存储芯片位置（这个可以交给我们来设置）


 public:
    // 构造函数，初始化所有成员变量
    explicit gpgpu_solo(std::string cmd = "$BENCHMARK_ROOT/bin/matmul_cu",

                        std::string gpu_type = "SM7_QV100",
                        std::string chip_location = "0 0")
        : cmd(std::move(cmd)), gpu_type(std::move(gpu_type)), chip_location(std::move(chip_location)) {}

    static std::string convertToFormattedString(const std::string &input) {
        std::istringstream stream(input);
        std::string token;
        std::string result;

        bool first = true;
        while (stream >> token) {
            if (!first) {
                result += ", ";
            }
            result += "\"" + token + "\"";
            first = false;
        }

        return result;
    }

    [[nodiscard]] std::string result() const {
        // 根据 chip_location 生成 log 文件名
        std::string log_name = "gpgpusim." + chip_location + ".log";
        // 将空格替换为下划线
        std::replace(log_name.begin(), log_name.end(), ' ', '.');

        std::stringstream ss;
        ss << "  - cmd: \"" << cmd << "\"\n"
           << "    args: [" << convertToFormattedString(chip_location) << "]\n"
           << "    log: \"" << log_name << "\"\n"
           << "    is_to_stdout: false\n"
           << "    clock_rate: 1\n"
           << "    pre_copy: \"$SIMULATOR_ROOT/gpgpu-sim/configs/tested-cfgs/" << gpu_type << "/*\"\n";
        return ss.str();
    }

    [[nodiscard]] std::string getCmd() const { return cmd; }
    void setCmd(const std::string &newCmd) { cmd = newCmd; }



    [[nodiscard]] std::string getGpuType() const { return gpu_type; }
    void setGpuType(const std::string &newGpuType) { gpu_type = newGpuType; }

    [[nodiscard]] std::string getChipLocation() const { return chip_location; }
    void setChipLocation(const std::string &newLocation) { chip_location = newLocation; }
};
