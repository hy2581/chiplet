// phase_config.hpp
#pragma once
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include "gem5_class.hpp"
#include "gpgpu_class.hpp"

class phase_config {
 private:
    std::string cmd;
    std::vector<std::string> args;
    std::string log;
    bool is_to_stdout {false};
    int clock_rate {500};
    std::string pre_copy;
    std::string config_result;

 public:
    // 默认构造函数
    phase_config() = default;

    // 接受 gem5_solo 实例的构造函数
    explicit phase_config(const gem5_solo &gem5_instance) {
        config_result = gem5_instance.result();
    }

    // 接受 gpgpu_solo 实例的构造函数
    explicit phase_config(const gpgpu_solo &gpgpu_instance) {
        config_result = gpgpu_instance.result();
    }

    // 设置 gem5_solo 实例
    void setConfig(const gem5_solo &gem5_instance) {
        config_result = gem5_instance.result();
    }

    // 设置 gpgpu_solo 实例
    void setConfig(const gpgpu_solo &gpgpu_instance) {
        config_result = gpgpu_instance.result();
    }

    // 获取结果
    [[nodiscard]] std::string result() const {
        if (!config_result.empty()) {
            // 如果 config_result 非空，直接返回
            return config_result;
        }
        else {
            // 否则，根据自身的字段构建输出
            std::stringstream ss;
            ss << "  - cmd: \"" << cmd << "\"\n";
            ss << "    args: [";
            for (size_t i = 0; i < args.size(); ++i) {
                ss << "\"" << args[i] << "\"";
                if (i < args.size() - 1) {
                    ss << ", ";
                }
            }
            ss << "]\n";
            ss << "    log: \"" << log << "\"\n";
            ss << "    is_to_stdout: " << (is_to_stdout ? "true" : "false") << "\n";
            ss << "    clock_rate: " << clock_rate << "\n";
            if (!pre_copy.empty()) {
                ss << "    pre_copy: \"" << pre_copy << "\"\n";
            }
            return ss.str();
        }
    }

    // Getter 方法
    const std::string &getCmd() const { return cmd; }
    const std::vector<std::string> &getArgs() const { return args; }
    const std::string &getLog() const { return log; }
    bool getIsToStdout() const { return is_to_stdout; }
    int getClockRate() const { return clock_rate; }
    const std::string &getPreCopy() const { return pre_copy; }

    // Setter 方法
    void setCmd(const std::string &value) { cmd = value; }
    void setArgs(const std::vector<std::string> &value) { args = value; }
    void setLog(const std::string &value) { log = value; }
    void setIsToStdout(bool value) { is_to_stdout = value; }
    void setClockRate(int value) { clock_rate = value; }
    void setPreCopy(const std::string &value) { pre_copy = value; }
};