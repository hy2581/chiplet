#pragma once
#include <algorithm>
#include <string>
#include <iostream>
#include <utility>
#include <sstream>

class gem5_solo {
  private:
    std::string cmd;
    std::string model;
    std::string cpu_type;
    bool caches;
    bool _cmd;
    std::string load;
    bool _o;
    std::string chip_location;
    std::string l1d_size; // 默认值 "128kB"
    std::string l1i_size; // 默认值 "32kB"
    std::string l2_size;  // 默认值 "512kB"

  public:
    explicit gem5_solo(
        std::string cmd = "$SIMULATOR_ROOT/gem5/build/X86/gem5.opt",
        std::string model =
            "$SIMULATOR_ROOT/gem5/configs/deprecated/example/se.py",
        std::string cpu_type = "BaseO3CPU",
        bool caches = true,
        bool _cmd = true,
        std::string load = "$BENCHMARK_ROOT/bin/matmul_c",
        bool _o = true,
        std::string chip_location = "0 0",
        std::string l1d_size = "128kB",
        std::string l1i_size = "32kB",
        std::string l2_size = "512kB")
        : cmd(std::move(cmd)), model(std::move(model)),
          cpu_type(std::move(cpu_type)), caches(caches), _cmd(_cmd),
          load(std::move(load)), _o(_o),
          chip_location(std::move(chip_location)),
          l1d_size(std::move(l1d_size)), l1i_size(std::move(l1i_size)),
          l2_size(std::move(l2_size)) {
    }

    [[nodiscard]] std::string result() const {
        // 根据 chip_location 生成 log 文件名
        std::string log_name = "gem5." + chip_location + ".log";
        // 将空格替换为下划线
        std::replace(log_name.begin(), log_name.end(), ' ', '_');

        std::stringstream ss;
        ss << "  - cmd: \"" << cmd << "\"\n"
           << "    args: [\"" << model << "\", "
           << R"("--cpu-type", ")" << cpu_type << "\", "
           << "\"--l1d_size=" << l1d_size << "\", "
           << "\"--l1i_size=" << l1i_size << "\", "
           << "\"--l2_size=" << l2_size << "\", "
           << (caches ? R"("--caches", "--l2cache", )" : "")
           << (_cmd ? "\"--cmd\", " : "") << "\"" << load << "\", "
           << (_o ? R"("-o", ")" + chip_location + "\"" : "") << "]\n"
           << "    log: \"" << log_name << "\"\n"
           << "    is_to_stdout: false\n"
           << "    clock_rate: 500\n";
        return ss.str();
    }

    // Getter 方法
    [[nodiscard]] const std::string& getCmd() const {
        return cmd;
    }
    [[nodiscard]] const std::string& getModel() const {
        return model;
    }
    [[nodiscard]] const std::string& getCpuType() const {
        return cpu_type;
    }
    [[nodiscard]] bool hasCaches() const {
        return caches;
    }
    [[nodiscard]] const std::string& getLoad() const {
        return load;
    }
    [[nodiscard]] const std::string& getChipLocation() const {
        return chip_location;
    }
    [[nodiscard]] const std::string& getL1dSize() const {
        return l1d_size;
    }
    [[nodiscard]] const std::string& getL1iSize() const {
        return l1i_size;
    }
    [[nodiscard]] const std::string& getL2Size() const {
        return l2_size;
    }

    // Setter 方法
    void setCmd(const std::string& newCmd) {
        cmd = newCmd;
    }
    void setModel(const std::string& newModel) {
        model = newModel;
    }
    void setCpuType(const std::string& newCpuType) {
        cpu_type = newCpuType;
    }
    void setCaches(bool newCaches) {
        caches = newCaches;
    }
    void setLoad(const std::string& newLoad) {
        load = newLoad;
    }
    void setChipLocation(const std::string& newLocation) {
        chip_location = newLocation;
    }
    void setL1dSize(const std::string& size) {
        l1d_size = size;
    }
    void setL1iSize(const std::string& size) {
        l1i_size = size;
    }
    void setL2Size(const std::string& size) {
        l2_size = size;
    }
};