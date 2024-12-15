#include "configuration.hpp"
#include "gem5_class.hpp"
#include "gpgpu_class.hpp"
#include <boost/filesystem.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp> // 需要安装 nlohmann/json 库
#include <stdexcept>
#include <string>

using json = nlohmann::json;
namespace fs = boost::filesystem;

void extractSolutionContent(const fs::path& solution_path,
                            std::string& string_gpu, std::string& string_cpu) {
    std::ifstream solution_file(solution_path.string()); // 转换为 std::string
    if (!solution_file) {
        throw std::runtime_error("Failed to open solution file: " +
                                 solution_path.string());
    }
    json solution_json;
    solution_file >> solution_json;

    for (auto& item : solution_json.items()) {
        const std::string& component_name = item.key();
        // const json &count = item.value(); // 如果不使用 count，可以移除这一行
        if (component_name.find("GTX") != std::string::npos) {
            string_gpu = component_name;
        } else if (component_name.find("SM") != std::string::npos) {
            string_gpu = component_name;
        } else if (component_name.find("CPU") != std::string::npos) {
            string_cpu = component_name;
        }
    }
}

int main() {
    try {
        // 检查并设置根目录
#ifdef ROOT_DIR
        std::cout << "Project root directory: " << ROOT_DIR << std::endl;
#else
        std::cout << "ROOT_DIR macro is not defined." << std::endl;
#endif
        fs::path root_dir = ROOT_DIR;
        fs::path filename = root_dir / "benchmark" / "my_test" / "test3" /
                            "matmul" / "test_o3.yml";

        // 创建目录结构
        fs::create_directories(filename.parent_path());
        // 打开输出文件
        std::ofstream outfile(filename.string()); // 转换为 std::string
        if (!outfile) {
            throw std::runtime_error("Failed to open file: " +
                                     filename.string());
        }

        // 读取 solution.json 文件
        fs::path solution_path = root_dir / "python" / "solution.json";
        std::ifstream solution_file(
            solution_path.string()); // 转换为 std::string
        if (!solution_file) {
            throw std::runtime_error("Failed to open solution file: " +
                                     solution_path.string());
        }
        json solution_json;
        solution_file >> solution_json;

        // 创建配置对象
        configuration config;
        std::string string_gpu;
        std::string string_cpu;
        extractSolutionContent(solution_path, string_gpu, string_cpu);

        // 添加 GPGPU-Sim 配置
        for (int i = 1; i <= 4; ++i) {
            gpgpu_solo gpgpu_inst;
            gpgpu_inst.setChipLocation(std::to_string(i) + " 0");

            gpgpu_inst.setGpuType(string_gpu);
            phase_config gpu_config(gpgpu_inst);
            config.addPhase1Config(gpu_config);
        }

        // 根据 solution.json 添加 GEM5 配置
        int process_count = 0;
        {
            auto component_name = string_cpu;
            int count = 3;
            for (int i = 0; i < count; ++i) {
                if (component_name == "O3CPU_01") {
                    gem5_solo gem5_inst;
                    gem5_inst.setCpuType("O3CPU");
                    gem5_inst.setL1dSize("128kB");
                    gem5_inst.setL1iSize("32kB");
                    gem5_inst.setL2Size("512kB");

                    gem5_inst.setChipLocation(std::to_string(process_count) +
                                              " " + std::to_string(i));

                    phase_config gem5_config(gem5_inst);
                    config.addPhase1Config(gem5_config);
                }
                if (component_name == "O3CPU_02") {
                    gem5_solo gem5_inst;
                    gem5_inst.setCpuType("O3CPU");
                    gem5_inst.setL1dSize("64kB");
                    gem5_inst.setL1iSize("16kB");
                    gem5_inst.setL2Size("256kB");

                    gem5_inst.setChipLocation(std::to_string(process_count) +
                                              " " + std::to_string(i));

                    phase_config gem5_config(gem5_inst);
                    config.addPhase1Config(gem5_config);
                }
                if (component_name == "O3CPU_03") {
                    gem5_solo gem5_inst;
                    gem5_inst.setCpuType("O3CPU");
                    gem5_inst.setL1dSize("256kB");
                    gem5_inst.setL1iSize("64kB");
                    gem5_inst.setL2Size("1024kB");

                    gem5_inst.setChipLocation(std::to_string(process_count) +
                                              " " + std::to_string(i));

                    phase_config gem5_config(gem5_inst);
                    config.addPhase1Config(gem5_config);
                }
            }

            ++process_count;
        }

        // 添加 Phase 2 配置
        phase_config popnet_config;
        popnet_config.setCmd("$SIMULATOR_ROOT/popnet_chiplet/build/popnet");
        popnet_config.setArgs({"-A", "7",    "-c", "2",
                               "-V", "18",   "-B", "20",
                               "-O", "20",   "-F", "10",
                               "-L", "1000", "-T", "10000000",
                               "-r", "1",    "-I", "../bench.txt",
                               "-R", "0",    "-D", "../delayInfo.txt",
                               "-P"});
        popnet_config.setLog("popnet_0.log");
        popnet_config.setClockRate(1);
        config.addPhase2Config(popnet_config);

        // 写入配置
        outfile << config.result();
        outfile.close();

        std::cout << "Configuration written to: " << filename.string()
                  << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}