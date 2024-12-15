# 安装

## 下载仓库并设置环境

配置参考：WSL，Ubuntu 20.04，GCC 7，CUDA 11.0

另外建议增大 WSL 的 swap 到至少 4GB，不然很有可能在执行 `scons build/X86/gem5.opt` 这一步时无法完成。

1. **从 GitHub 上下载仓库。**

   ```bash
   git clone git@github.com:hy2581/chiplet.git
   ```

   进入仿真器根目录，以下的示例命令都假设从仿真器根目录开始执行。

2. **初始化并更新子模块（submodule）。**

   ```bash
   git submodule init
   git submodule update
   ```

3. **运行脚本，初始化环境变量。**

   ```bash
   source setup_env.sh
   ```

   运行成功应出现：`setup_environment succeeded`

4. **修改 snipersim 和 gpgpu-sim 代码。（这一步不能省略）**

   ```bash
   ./apply_patch.sh
   ```

   更多细节参见下文“打包和应用 Patch”章节。

5. **编译安装 snipersim。**
    新版本的 snipersim 提供了自动化的编译脚本，直接执行 `make` 即可。

   ```bash
   cd snipersim
   make -j4
   ```

6. **编译安装 Gem5。**
    请查看 Gem5 文档获取详细安装指南。LegoSim 中可以运行 X86 和 ARM 架构仿真器：

   ```bash
   cd gem5
   scons build/X86/gem5.opt
   ```

   或者

   ```bash
   cd gem5
   scons build/ARM/gem5.opt
   ```

7. **编译安装 GPGPU-Sim。**
    GPGPU-Sim 安装有前置条件：

   1. **安装 CUDA**：新版本的 GPGPU-Sim 可以支持 CUDA 4 到 CUDA 11 的任意版本，详细信息请参见 GPGPU-Sim 的 README。
   2. **编译器要求**：GPGPU-Sim 对于编译器版本有要求，建议使用 GCC 7。

   配置好 CUDA 和编译器后，可以直接执行 `make`。

   ```bash
   cd gpgpu-sim
   make -j4
   ```

8. **编译安装 popnet**

   ```bash
   cd popnet_chiplet
   mkdir build
   cd build
   cmake ..
   make -j4
   ```

9. **编译安装芯粒间通信程序。**`interchiplet` 提供了芯粒间通信所需要的 API 和实现代码。

   ```bash
   cd interchiplet
   mkdir build
   cd build
   cmake ..
   make
   ```

   编译完成后应找到 `interchiplet/bin/interchiplet`。

   **注意**：`zmq_pro` 需要安装 ZeroMQ 环境。通常会在 `cmake` 步骤被忽略。

10. **编译 make_yml，这段代码实现了仿真模块驱动的自动化生成。**

    ```bash
    cd ../..
    mkdir build
    cd build
    cmake ..
    make
    ```

# 验证安装

正确执行上述过程后，可以使用 `benchmark/matmul` 验证环境设置是否正确。

1. **设置仿真器环境**

   ```bash
   source setup_env.sh
   ```

2. **编译可执行文件**

   ```bash
   cd benchmark/matmul
   make
   ```

3. **执行可执行文件**示例包含 4 个进程，分别是 1 个 CPU 进程和 3 个 GPU 进程。必须在 `benchmark/matmul` 目录执行。

   ```bash
   ../../interchiplet/bin/interchiplet ./matmul.yml
   ```

   执行后，可以在 `benchmark/matmul` 目录下找到一组 `proc_r{R}_p{P}_t{T}` 的文件夹，对应于第 R 轮执行的第 P 阶段的第 T 个线程。

   在文件夹中可以找到以下文件：

   1. GPGPU-Sim 仿真的临时文件和日志文件 `gpgpusim_X_X.log`。
   2. Sniper 仿真的临时文件和日志文件 `sniper.log`。
   3. Popnet 的日志文件 `popnet.log`。

4. **清理可执行文件和输出文件。**

   ```bash
   make clean
   ```

5. **编译负载程序（待测例程）。**

   以上程序的正确执行，说明已经成功进行了仿真器的配置，下面开始仿真器的自动化。这一步是负载程序的编译：

   ```bash
   cd $SIMULATOR_ROOT
   cd benchmark/my_test/test3/matmul/
   cd make
   ```

6. **执行自动化**

   ```bash
   cd $SIMULATOR_ROOT
   cd result
   python3 $SIMULATOR_ROOT/python/solver.py
   ```

   如果这一步有问题，请再执行以下步骤：

   ```bash
   cd $SIMULATOR_ROOT
   source setup_env.sh
   cd result
   python3 $SIMULATOR_ROOT/python/solver.py
   ```

7. **清理可执行文件和输出文件。**

   ```bash
   make clean
   ```

# 打包和应用 Patch

由于 Sniper 和 GPGPU-Sim 是用子模块（submodule）方式引入的，对于 snipersim 和 gpgpu-sim 的修改不会通过常规的 Git 流程追踪。因此，工程提供了 `patch.sh` 和 `apply_patch.sh` 两个脚本通过 Patch 管理 snipersim 和 gpgpu-sim 的修改。

- **生成 Patch（patch.sh 脚本）：**

  ```bash
  ./patch.sh
  ```

  1. 使用 `patch.sh` 脚本将 snipersim 和 gpgpu-sim 的修改分别打包到 `snipersim.diff` 和 `gpgpu-sim.diff` 文件中。`diff` 文件保存在 `interchiplet/patch` 目录下，并会被 Git 追踪。
  2. `patch.sh` 脚本还会将被修改的文件按照文件层次结构保存到 `.changed_files` 文件夹中，用于在 `diff` 文件出错时进行查看和参考。

- **应用 Patch（apply_patch.sh 脚本）：**

  ```bash
  ./apply_patch.sh
  ```

  1. 使用 `apply_patch.sh` 脚本将 `snipersim.diff` 和 `gpgpu-sim.diff` 文件应用到 snipersim 和 gpgpu-sim，重现对文件的修改。
  2. 当应用出错时，可以参考 `.changed_files` 中的文件手动修改 snipersim 和 gpgpu-sim 的文件。

**注意**：不建议直接使用 `.changed_files` 覆盖 snipersim 和 gpgpu-sim 文件夹。因为 snipersim 和 gpgpu-sim 本身的更新可能会与芯粒仿真器修改相同的文件。使用 Patch 的方式会报告修改的冲突。如果直接覆盖，可能会导致不可预见的错误。

# 添加测试程序

测试程序统一添加到 `benchmark` 目录下，每一个测试程序都有独立的文件夹。

测试程序的文件管理推荐按照 `matmul` 组织，并使用类似的 Makefile，但并不绝对要求。

运行测试程序需要编写 YAML 配置文件。

## YAML 配置文件格式

```yaml
# Phase 1 configuration
phase1:
  # Process 0
  - cmd: "$BENCHMARK_ROOT/bin/matmul_cu"
    args: ["0", "1"]
    log: "gpgpusim.0.1.log"
    is_to_stdout: false
    pre_copy: "$SIMULATOR_ROOT/gpgpu-sim/configs/tested-cfgs/SM2_GTX480/*"
  # Process 1
  - cmd: "$BENCHMARK_ROOT/bin/matmul_cu"
    args: ["1", "0"]
    log: "gpgpusim.1.0.log"
    is_to_stdout: false
    pre_copy: "$SIMULATOR_ROOT/gpgpu-sim/configs/tested-cfgs/SM2_GTX480/*"
  # ...

# Phase 2 configuration
phase2:
  # Process 0
  - cmd: "$SIMULATOR_ROOT/popnet/popnet"
    args:
      ["-A", "2", "-c", "2", "-V", "3", "-B", "12", "-O", "12",
       "-F", "4", "-L", "1000", "-T", "10000000", "-r", "1",
       "-I", "../bench.txt", "-R", "0"]
    log: "popnet.log"
    is_to_stdout: false
```

YAML 配置文件的第一层支持以下关键字：

- `phase1`：配置第一阶段的仿真器进程。
- `phase2`：配置第二阶段的仿真器进程。

这两个关键字下面都是数组，每项对应一个并发的仿真器进程。`phase1` 和 `phase2` 都可以支持多个仿真进程。

仿真器进程的配置支持以下关键字：

- `cmd`：仿真器的命令。字符串形式，支持环境变量 `$BENCHMARK_ROOT` 和 `$SIMULATOR_ROOT`。
- `args`：仿真器的参数。字符串数组形式，支持环境变量 `$BENCHMARK_ROOT` 和 `$SIMULATOR_ROOT`。
- `log`：日志的名称。不能使用相对路径或绝对路径。
- `is_to_stdout`：是否将仿真器的标准输出/错误输出重定向到 interchiplet 的标准输出。
- `pre_copy`：有些仿真器需要额外的文件才能启动仿真。该关键字为字符串，如果需要复制多个文件，用空格隔开，并用引号包围。

在 YAML 文件中使用相对路径时，以当前路径作为基础。推荐使用环境变量构成绝对路径：

- `$BENCHMARK_ROOT`：表示测试程序的路径，根据 YAML 文件的位置决定。
- `$SIMULATOR_ROOT`：表示仿真器的路径，通过 `setup_env.sh` 决定。

## 运行 InterChiplet

仿真器的主程序是 `InterChiplet`。在运行路径下执行以下命令：

```bash
$SIMULATOR_ROOT/interchiplet/bin/interchiplet $BENCHMARK_ROOT/bench.yml
```

`InterChiplet` 命令格式如下：

```bash
interchiplet <bench>.yml [--cwd <string>] [-t|--timeout <int>] [-e|--error <float>] [-h]
```

命令参数说明：

- `<bench>.yml`：指定测试程序的配置文件。
- `--cwd <string>`：指定执行仿真的路径。
- `-t <int>` 或 `--timeout <int>`：指定仿真退出的轮次。不论结果是否收敛，都会结束仿真。
- `-e <float>` 或 `--error <float>`：指定仿真退出的条件。当仿真误差小于该比例时，结束仿真。
