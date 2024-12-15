# -*- coding: utf-8 -*-
import json
import ortools
import time
from ortools.linear_solver import pywraplp
import tkinter as tk



def solve_problem(target):
    with open('lib.json', 'r') as file:
        lib = json.load(file)

    lib_len = len(lib)

    solver = pywraplp.Solver.CreateSolver('SCIP')
    infinity = solver.infinity()
    def check_max(value):
        if(value == 0):
            return infinity
        return value
    x = {}
    for i in range(lib_len):
        x[i] = solver.IntVar(0, infinity, 'x[%i]' % i)

    constraint_area = solver.Constraint(target["area_min"], check_max(target["area_max"]))
    for i in range(lib_len):
        constraint_area.SetCoefficient(x[i], lib[i]["area"])

    constraint_common_compoutitly = solver.Constraint(target["common_compoutitly_min"], check_max(target["common_compoutitly_max"]))
    for i in range(lib_len):
        constraint_common_compoutitly.SetCoefficient(x[i], lib[i]["common_compoutitly"])

    constraint_accelerate_compoutitly = solver.Constraint(target["accelerate_compoutitly_min"], check_max(target["accelerate_compoutitly_max"]))
    for i in range(lib_len):
        constraint_accelerate_compoutitly.SetCoefficient(x[i], lib[i]["accelerate_compoutitly"])

    constraint_cpu_memory = solver.Constraint(target["cpu_memory_min"], check_max(target["cpu_memory_max"]))
    for i in range(lib_len):
        constraint_cpu_memory.SetCoefficient(x[i], lib[i]["cpu_memory"])

    constraint_gpu_memory = solver.Constraint(target["gpu_memory_min"], check_max(target["gpu_memory_max"]))
    for i in range(lib_len):
        constraint_gpu_memory.SetCoefficient(x[i], lib[i]["gpu_memory"])

    constraint_cpu_num = solver.Constraint(target["cpu_num_min"], check_max(target["cpu_num_max"]))
    for i in range(lib_len):
        constraint_cpu_num.SetCoefficient(x[i], lib[i]["is_cpu"])

    constraint_gpu_num = solver.Constraint(target["gpu_num_min"], check_max(target["gpu_num_max"]))
    for i in range(lib_len):
        constraint_gpu_num.SetCoefficient(x[i], lib[i]["is_gpu"])

    objective = solver.Objective()
    for i in range(lib_len):
        objective.SetCoefficient(x[i], lib[i]["cost"])

    objective.SetMinimization()

    status = solver.Solve()

    solution = {}

    if status == pywraplp.Solver.OPTIMAL:
        print('Solution:')
        for i in range(lib_len):
            print('x[%i] = ' % i, x[i].solution_value())
            if(x[i].solution_value() > 0):
                solution[lib[i]["name"]] = x[i].solution_value()
        print('Optimal objective value =', solver.Objective().Value())
    else:
        print('The problem does not have an optimal solution.')
    with open('solution.json', 'w') as file:
        json.dump(solution, file, indent=4)

# 创建Tkinter窗口
root = tk.Tk()
root.title("输入目标值")

# 读取 target.json 文件中的数值
try:
    with open('target.json', 'r') as file:
        target_data = json.load(file)
except FileNotFoundError:
    target_data = {}

# 验证输入框内容是否为数字
def validate_numeric_input(P):
    if P.isdigit() or P == "." or P == "":
        return True
    else:
        return False

vcmd = (root.register(validate_numeric_input), '%P')

# 创建输入框和标签
entries = {}
labels = {
    "area": "面积：",
    "common_compoutitly": "通用算力：",
    "accelerate_compoutitly": "加速算力：",
    "cpu_memory": "内存（MB）：",
    "gpu_memory": "显存（MB）：",
    "cpu_num": "CPU数量：",
    "gpu_num": "GPU数量："
}

# 预设默认值
default_values = {
    "area_min": 0,
    "area_max": 2000,
    "common_compoutitly_min": 0,
    "common_compoutitly_max": 10,
    "accelerate_compoutitly_min": 0,
    "accelerate_compoutitly_max": 10,
    "cpu_memory_min": 0,
    "cpu_memory_max": 1024,
    "gpu_memory_min": 0,
    "gpu_memory_max": 16384,
    "cpu_num_min": 1,
    "cpu_num_max": 16,
    "gpu_num_min": 1,
    "gpu_num_max": 8
}

row = 0
for key, text in labels.items():
    label = tk.Label(root, text=text)
    label.grid(row=row, column=0, padx=10, pady=5, sticky="e")
    entry_min = tk.Entry(root)
    entry_min.grid(row=row, column=1, padx=10, pady=5)
    entry_max = tk.Entry(root)
    entry_max.grid(row=row, column=2, padx=10, pady=5)
    entries[key + "_min"] = entry_min
    entries[key + "_max"] = entry_max
    # 设置默认最小值
    min_key = key + "_min"
    min_value = target_data.get(min_key, default_values.get(min_key, 0))
    entry_min.insert(0, min_value)
    # 设置默认最大值
    max_key = key + "_max"
    max_value = target_data.get(max_key, default_values.get(max_key, 100))
    entry_max.insert(0, max_value)
    entry_min.config(validate="key", validatecommand=vcmd)
    entry_max.config(validate="key", validatecommand=vcmd)
    row += 1

def on_submit():
    target = {}
    for key, entry in entries.items():
        value = entry.get()
        if value == "":
            value = 0
        target[key] = float(value)
    # 将输入数值保存在 target.json 文件中
    with open('target.json', 'w') as file:
        json.dump(target, file, indent=4)
    print(target)
    solve_problem(target)
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, f"求解器已完成求解。")
    

submit_button = tk.Button(root, text="Submit", command=on_submit)
submit_button.grid(row=row, columnspan=3, pady=10)

# 添加文本框
row += 1
result_text = tk.Text(root, height=1, width=50)
result_text.grid(row=row, columnspan=3, padx=10, pady=10)

root.mainloop()