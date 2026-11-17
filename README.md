# Token slot 地址翻译

Triton Ascend独立组件。接口为solution.prepare、solution.run及prepared.run。硬件Atlas A3，CANN9.1.0。已验证Python3.10.12、torch2.7.1+cpu、torch_npu2.7.1.post8、triton-ascend3.2.2（triton模块版本3.2.0）。

## 安装和使用

```bash
source /usr/local/Ascend/cann/set_env.sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export ASCEND_RT_VISIBLE_DEVICES=0
```

需Linux aarch64环境的C++编译器和匹配Python开发头文件，用于首次JIT编译。

```python
import torch
import torch_npu
from solution import prepare

slots = torch.tensor([-1, 0, 3, 4], dtype=torch.int32, device="npu:0")
mapping = torch.tensor([[3, 0]], dtype=torch.int32, device="npu:0")
prepared = prepare([slots, mapping], {"block_size": 4})
output = prepared.run()  # [[-1, 12, 15, 0]]; 每次返回新输出
```

prepare允许一次性回读检查值域。prepared保留Tensor对象，读取合法更新和同形状set_后的当前存储，在调用时的当前stream执行。shape/dtype/device改变时抛出ValueError；页长改变后须重新prepare。

## 完整验证

从本目录运行验证驱动，测试产物输出到工程外：

```bash
python3 verify.py --output-dir ../token-slot-results --confirm-task-timing
```

完整性能协议为每侧预热50次、交替6轮、每轮200次，墙钟包括完整prepared.run与轮末设备同步。首次编译和prepare在计时外。--confirm-task-timing表示按附件协议执行。失败或中断保存FAIL与错误堆栈并返回非零退出码。

仅功能验证：

```bash
python3 verify.py --output-dir ../token-slot-functional --functional-only
```

CPU验证从本组件根目录运行：

```bash
python3 -m unittest discover -s tests -v
```

CPU检查覆盖附件标杆、整数算法模型、统计与CPU输入拒绝；NPU编译、功能、双stream、allocator和性能由真机验证产生。

固定典型性能调优集为slot_perf_large，来自2个主性能用例的ceil(2/10)=1项，覆盖最大输出规模：

```bash
python3 verify.py --output-dir ../token-slot-typical --typical --confirm-task-timing
```

fixtures为本任务原始CPU标杆、NPU基线、生成器和用例。case.json中的performance_reference和expected_checks来自任务发布方；本次实测在task_submission/result.json、generalization.json及日志中。设计见docs/design.md，复现和报告见task_submission/。
