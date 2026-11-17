# TokenSlotTranslate 设计文档 v1.0

## 接口和生命周期

solution.py公开prepare、run，prepare返回Prepared。一次性验证输入类型、元数据和值域，按op.json允许回读CPU。设备计算由Triton完成。Prepared保存原始Tensor对象和固定shape/device/S，不保存data_ptr或派生索引。

run分配新的连续int32输出，在输入设备调用时的当前stream启动。N=0返回正确形状空Tensor。合法原位更新和同形状set_均读取当前数据。

## 整数算法

令x=max(slot,0)。S=2^k时q=x>>k，r=x&(S-1)。非幂二S令C=ceil(2^32/S)，q0=(int64(x)*C)>>32。因为0<=x<2^31，倒数近似误差满足0<=x*(C/2^32-1/S)<1/2，q0为正确商或商+1。q0*S>x时减1，再计算r=x-q*S。纠偏乘法和比较使用int64。prepare保证M*S+S-1<=INT32_MAX，最终乘加使用int32。负slot使用安全索引0，最终结果为-1。

## 分核与内存

grid=(ceil(N/TILE),L)，TILE=max(256,min(2048,next_power_of_2(N)))。每个program处理一层的一段token，尾部mask。layer*B+q为映射偏移。

B<=8192时，连续读取本层映射表到片上缓存，再用tl.gather查表。当前后端gather接受fp32 lane，将int32数据按原始32位模式bitcast为fp32，查表后bitcast回int32，保持所有整数位。过程中只进行位复制。S=1的2147483646/2147483647及近int32上界输出用例验证该路径。

B>8192时，按q从全局内存读取映射，使用有效slot与尾部mask。该通用路径无需片上保存整表。launch只根据N、B、S等元数据选分块和分支。

显式分配仅输出4*L*N字节。两个主性能输入输出为16384、8388608字节。片上映射表最大32768字节，TILE上限2048用于控制编译器片上临时缓冲区。峰值按预热后PyTorch NPU分配器活跃分配增量统计，保留prepared与输入，单次调用后同步再取峰值；删除输出并执行20次调用后同步、gc，验证活跃分配恢复。原始结果与版本见task_submission/result.json。

## 验证

fixtures/case.json为检查清单；golden.calculate为CPU标杆，golden.reference("baseline").prepare为冻结NPU基线。计时排除输入准备、prepare、首次编译和CPU标杆。驱动实现两侧各50次预热、交替6轮×200次完整入口墙钟，包含新输出分配、Python提交、设备计算及轮末同步。按配对轮次共同重采样10000次，seed=20261031，分位数线性插值。

主性能2项，典型集数量ceil(2/10)=1，固定slot_perf_large作为调优集；达标后全量测试。14个附件正向输入、10项拒绝、5项动态更新、4项set_、7项补充边界及直接run均逐项保存；两个性能输入分别验证两条stream各保留3份输出、allocator峰值和20次释放回收。S64/S96诊断各执行完整配对协议。

## 环境与贡献范围

Atlas A3 Ascend910_9382、CANN9.1.0，Python3.10.12、PyTorch2.7.1+cpu、torch_npu2.7.1.post8、triton-ascend3.2.2。代码采用Apache-2.0许可证。独立组件完成验收后，按用户要求向xLLM-AI/xllm的NPU算子目录提交代码PR；当前阶段交付私人仓固定commit和验收材料。
