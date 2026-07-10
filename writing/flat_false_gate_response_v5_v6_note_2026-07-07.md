# 平面假目标门响应 v5/v6 记录

## 1. 物理响应应如何理解

在激光脉冲和接收门都近似为矩形函数时，单个深度点的门响应不是任意线性衰减，而是两个矩形时间窗的重叠长度归一化：

```text
W_g(R) = overlap(rect_laser(R), rect_gate(R)) / min(T_laser, T_gate)
```

等价到深度轴上，如果目标深度为 `R`、门中心为 `R_g`：

```text
half_sum = 0.5 * (gate_width + pulse_width)
cap = min(gate_width, pulse_width)
overlap = min(max(half_sum - abs(R - R_g), 0), cap)
W_g(R) = overlap / cap
```

因此响应形状是三角形或梯形，而不是所有 gate 上都简单线性衰减。只有当相邻 gate 中心落在 `half_sum` 范围内时，旁门才有非零响应；如果 gate 间距大于这个支撑范围，旁门会自然变成 0。

## 2. 对平面假目标的含义

平面假目标的合理建模不是“只出现模型的一部分”，而是：

```text
整个目标轮廓在同一个平面深度上反射；
不同 gate 的亮度由 W_g(R_flat) 决定；
如果 gate 与该平面深度不重叠，整幅轮廓会整体变暗甚至消失。
```

这与真三维目标不同。真三维目标在不同 gate 中对应不同深度切片，所以形状区域和亮度都会随 gate 改变；平面假目标则应保持同一轮廓，只改变整体门响应强度。

## 3. v5 设置

v5 使用：

```text
FlatGeometryMode = flatten-camera-depth
FlatTargetGateIndexMode = round-robin
FlatMinResponse = 0.35
FlatEchoGain = 2.0
ReflectanceMode = hash-log-uniform
ReflectanceMin = 0.6
ReflectanceMax = 2.8
```

结果：

| 指标 | flat_false | true3d |
|---|---:|---:|
| gate-stack corr | 0.9993 | 0.3011 |
| mask IoU | 0.9826 | 0.2892 |
| absdiff | 0.0058 | 0.1330 |

三种子训练：

| 输入 | mean best val acc |
|---|---:|
| Full 3-gate stack | 0.9545 |
| Gate 0 only | 0.8788 |
| Gate 1 only | 0.9394 |
| Gate 2 only | 0.9545 |

解释：

v5 保持了“平面假目标跨门轮廓一致”的物理直觉，同时通过反射率随机化削弱了 v4 的直接亮度捷径。但 `FlatMinResponse=0.35` 仍然是人为保底，不是纯矩形卷积结果。

## 4. v6 设置

v6 只改一项：

```text
FlatMinResponse = 0.0
```

也就是平面假目标完全按矩形脉冲/矩形门重叠响应显示，没有旁门保底。

结果：

| 指标 | flat_false | true3d |
|---|---:|---:|
| gate-stack corr | 0.4193 | 0.3011 |
| mask IoU | 0.3596 | 0.2892 |
| absdiff | 0.1686 | 0.1330 |
| max/mean ratio | 150.7863 | 34.7651 |

seed42 训练：

| 输入 | best val acc |
|---|---:|
| Full 3-gate stack | 1.0000 |
| Gate 1 only | 0.9545 |

解释：

v6 更接近“无旁门保底”的矩形卷积物理模型，但由于当前 gate 间距相对较大，平面假目标在非命中 gate 中接近黑帧。这样会带来新的黑帧/亮帧捷径，反而不利于证明网络利用了真正的多 gate 结构。

## 5. 当前结论

不能简单说 `FlatMinResponse=0` 就一定更好。更准确的说法是：

```text
平面假目标的门响应应由矩形脉冲与矩形接收门卷积决定；
但门宽、脉宽、门间距、曝光归一化必须一起设计。
如果相邻门没有重叠，纯卷积会造成强黑帧捷径；
如果人为保底过高，又会引入非物理的持续亮度。
```

## 6. 下一版 v7 建议

推荐做一个“重叠门 + 曝光控制”的 v7，而不是继续调 PNG 后处理：

1. 重新渲染 true3d 和 flat_false，统一使用更宽的 `receiver_gate_width` 或更小的 gate spacing，使相邻 gate 有非零重叠响应。
2. 保持 `FlatMinResponse=0`，让旁门亮度来自真实矩形卷积，而不是人为保底。
3. 加入每样本反射率随机化，继续保留 v5 的 `hash-log-uniform`。
4. 增加弱背景散射、Poisson 光子噪声和轻微模糊，降低黑帧/峰值捷径。
5. 做单门消融、gate dropout、scalar baseline 诊断，只有当 full stack 明显优于单门时再作为主结果。

论文中建议把 v5 作为当前最佳可解释控制组，把 v6 作为物理响应负结果/消融，用来说明为什么必须联合设计选通参数。
