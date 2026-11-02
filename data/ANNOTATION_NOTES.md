# 推理前标注约定与复核范围

此处是合成 fixture 的业务定义，不是真实设备规格。
标签由 `build_fixtures.py` 的 SPECS 独立生成，在任何模型推理前写定。
自然语言模板仅是表达变体，不能当成独立场景覆盖数。

关键样本按协议逐项核对：

- `Camera Mode: Slow Motion` 对应 CameraModeType/SLOWMOTION，wire ID=0。
- `Frame Rate: 48 FPS` 对应 VideoFrameRateType/__48FPS，wire ID=4。
- `Frame Rate: 144 FPS` 无精确成员；1440FPS 不等价，必须 UNKNOWN。
- `CameraModeType/SLOWMOTION=0` 与 `WorkingModeType/CAPTURE=0` 不是同一枚举。
- `Working Mode: record video` 表示工作状态 RECORDING，不是相机拍摄模式 VIDEONORMAL。
- ISO 100 与 wire ID 0 不同；-2EV 与 wire ID 0 也不同。
- 曝光补偿 +1 和 1 数值等价；-1 与 +1 不等价。
- Image Texture/Crisp 是新增参数类型；不把它当作 Sharpness/High 的等价答案。
- 新增 1440FPS 候选前，该参数不受支持；新增后期望对应 wire ID=13。

负例约定：此 fixture 未声明的模式/值均为不支持，
例如 Hyperlapse（带运动路径的延时）并未声明与 TIMELAPSE 等价；
shade 与 cloudy、horizon lock 与 stabilization boost、extra high 与 high
也没有等价约定，所以作为相近但不同的负例。
不进行最接近值回退，不自动取整 59.94FPS→60FPS。

命名实验 HIGH 与 IMAGE_ACUTANCE_PLUS 的等价只由
`naming_variants.json` 的显式语义契约确定，分别在两个独立候选空间测试。
不能把这个合成定义套用到其他真实协议。

复核由本次实现中的规范对照与自动一致性检查完成；
未请真实业务专家或用户进行独立人工标注，不声称存在此类人工验收。
