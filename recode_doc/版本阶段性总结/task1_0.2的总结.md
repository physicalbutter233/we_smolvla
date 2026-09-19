## 版本内容

**任务定义**：从三个固定位置中随机摆放蓝色的方块，要求机械臂将其抓取并放到盒子中
**数据信息**：每个位置的方块都抓取10次，过程中尽量保持手腕相机不转动+匀速运动，数据存放位置为`we_smolvla/datasets/raw/so101_pick_blue_cube_918`
**训练参数**：batchsize为16，模型默认grad_accumulation为1，其他详见output的train_config.json与datasets的info.json
**评估参数**：采用异步推理，单步安全阈值robot.max_relative_target与动作执行频率fps不设置，默认分别为None和30，actions_per_chunk为50、chunk_size_threshold为0.5。

## 模型效果

**训练曲线** 如图所示[[../images/task1_0.2的训练曲线.png]]
**评估结果** 如视频所示`we_smolvla/outputs/train/task1_917/vedio/task1_0.2_evaluate.mp4`。从视频中可以看出，模型在能够连续两次完成任务（成功率有待测试），但是一旦第一次没有抓住物体，后续就再也抓不到了，即**模型的自纠功能极差**

## 问题总结

- **纠正数据的补充** 模型目前没有自纠的能力，下版本尝试引入**dagger**数据来完善这一功能。
