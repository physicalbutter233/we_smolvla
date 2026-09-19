## 版本内容
**任务定义**：从三个固定位置中随机摆放蓝色的方块，要求机械臂将其抓取并放到盒子中
**数据信息**：每个位置的方块都抓取10次，数据存放位置为`we_smolvla/datasets/raw/so101_pick_blue_cube_917`
**训练参数**：batchsize为16，模型默认grad_accumulation为1，其他详见output的train_config.json与datasets的info.json
**评估参数**：max_relative_target=30，dataset.fps=30，n_action_steps=10。由于不知道lerobot_record默认为同步推理，因此本次评估使用lerobot_record的指令进行部署的，即模型测试是在同步推理的情况下进行的。
## 模型效果
**训练结果** 训练曲线如图所示[[images/task1_0.1的训练曲线.png]]，训练结果的位置为`we_smolvla/outputs/train/task1_917/checkpoints/040000/pretrained_model`。
**评估结果** 如视频所示`we_smolvla/outputs/train/task1_917/vedio/task1_0.1_evaluate.mp4`。从视频中可以看出，模型的空间能力很差，完全抓不住方块；并且动作卡顿。
## 问题总结
- **数据的质** 通过与[官方可视化数据](https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Flerobot%2Fsvla_so100_pickplace)对比可得，官方的数据中，腕部相机全程几乎保持不动、动作匀速、有稳定的补光条件，而这些是我们在这般数据中没有在意的，因此在后续的数据中，需要做到腕部相机几乎不动+尽量匀速+稳定光照条件。
- **评估细节** 评估阶段动作一段一段的，可能是因为同步推理导致的，后续使用异步推理，测试卡顿是否有好转。