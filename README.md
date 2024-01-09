# Ai_Core
linear_over-parameterization:
1. 目前支持普通卷积nn.Conv2d（group=1，非1*1）的过参数化
2. 该算法应在模型从零开始训练时使用，因为其中的参数会重新进行初始化
3. 该算法的特点是：设计小模型 --> 训练线性过参数化大模型 --> 保存时合并为原始小模型结构。因此你可以设计一个小模型，然后用此算法训练提高小模型精度，最后推理部署仍使用小模型结构
4. 得到的小模型后续仍可叠加剪枝和量化等压缩算法

low_rank_decompose:
1. 目前支持nn.Conv2d（group>1不支持）和nn.Linear的分解
2. 建议从已经训练好的float模型开始finetune
3. 分解之后的模型调整好学习率，准确率一般会迅速恢复


auto_pruning:
1. 通过本工具得到训练好的float onnx模型，以及MNN模型压缩参数文件
2. 通过MNN转换工具，输入这两个文件，得到最终的MNN稀疏模型（如果直接部署稀疏的float模型，而不叠加量化，那么转换时需要使用MNNConvert的 --weightQuantBits 8 参数进行转换，才会进行稀疏编码，否则模型大小将不变）
模型稀疏之后，可以进一步使用PyTorch训练量化工具进行量化，得到稀疏量化模型。

weight_quantizer:
1. 建议从训练好的float模型进行finetune
2. 创建WeightQuantizer对象，对原始模型进行转换，然后用转换之后的模型进行训练
3. 保存onnx模型之前，去掉插入的节点
4. 保存onnx模型之后，导出MNN模型压缩参数文件，示例代码如下（关注其中quantizer的用法）：