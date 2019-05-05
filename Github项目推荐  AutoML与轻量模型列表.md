## Github项目推荐 | AutoML与轻量模型列表

AI研习社 [AI研习社](javascript:void(0);) *昨天*

![img](https://mmbiz.qpic.cn/mmbiz_png/bicdMLzImlibRKJzRrSpZicEC9kzAbKNOvdI49Y7xhUddYSIbpY3Yd0fWyibk92cgobjAIOK50m2VQXVy37haeNC2g/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

## 

# 

## 

### 



### **awesome-AutoML-and-Lightweight-Models**

by **guan-yuan**

本项目旨在为自动化研究(特别是轻量级模型)提供信息。有兴趣的同学可以进行收藏或者在Github中推荐/提交项目（论文、项目仓库等）。

高质量（最新）AutoML项目和轻量级模型汇总列表，列表包括以下内容：

- 1.神经结构搜索
- 2.轻量级结构
- 3.模型压缩和加速
- 4.超参数优化
- 5.自动化特征工程

### **Github项目地址：**

### [https://github.com/guan-yuan/awesome-AutoML-and-Lightweight-Models](https://mp.weixin.qq.com/s?__biz=MjM5ODU3OTIyOA==&mid=2650676242&idx=3&sn=add9301253b0d43bd356308b89b50eb8&chksm=bec2216189b5a877c0129fbb0c0908f9d5f0e1c49d104c9e2a922510128d4a16b3464e2cf832&mpshare=1&scene=1&srcid=&key=eeca034c219d0a57825f0f29cae3b33785dbc44551d53acf272e732573c9a73933215409daf6cc073f09bda6386a3278e98e027d651ef18ad72b80ee15970c5c1977cdcae4ef296a47e537eff90f9cf8&ascene=1&uin=MjMzNDA2ODYyNQ%3D%3D&devicetype=Windows+10&version=62060739&lang=zh_CN&pass_ticket=f7CgN3cUgVMojzAiWUZuwpvAv7bNHgQChXXDR2CF3ufVoHK%2Fz4hhJlYnE7b3oNOd)

详细论文内容，请点击**阅读原文**后点击相关链接访问。

## **1 神经结构搜索**

### **[论文]**

#### ***梯度：***

- ASAP: Architecture Search, Anneal and Prune | [2019/04]
- Single-Path NAS: Designing Hardware-Efficient ConvNets in less than 4 Hours | [2019/04]
- dstamoulis/single-path-nas | [Tensorflow]
- Automatic Convolutional Neural Architecture Search for Image Classification Under Different Scenes | [IEEE Access 2019]
- sharpDARTS: Faster and More Accurate Differentiable Architecture Search | [2019/03]
- Learning Implicitly Recurrent CNNs Through Parameter Sharing | [ICLR 2019]
- lolemacs/soft-sharing | [Pytorch]
- Probabilistic Neural Architecture Search | [2019/02]
- Auto-DeepLab: Hierarchical Neural Architecture Search for Semantic Image Segmentation | [2019/01]
- SNAS: Stochastic Neural Architecture Search | [ICLR 2019]
- FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search | [2018/12]
- Neural Architecture Optimization | [NIPS 2018]
- renqianluo/NAO | [Tensorflow]
- DARTS: Differentiable Architecture Search | [2018/06]
- quark0/darts | [Pytorch]
- khanrc/pt.darts | [Pytorch]
- dragen1860/DARTS-PyTorch | [Pytorch]

#### ***强化学习：***

- Template-Based Automatic Search of Compact Semantic Segmentation Architectures | [2019/04]
- Understanding Neural Architecture Search Techniques | [2019/03]
- Fast, Accurate and Lightweight Super-Resolution with Neural Architecture Search | [2019/01]
- falsr/FALSR | [Tensorflow]
- Multi-Objective Reinforced Evolution in Mobile Neural Architecture Search | [2019/01] 
- moremnas/MoreMNAS | [Tensorflow]
- ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware | [ICLR 2019]
- MIT-HAN-LAB/ProxylessNAS | [Pytorch, Tensorflow]
- Transfer Learning with Neural AutoML | [NIPS 2018]
- Learning Transferable Architectures for Scalable Image Recognition | [2018/07]
- wandering007/nasnet-pytorch | [Pytorch]
- tensorflow/models/research/slim/nets/nasnet | [Tensorflow]
- MnasNet: Platform-Aware Neural Architecture Search for Mobile | [2018/07]
- AnjieZheng/MnasNet-PyTorch | [Pytorch]
- Practical Block-wise Neural Network Architecture Generation | [CVPR 2018]
- Efficient Neural Architecture Search via Parameter Sharing | [ICML 2018]
- melodyguan/enas | [Tensorflow]
- carpedm20/ENAS-pytorch | [Pytorch]
- Efficient Architecture Search by Network Transformation | [AAAI 2018]

#### ***进化算法：***

- Single Path One-Shot Neural Architecture Search with Uniform Sampling | [2019/04]
- DetNAS: Neural Architecture Search on Object Detection | [2019/03]
- The Evolved Transformer | [2019/01]
- Designing neural networks through neuroevolution | [Nature Machine Intelligence 2019]
- EAT-NAS: Elastic Architecture Transfer for Accelerating Large-scale Neural Architecture Search | [2019/01]
- Efficient Multi-objective Neural Architecture Search via Lamarckian Evolution | [ICLR 2019]

#### ***SMBO（Sequential Model-Based Optimization - 基于序列模型的优化）：***

- MFAS: Multimodal Fusion Architecture Search | [CVPR 2019]
- DPP-Net: Device-aware Progressive Search for Pareto-optimal Neural Architectures | [ECCV 2018]
- Progressive Neural Architecture Search | [ECCV 2018]
- titu1994/progressive-neural-architecture-search | [Keras, Tensorflow]
- chenxi116/PNASNet.pytorch | [Pytorch]

#### ***随机搜索：***

- Exploring Randomly Wired Neural Networks for Image Recognition | [2019/04]
- Searching for Efficient Multi-Scale Architectures for Dense Image Prediction | [NIPS 2018]

#### ***超网络：***

- Graph HyperNetworks for Neural Architecture Search | [ICLR 2019]

#### ***贝叶斯优化：***

- Inductive Transfer for Neural Architecture Optimization | [2019/03]

#### ***偏序修剪：***

- Partial Order Pruning: for Best Speed/Accuracy Trade-off in Neural Architecture Search | [CVPR 2019]
- lixincn2015/Partial-Order-Pruning | [Caffe]

#### ***知识提炼：***

- Improving Neural Architecture Search Image Classifiers via Ensemble Learning | [2019/03]

### **[项目]**

- Microsoft/nni | [Python]



## **2 轻量级结构**

### **[论文]**

#### ***分割：***

- ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network | [2018/11]
- sacmehta/ESPNetv2 | [Pytorch]
- ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation | [ECCV 2018]
- sacmehta/ESPNet | [Pytorch]
- BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation | [ECCV 2018]
- ooooverflow/BiSeNet | [Pytorch]
- ycszen/TorchSeg | [Pytorch]
- ERFNet: Efficient Residual Factorized ConvNet for Real-time Semantic Segmentation | [T-ITS 2017]
- Eromera/erfnet_pytorch | [Pytorch]

#### ***物体检测：***

- Pooling Pyramid Network for Object Detection | [2018/09]
- tensorflow/models | [Tensorflow]
- Tiny-DSOD: Lightweight Object Detection for Resource-Restricted Usages | [BMVC 2018]
- lyxok1/Tiny-DSOD | [Caffe]
- Pelee: A Real-Time Object Detection System on Mobile Devices | [NeurIPS 2018]
- Robert-JunWang/Pelee | [Caffe]
- Robert-JunWang/PeleeNet | [Pytorch]
- Receptive Field Block Net for Accurate and Fast Object Detection | [ECCV 2018]
- ruinmessi/RFBNet | [Pytorch]
- ShuangXieIrene/ssds.pytorch | [Pytorch]
- lzx1413/PytorchSSD | [Pytorch]
- FSSD: Feature Fusion Single Shot Multibox Detector | [2017/12]
- ShuangXieIrene/ssds.pytorch | [Pytorch]
- lzx1413/PytorchSSD | [Pytorch]
- dlyldxwl/fssd.pytorch | [Pytorch]
- Feature Pyramid Networks for Object Detection | [CVPR 2017]
- tensorflow/models | [Tensorflow]

## 

## **3 模型压缩和加速**

### **[论文]**

#### ***压缩：***

- Slimmable Neural Networks | [ICLR 2019]
- JiahuiYu/slimmable_networks | [Pytorch]
- AMC: AutoML for Model Compression and Acceleration on Mobile Devices | [ECCV 2018]
- AutoML for Model Compression (AMC): Trials and Tribulations | [Pytorch]
- Learning Efficient Convolutional Networks through Network Slimming | [ICCV 2017]
- foolwood/pytorch-slimming | [Pytorch]
- Channel Pruning for Accelerating Very Deep Neural Networks | [ICCV 2017]
- yihui-he/channel-pruning | [Caffe]
- Pruning Convolutional Neural Networks for Resource Efficient Inference | [ICLR 2017]
- jacobgil/pytorch-pruning | [Pytorch]
- Pruning Filters for Efficient ConvNets | [ICLR 2017]

#### ***加速：***

- Fast Algorithms for Convolutional Neural Networks | [CVPR 2016]
- andravin/wincnn | [Python]

### **[项目]**

- NervanaSystems/distiller | [Pytorch]
- Tencent/PocketFlow | [Tensorflow]

### **[教程/博客]**

- Introducing the CVPR 2018 On-Device Visual Intelligence Challenge

##  

## **4 超参数优化**

### **[论文]**

- Tuning Hyperparameters without Grad Students: Scalable and Robust Bayesian Optimisation with Dragonfly | [2019/03]
- dragonfly/dragonfly
- Google vizier: A service for black-box optimization | [SIGKDD 2017]

### **[项目]**

- Microsoft/nni | [Python]
- dragonfly/dragonfly | [Python]

### **[教程/博客]**

- Hyperparameter tuning in Cloud Machine Learning Engine using Bayesian Optimization
- Overview of Bayesian Optimization
- Bayesian optimization
- krasserm/bayesian-machine-learning | [Python]

##  

## **5 自动化特征工程**

### **[模型分析器]**

- Netscope CNN Analyzer | [Caffe]
- sksq96/pytorch-summary | [Pytorch]
- Lyken17/pytorch-OpCounter | [Pytorch]

### **[参考]**

- LITERATURE ON NEURAL ARCHITECTURE SEARCH
- handong1587/handong1587.github.io
- hibayesian/awesome-automl-papers
- mrgloom/awesome-semantic-segmentation
- amusi/awesome-object-detection









**备受大家期待的强化学习课程终于上线啦！**

扫描下方邀请卡，解锁更多课时

![img](https://mmbiz.qpic.cn/mmbiz_png/bicdMLzImlibSpgphl35CPgEkd2BzG7jk93Ysry7AZWxhibuUPmG2d1FC3J9sDcrcicouhysXVjpAaLFVjRGXj9cuQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![img](https://mmbiz.qpic.cn/mmbiz_gif/bicdMLzImlibRAS3Tao2nfeJk00qqxX3axIgPV3yia4NPESGdUJEM9vsfw1O4Dg1iat7lVNAmbCMY65ia2pzfBXm5kg/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1)

点击 **阅读原文**，查看更多内容

[阅读原文](https://mp.weixin.qq.com/s?__biz=MjM5ODU3OTIyOA==&mid=2650676242&idx=3&sn=add9301253b0d43bd356308b89b50eb8&chksm=bec2216189b5a877c0129fbb0c0908f9d5f0e1c49d104c9e2a922510128d4a16b3464e2cf832&mpshare=1&scene=1&srcid=&key=eeca034c219d0a57825f0f29cae3b33785dbc44551d53acf272e732573c9a73933215409daf6cc073f09bda6386a3278e98e027d651ef18ad72b80ee15970c5c1977cdcae4ef296a47e537eff90f9cf8&ascene=1&uin=MjMzNDA2ODYyNQ%3D%3D&devicetype=Windows+10&version=62060739&lang=zh_CN&pass_ticket=f7CgN3cUgVMojzAiWUZuwpvAv7bNHgQChXXDR2CF3ufVoHK%2Fz4hhJlYnE7b3oNOd##)







微信扫一扫
关注该公众号