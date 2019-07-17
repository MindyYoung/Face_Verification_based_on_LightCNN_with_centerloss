An Enhanced Light CNN with Ceterloss for Face Verificaion
===================

# 1.背景

《机器学习》课程大作业

我们考虑到应用的数据主要来自于互联网，因此标签有明显噪音，因此这里我们的工作是如何利用这一类大规模噪声标签数据来获取训练产生模型的泛化性能。

# 2.主要贡献

ELCNN 为对 CNN 的有限改进，其在 CNN 的池化层与卷积层中加入函数处理层，并在得到的特征后在特定的 softmax 损失函数迭代后加入带权 centerloss损失项平衡 softmax 中出现的部分数据冗余与内聚效果差的问题，在该过程中
ELCNN 的主要贡献如下：
- MFM 函数层能够去除数据中的噪声，丢弃部分数据，降低训练得到的特征数据的噪声。
- Softmax 函数能良好的体现类的中心距离，但在迭代后期会出现参数冗余内聚效果差的情况，我们在后期加入 centerloss 带权正则项，提高内聚效果。
- 在处理过程中，LCNN 又提出了预训练深度网络语义引导方法来处理噪声数据集，降低参数空间与模型复杂度从而加快运算速度提高模型性能。

# 3.环境说明
- caffe+python
- 将图像存入images_aligned_sample文件夹内
- same_pairs.txt：待测试的属于同一人的图像对的地址
- diff_pairs.txt：待测试的不属于同一人的图像对的地址
- 运行src/verif.py

# 4.参考文献

	@article{wulight,
	  title={A Light CNN for Deep Face Representation with Noisy Labels},
	  author={Wu, Xiang and He, Ran and Sun, Zhenan and Tan, Tieniu}
	  journal={arXiv preprint arXiv:1511.02683},
	  year={2015}
	}
	@article{wu2015lightened,
	  title={A Lightened CNN for Deep Face Representation},
	  author={Wu, Xiang and He, Ran and Sun, Zhenan},
	  journal={arXiv preprint arXiv:1511.02683},
	  year={2015}
	}
	@article{wu2015learning,
	  title={Learning Robust Deep Face Representation},
	  author={Wu, Xiang},
	  journal={arXiv preprint arXiv:1507.04844},
	  year={2015}
	}# Face_Verification_based_on_LightCNN_with_centerloss
# Face_Verification_based_on_LightCNN_with_centerloss
