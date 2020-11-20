## 参考资料
1.[天池&Datawhale 新闻文本分类比赛 第2名参赛经验、源码分享](https://blog.csdn.net/lz123snow/article/details/108508189)
2.[新闻文本分类思路分享-rank10](https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.84.6406111acg3qXw&postId=132270)
3.[rank6_NLP_newstextclassification](https://github.com/Warrenheww/rank6_NLP_newstextclassification?spm=5176.12282029.0.0.7d0d19a2MnMN2I)
4.[正式赛_Rank8_主要思路分享](https://tianchi.aliyun.com/forum/postDetail?spm=5176.12586969.1002.93.6406111acg3qXw&postId=131980)


## 模型

- xlnet  0.982250
- electra-base-discriminator 0.97+
- chinese-roberta-wwm-ext 0.984450
- albert_chinese_base 0.96
- longformer-chinese-base 0.9825
- hfl/chinese-electra-base-generator 0.93

融合：
- xlnet+electra-base-discriminator+chinese-roberta-wwm-ext+albert_chinese_base：0.986375
- xlnet+electra-base-discriminator+chinese-roberta-wwm-ext+albert_chinese_base+longformer-chinese-base：0.98575
- xlnet+electra-base-discriminator+chinese-roberta-wwm-ext+albert_chinese_base+longformer-chinese-base+hfl/chinese-electra-base-generator：0.9855
