# ImageRetrieval_PTLCH_ESWA2024

# Using the code should cite the following paper:

[1] Ren H, Guo J, Cheng S, et al. Pooling-based Visual Transformer with low complexity attention hashing for image retrieval[J]. Expert Systems with Applications, 2024, 241: 122745. (IF=8.5).(https://github.com/cslxju/ImageRetrieval_PTLCH_ESWA2024)

# Description:
Retrieving similar images is becoming an urgent need for us with the continuous growth of large-scale data. However, whether the dominant image retrieval methods are Convolutional Neural Networks (CNNs) or the recently emerging Visual Transformer (ViT), their complex computation, insufficient feature extraction, and mismatched weights greatly influence the efficiency and retrieval accuracy. In this paper, we propose a Pooling-based Visual Transformer with low complexity attention hashing (PTLCH) for image retrieval. First, a backbone network for Pooling-based Vision Transformer (PiT) feature learning is designed to combine the pooling in CNN and the ViT to achieve the purpose of spatial dimensionality reduction while learning rich semantic information. Second, a low complexity attention (LCA) module is incorporated into PiT, which works by combining the positional deviation with the key matrix and the value matrix and then matrix multiplying with the query matrix. LCA explores rich contextual information to enable network learning of more granular feature information. Finally, a new loss framework is proposed where we focus on the effect of difficult and erroneous samples on accuracy. By using different improved cross-entropy losses, better weights are assigned to the learning samples of our network, which effectively improves learning hash coding. We have conducted extensive experiments on three public datasets, CIFAR-10, ImageNet100, and MS-COCO, which have the highest mean average precision of 93.76%, 92.62%, and 90.60%, respectively.


