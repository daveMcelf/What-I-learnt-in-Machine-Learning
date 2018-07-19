# Background
 This repository is made for the purpose of sharing the machine learning related papers that I have read during my researching period on this field. Since Machine learning is a tremendously broad topic, I only limited my focus on some branches including: Image Detection and Segmentation, Image Captioning, Image Generation...etc. 
 
# ML Related Papers
### Image Detection and Segmentation
- [Yolo](https://arxiv.org/abs/1506.02640) : A image detection framework that focus on speed. It achieve a real-time frame rate (45fps) on detection task.
- [Yolo V2](https://arxiv.org/abs/1612.08242) : An improve version of the original version of Yolo, which make an improvement on both the speed and accuracy.
- [Yolo V3](https://pjreddie.com/media/files/papers/YOLOv3.pdf) : Yet another improvement of Yolo. A trade-off for improvement of accuracy and speed. The speed is still real-time though.
- [Mask-RCNN](https://github.com/matterport/Mask_RCNN) : maybe detect by putting box and the image is not enough(in my humble opinion :p), putting a segmentation mask on the images is the way to go.
- [Faster-RCNN](https://arxiv.org/abs/1506.01497) : A object detection which has a state-of-the-art result in accuracy, that can be able to work in real-time(5fps)
- [Resnet](https://arxiv.org/pdf/1512.03385.pdf) : Who said deeper network is bad? This architecture is used a really deep network(up to 152 layers) and be able to not be complex. It introduce the skip connection and shortcut to other architecture including the Yolov3 and Faster-RCNN.

### Image/Video Captioning
- [DenseCap](https://arxiv.org/abs/1511.07571) : an end-to-end architecture that dense image captioning task using  a propose Fully Convolutional Localization Network.
- [Captioning Transformer with Stacked Attention Modules](www.mdpi.com/2076-3417/8/5/739/pdf) : captioning image with Transformer modules.
- [Show, Attend, Tell](https://arxiv.org/pdf/1502.03044.pdf) : Image Captioning with attention module.
- [Image Captioning with Semantic Attention](https://ieeexplore.ieee.org/document/7780872/) : Similar to the ShowTell, but combine the top-down and bottom-up feature for better image captioning task.
- [Dense Captioning Events in Videos](https://arxiv.org/pdf/1705.00754.pdf) : As we can see, Dense Captioning for events in Video.... lol.

### Attention Mechanism/ RNN / LSTM
- [Attention is all you need (Transfomer Model)](https://arxiv.org/pdf/1706.03762.pdf) : probably the best research paper in 2017 (again, in my humble opinion), which use stack attention module to replace the normal RNN/LSTM on sequence-to-sequence model.
- [Areas of Attention for Image Captioning](https://arxiv.org/pdf/1612.01033.pdf) : compare and propose three methods for image captioning using: activation grid, object proposal, and spatial transformer. The model with spatial tranformer give the best performances.

### Image Generation
- [Generative Adversarial Network](https://arxiv.org/pdf/1406.2661.pdf): Train two network adversarially to compete each other so that it can later fool human. It has many applications including image generation, super-resolution, style-transfering etc.
- [DeLiGAN](https://arxiv.org/pdf/1706.02071.pdf) : Combine GAN with VAE to allow GAN work in small and diverse dataset.

### Other
- [Unsupervised Action Discovery](http://openaccess.thecvf.com/content_ICCV_2017/papers/Soomro_Unsupervised_Action_Discovery_ICCV_2017_paper.pdf) : Reading.....

### Blogs & Videos
- [2017 Trend in Machine Learning](https://medium.com/@karpathy/a-peek-at-trends-in-machine-learning-ab8a1085a106) by Andrej Karpathy
- [Good Inspiration on why you should learn RNN](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) by Andrej Karpathy
- [How OpenAI beat pro Dota2 gamer 1vs1](https://blog.openai.com/dota-2/) by OpenAI
- [How OpenAI beat human 5vs5 in Dota2 game](https://blog.openai.com/openai-five/) by OpenAI
- [Implement Your own Darknet from Scratch with PyTorch](https://blog.paperspace.com/tag/series-yolo/) by Ayoosh Kathuria
- [Transformer Model illustrated](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar
### Git Good Project
- [Implementaion of Mask-RCNN](https://github.com/matterport/Mask_RCNN)
- [Implementation of Darknet by PJReddie](https://github.com/pjreddie/darknet)
- [Implementation of Darknet by AlexyAB](https://github.com/AlexeyAB/darknet)
- [Python Implementation of Faster-RCNN](https://github.com/rbgirshick/py-faster-rcnn)
- [Torch Implementation of DenseCap](https://github.com/jcjohnson/densecap)
- [Line-by-Line Attention Transformer implementation](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [PyTorch Tutorial](https://github.com/yunjey/pytorch-tutorial)
- [A collection of papers and project of Image Captioning](https://github.com/handong1587/handong1587.github.io/blob/master/_posts/deep_learning/2015-10-09-captioning.md)
- [PyTorch Github](https://github.com/pytorch/pytorch)

### Statistical Terms (maybe for people like me C:)
- False Positive: falsely detecting non/background object as object
- False Negative: falsely/fail detecting object even the object exist
- True Positive Rate (TPR): the ratio between all True Positive out of Positives
- False Positive Rate (FPR): the ratio between all False Positive out of Negatives
- Detection Error Tradeoff (DET): Graphical plot for error rate in binary classification between FPR and Miss Detection rate(FNR)
- Receiver Operation Characteristic curve (ROC): Graphical plot of error rate in binary classification between TPR and FPR
- Recall:  the ratio of correctly predicted true positive over the total of positive, can be used as True Positive Rate. Recall = TP/P
- Precision: the ratio between true positive over all positive prediction(TP and FP), Precision= TP/(TP+FP)
- F1 score: the mean of all the recall and precision
- Average Precision: scalar way to evaluate the performance of classifier and is the area under the prediction-recall curves
- Jaccard index = Intersection over Union(IoU): ratio between area of overlap over area of union (area of the intersection divided by the area of the union of the two rectangular bounding boxes (ground truth and prediction))
- Cross validation: model validation technique to assess how the results of a statistical analysis will generalize to an independent data set
