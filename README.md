# GLOMERULI DETECTION​

**MACHINE LEARNING IN APPLICATION**

*06th October 2023*

Identifying glomeruli, which involves detection and classification into normal and sclerosed glomeruli, is an essential
step in various nephropathology studies such as those involving DM1 (type 1 Diabetes Mellitus) and DM2 (type 2 Diabetes
Mellitus). However, manually counting glomeruli can be tedious and a time-consuming task. That’s why it’s necessary to use
image processing tools that can accurately detect and classify glomeruli. In this paper they have been identified through two
semantic segmentation architectures: SegNet and U-Net.
On the other hand, since there wasn’t available ground truth for glomeruli classification, an unsupervised method was required
to determine distinct levels of glomeruli necrotization. In particular, to reach this goal a K-means method was performed
after an initial feature extraction and dimensionality reduction phase. These methods were tested on a dataset composed of 9
WSIs belonging to human kidney sections. The best approach of semantic segmentation was SegNet which returned slightly
better results than U-Net. Regarding clustering, K-means did not provide a comfortable classification, probably due to the complex
nature of our initial images. Therefore, in presence of labels, it might be better to approach this problem with a supervised
technique rather than an unsupervised method.

The digital tissue images used in this study are WSIs (Whole Slide Images) of renal biopsies. The dataset is composed
of 9 different WSIs of human kidney tissue provided with annotations which show glomeruli.

**Check *Report.pdf* for the full analysis.**
