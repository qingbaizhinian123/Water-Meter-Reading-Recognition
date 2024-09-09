***Water Meter Reading Recognition Method Based on Character Attention Mechanism***

Our process is mainly divided into three stages, which are the detection stage of the water meter reading area, the location stage of the character, and the recognition stage of the character.Among them, the positioning phase of characters is mainly referred to the article CRAFT.
CRAFT: Character-Region Awareness For Text detection | [Paper](https://arxiv.org/abs/1904.01941) | [Pretrained Model](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ) | [Supplementary](https://youtu.be/HI8MzpY8KMI)

First, you can run file (det.py) to get the detection area of the water meter reading, then, you can run file (test.py) to get the location information of a single character, combine these two parts to get the character area, and finally, you can pass the character through file (rec.py) to get the character recognition result.

##Requirement
CUDA 11.6
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch
torch==0.4.1.post2
torchvision==0.2.1
opencv-python==3.4.2.17
scikit-image==0.14.2
scipy==1.1.0

