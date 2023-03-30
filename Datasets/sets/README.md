# In-bed Multi-modal Human Dataset
This dataset recorded in-bed human subjects from multiple modality sensors including RGB, IR for pose estimation purpose. 
The data is collected in both a home setting and hospital setting environment named 'danaLab' and 'simLab' with 102 ги73 male, 28 femaile) and 7 subjects (4 male 3 female) respectively. 
For each setting, subjects are requested to give 15 poses as wish in 3 general categories as supine, left lying and right lying. For each pose,data is collected from 3 cover conditions as uncover, cover1 and cover2. 

This release is for ICIP VIPCup 2021.
Train, validation and test set will be released as individual zip files. 
Training and validation set are both from home setting. 
Training:  
subj 1 - 30, nocover,  annotated 
subj 31 - 50,  cover1, unanotated 
subj 51 - 70, cover2, unanotated 
Valdiation:
subj 71-75, cover1, annotated 
subj 76-80, cover2, annotated 
where cover1 for thin, cover2 for thick.  
Both LWIR and RGB will be provided. LWIR is normalized within temperature range 20-40C. 

Test include 2 phases in the challege with mixture of home and hospital settings. 


## File structure
dataset structure 
```
[SLP_VIPCup]
|--	train
|	|-- subjNumber[00001]
|	|-- ...
|-- valid
|	|-- ...
|-- test1 # home + hospital mixed 
|	|-- ...
|-- test2 # home + hospital mixed 
|	|-- ...
|-- README.md
```


Each subject is named with a numbered folder, with structure
```
[subjNumber]
|--	IR
|	|-- uncover
|	|-- cover1
|	|-- cover2
|-- RGB 
	...
|	|-- joints_gt_IR.mat
|	|-- joints_gt_RGB.mat
|	|-- align_PTr_IR.npy
|	|-- align_PTr_RGB.npy

```
 

## Pose Definition
We follow the Leeds Sports Pose Dataset pose definition with 14 joints labeling, namely, 

`Right ankle
Right knee
Right hip
Left hip
Left knee
Left ankle
Right wrist
Right elbow
Right shoulder
Left shoulder
Left elbow
Left wrist
Thorax
Head top`

## Anotations 
`joints_gt_<modality>.mat` is the pose label file.  `RGB` and `IR` are manually labeled with physical paramter tuning strategy. Please refer our MICCAI'19 paper for details. Pose data from other modalities can be generated via the alignment homography provided. Possible bias could be introduced via the mapping process so we don't officially release them as 'unbiase' ground truth but you can easily generate them for your study as they are still semantically correct. 

The lable matrix `joints_gt_<modality>.mat` has the format  `<x,y,if_occluded>` x n_joints x n_subjects. Original label is 1 based coordinate, please transfer to 0 based for server evaluation by `x=x-1, y=y-1`. 


## Domain Alignment 
At beginning of each experiment session, we aign the all modalities with several weighted icons. The images of icons images are captured in all modalities and provided in raw form in this dataset in case you would like to have a customize re-calibration. They are given in name of `align_<mod>001.[png|npy]` depending on if that is image or deeper raw data. 

`align_PTr_<modality>.npy` saved the homography transformation matrix to reference frame which is PM in our case. To get transformation between other modalities for example RGB and IR, we can simply use `inv(H_RGB) * H_IR`. 


## Reference
@article{liu20120simultaneously, title={Simultaneously-Collected Multimodal Lying Pose Dataset: Towards In-Bed Human Pose Monitoring under Adverse Vision Conditions}, author={Liu, Shuangjun and Huang, Xiaofei and Fu, Nihang and Li, Cheng and Su, Zhongnan and Ostadabbas, Sarah}, journal={arXiv preprint arXiv:2008.08735}, year={2020} }

@inproceedings{liu2019seeing,
  title={Seeing under the cover: A physics guided learning approach for in-bed pose estimation},
  author={Liu, Shuangjun and Ostadabbas, Sarah},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={236--245},
  year={2019},
  organization={Springer}
}

## Contact
Shuanjun Liu,
email: shuliu@ece.neu.edu

Sarah Ostadabbas, 
email: ostadabbas@ece.neu.edu

