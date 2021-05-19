# Local-Style-Curriculum-Learning

Style Curriculum Learning for Robust Medical Image Segmentation



![image-20210519165904198](http://m.qpic.cn/psc?/V12kySKV4IhBFe/45NBuzDIW489QBoVep5mca00S7NavADVMtrp3ZfazfBsU0sF9ncSMnm5.hKHwnFTyiMNi3xtgHv6i9weYPngqsGfpDYKmZhuN9davucmHPM!/b&bo=jwNfAgAAAAADN8M!&rf=viewer_4)



## Dataset

[Multi-Centre, Multi-Vendor & Multi-Disease Cardiac Image Segmentation Challenge (M&Ms)](https://www.ub.edu/mnms/)

The challenge cohort was composed of 375 patients with hypertrophic and dilated cardiomyopathies as well as healthy subjects. All subjects were scanned in clinical centres in three different countries (Spain, Germany and Canada) using four different magnetic resonance scanner vendors (Siemens, General Electric, Philips and Canon).

**Training set (150+25 studies)** The training set contained 150 annotated images from two different MRI vendors (75 each) and 25 unannotated images from a third vendor. The CMR images have been segmented by experienced clinicians from the respective institutions, including contours for the left (LV) and right ventricle (RV) blood pools, as well as for the left ventricular myocardium (MYO). Labels are: 1 (LV), 2 (MYO) and 3 (RV).

**Testing set (200 studies)** The 200 test cases corresponded to 50 new studies from each of the vendors provided in the training set and 50 additional studies from a fourth unseen vendor, that were tested for model generalizability. 20% of these datasets were used for validation and the rest were reserved for testing and ranking participants.

![image-20210519165622003](http://m.qpic.cn/psc?/V12kySKV4IhBFe/45NBuzDIW489QBoVep5mca00S7NavADVMtrp3ZfazfBk5aII2IMf9KHlyqzxCl7ApD2S5jfzKi8ENIzGKkAIwTDzpvUseGlXVOqRZcROlCU!/b&bo=qAPUAQAAAAADJ3w!&rf=viewer_4)



## Method

![image-20210519170937289](http://m.qpic.cn/psc?/V12kySKV4IhBFe/45NBuzDIW489QBoVep5mca00S7NavADVMtrp3ZfazfAcPmunzFxyzozSnUvP0pfkpZiVXwlCfcHk7Ryr6dBB2sJIu5PIDC37EQyFRtoYw9o!/b&bo=mAPcAQAAAAADJ0Q!&rf=viewer_4)



## Getting Started


### Dependency

- PyTorch >= 0.4.1
- Check the requirements.txt

```bash
pip install -r requirements.txt
```



### Pretrained models

- Pretrained models can be found in the `./Ablation_pretrained_model`

![image-20210519180231471](http://m.qpic.cn/psc?/V12kySKV4IhBFe/45NBuzDIW489QBoVep5mcW1w8w5easkTqE0vkKR424eLmnrapIARumcPvNOR1rzYYNx1QHFzAmMxoa.6qVpHcPryDXL1HEjEfjJCgohzipI!/b&bo=DwOtAAAAAAADF5M!&rf=viewer_4)



## Results

We compared the results with the top three teams (P1,P2,P3) in the M&Ms challenge.(#R is ranking)

We adopted the same evaluation criteria and ranking method as the M&Ms challenge performs and standardizes (https://www.ub.edu/mnms/).

| Method | weighted_dice | #R   | weighted_hdb | #R   | training_time | #R   | test_time |  #R  |
| :----: | :-----------: | ---- | :----------: | ---- | :-----------: | ---- | :-------: | :--: |
|  Ours  |    0.8710     | 3    |  **10.602**  | 1    |    **5h**     | 1    | **0.2s**  |  1   |
|   P1   |  **0.8813**   | 1    |    11.111    | 2    |      60h      | 3    |    1s     |  2   |
|   P2   |    0.8750     | 2    |    11.370    | 3    |      48h      | 2    |   4.8s    |  3   |
|   P3   |    0.8705     | 4    |    11.887    | 4    |    4-5days    | 4    |    N/A    |  4   |
|  Unet  |    0.8345     | 5    |    16.674    | 5    |       /       | /    |     /     |  /   |

