<h1 align=center>Data Augmentation for Object Detection</h1>

<div align="center">hcshen@gmail.com</div>

&emsp; Data Augmentation(DA) is a essential step in deep learning model training. Objcet Detection(OD), one of most important task in computer vision, is highly demand on annotation dataset.This repository propose an effective tool package for OD tasks, we named as `detaug`. 
 

<center>
<img src="./images/image_000041_0.jpg" width=304 height=400/>
<p>image 0: the image without augmentation.</p>
</center>

## 1. Usage

&emsp; All of OD augmentation tool functions were packaged into **detaug**, you can use it by<br>
```bash
python setup.py bdist_wheel

pip install ./dist/detaug-xxx-py3-none-any.whl
```
in your envionment, and the `*.whl` was stored in folder path `./dist/`.

After `detaug` was installed, you could use it by `import detaug`.

## 2. Methods of `detaug`

|SN|name|function|
|:---:|:---:|:---:|
|1|rotate | Rotation|
|2|flip | Flip|
|3|random_crop | RandomCrop|
|4|random_hue | RandomHue|
|5|random_swap | RandomSwap|
|6|random_contrast | RandomContrast|
|7|random_saturation | RandomSaturation|
|8|gray_scale | GrayScale|
|9|filter_transform | FilterTransform|
|10|add_noise | NoiseAdd|
|11|fusion | ImagesFusion|
|12|perspective_transform | PerspectiveTransform|
|13|hist_equalize | HistEqualize|

#### 2.1 rotation

```python
input = cv2.imread(imgpath)
result, tbboxes = detaug.Rotation(input, bboxes, angle=0, scale=1.0)

# format of bboxes is [[x1, y1, x2, y2, label], ...]
# angle: value in [0, 360], default 0 means without rotation
# scale: default 1, if rescale the output image, change it.
```
---

#### 2.2. flip #

```python
input = cv2.imread(imgpath)
result, tbboxes = detaug.Flip(image, bboxes, direction='horizon')

# direction includes: horizon, vertical, horizon_vertical
# Flip is equal to rotation with angle 90, 180, 270 degree.
```
---
<center>
<img src="./images/image_000041_1.jpg" width=304 height=400>
<img src="./images/image_000041_2.jpg" width=304 height=400>
</center>
<center>
<p>image 1: rotation &thinsp; &thinsp;
image 2. flip</p>
</center>
#### 2.3. random crop #

```python
input = cv2.imread(imgpath)
result, tbboxes = detaug.RandomCrop(input, bboxes, crop_size)


# crop_size:
# 2 formats:
#         int: output image shape: [crop_size, crop_size]
#         tuple or list: must be with 2 values like (w, h), output image shape: [w, h]

```
---

#### 2.4. perspective transformation

```python

result, tbboxes = detaug.PerspectiveCenterTransform(image, bboxes, direction, scale=0.8)
# direction: including 4 directions: top, bottom, left, right;
# scale: a float value in [0.0, 1.0], the perspetective transform edge scale ratio.
```
---
<center>
<img src="./images/image_000041_3.jpg" width=304 height=400>
<img src="./images/image_000041_12.jpg" width=304 height=400>
</center>
<center>
<p>image 3: random crop &thinsp; &thinsp;
image 4. perspective transformation</p>
</center>

#### 2.5. add noise

```python
result, tbboxes = detaug.NoiseAdd(image, bboxes, noise='salt', salt_ratio=0.3)
# noise: 'salt' or 'gauss';
# salt_ratio: a float value in [0.0, 1.0], the ratio of salt noise in whole image;

#scale: a float value over 0.0, the scale of gauss noise diameter.
result, tbboxes = detaug.NoiseAdd(image, bboxes, noise='gauss', scale=1.0)

```
---

#### 2.6. image filter

```python
input = cv2.imread(imgpath)

# 1. gauss blur
# kisze: a 2dim tuple, the shape of gauss blur kernel
result, tbboxes = detaug.FilterTransform(image, bboxes, filter='gauss', ksize=(5, 5))

# 2. median filter
# ksize: an int, the diameter of median filter kernel
result, tbboxes = detaug.FilterTransform(image, bboxes, filter='median', ksize=5)

# 3. avg filter
# ksize: a 2dim tuple, the shape of average filter kernel
result, tbboxes = detaug.FilterTransform(image, bboxes, filter='average', ksize=(5, 5))

# 4. bilateral
# d: an int, the diameter of filter kernel
# sigmaColor: an int, the incidence range of neighbor color, variance of color distribution
# sigmaSpace: an int, the incidence range of spatial neighbor coordinates, variance of spatial distribution
result, tbboxes = detaug.FilterTransform(image, bboxes, filter='bilateral', d=9, sigmaColor=50, sigmaSpace=50)

# 5. kernel filter, customized filter
# kernel: a numpy array, the customized filter kernel
result, tbboxes = detaug.FilterTransform(image, bboxes, filter='customized', kernel=np.array([[1. 1], [1. 1]]))

```
---

<center>
<img src="./images/image_000041_10.jpg" width=304 height=400>
<img src="./images/image_000041_9.jpg" width=304 height=400>
</center>
<center>
<p>image 5: add noise &thinsp; &thinsp;
image 6. image filter</p>
</center>


#### 2.7. mixup

#### 2.8. contrast # 

Adjust image contrast randomly.

```python
input = cv2.imread(imgpath)
result, tbboxes = detaug.RandomContrast(image, bboxes, low=0.5, high=1.5)
# low, high: rescale image contrast with a random value, and the vaule is random selected in [low, high]

```
---

#### 2.9. gray scale # 

```python
input = cv2.imread(imgpath)
result, tbboxes = detaug.GrayScale(image, bboxes)
# convert BGR cv2 image to gray image and copy 3 equal channels
```
---

<center>
<img src="./images/image_000041_6.jpg" width=304 height=400>
<img src="./images/image_000041_8.jpg" width=304 height=400>
</center>
<center>
<p>image 7: contrast transform &thinsp; &thinsp;
image 8. image gray scale</p>
</center>


#### 2.10. color channels shuffle # 

```python
input = cv2.imread(imgpath)
result, tbboxes = detaug.RandomSwap(image, bboxes)

# random shuffling the order of RGB channels
```
---

#### 2.11. image pyramid # 

```python
results, tbboxes = [], []
for result, tbboxes in ImagePyramid(image, bboxes, num_levels=4):
    result, bboxes = ImagePyramid(image, bboxes, num_levels=4)
    results.append(result)
    tbboxes.append(bboxes)

# num_levels: an int, the number of generating pyramid image levels.

```
---


#### 2.12. hue # 

```python
result, tbboxes = detaug.RandomHue(image, bboxes, delta=18.0)
# delta: a float over 0.0, the fluctuation range of random hue transforming.

```
---


#### 2.13. saturation # 

```python
result, tbboxes = detaug.RandomSaturation(image, bboxes, low=0.5, high=1.5)
# low: a float over 0.0, the lower boundray of added saturation random distribution;
# high: a float over low, the upper boundray of added saturation random distribution;
```
---


<center>
<img src="./images/image_000041_5.jpg" width=304 height=400>
<img src="./images/image_000041_7.jpg" width=304 height=400>
</center>
<center>
<p>image 9: image RGB channels shuffling &thinsp; &thinsp;
image 10. saturation transformation</p>
</center>



#### 2.14. images fusion # 

```python
dimg, dboxes = detaug.ImagesFusion(fg_image, bg_image, bboxes, dscale=0.3)

# merge background image with an annotated foreground image.
# fg_image: the foreground image loaded from cv2.imread;
# bg_image: the backround image loaded from cv2.imread;
# bboxes: the annotation of foreground image;
# dscale: value in [0.0, 1.0], the lower boundray ratio value if foreground image resize while fitting background iamge.
```

<center>
<img src="./images/image_000041_11.jpg" width=304 height=400>
<img src="./images/image_000041_13.jpg" width=304 height=400>
</center>
<center>
<p>image 11: images fusion &thinsp; &thinsp;
image 12. histgram equalization</p>
</center>


### Multi-Thredings

This respository supplied 2 kinds of multi-threding computing:

I. one augment image, one threding;

II. one augment method, one threding;

```python
args_dict = {
    'flip': [
        {'direction': 'vertical'},
        {'direction': 'horizon_vertical'},
        {'direction': 'horizon'}
    ],
    'random_crop': [
        {'crop_size': 608},
        {'crop_size': [512, 608]},
        {'crop_size': (608, 416)}
    ],
 'filter_transform': [
     {'filter': 'gauss', 'ksize': (5, 5)},
     {'filter': 'average', 'ksize': (5, 5)},
     {'filter': 'median', 'ksize': 5}
 ],
 'rotate': [
     {'angle': -5},
     {'angle': 5},
     {'angle': 355},
     {'angle': 185}],
}


# one augmented method, one threding
aug_imgs, aug_anns = detaug.GroupAug(image, bboxes, args_dict)

```

detaug
### Multi-Processes


### Requests:
1. support both multi-process and single process
2. offline and online

### Environment Requirement

```
numpy
opencv-python
Pillow
```


### Version update
- 0.1.1. 

the initial version of OD augmentation package;

- 0.1.2.

update perspective transform, RGB image hist equlization, random crop.

- 0.1.3.

update the method of iteration output return while using detaug.GroupAug.

- 0.1.4.

fixed the bug: function NoiseAdd processed on original image.

- 0.1.5.

fixed the bug: added @decorator to avoid changing original image.

- 0.1.6.

[1]. update the method of bounding-box coordinates mapping in perspective transform, replace 2 points(left-top, right-down) with full 4 corner points;

[2]. update threshold of croped boxes filtering with min bounding-box width or height, adding a ratio threshold fine-tuning.

- 0.1.7.

added limits of bounding-box size in image rotation, based on minimum width or height in origin bounding-boxes, adding a ratio threshold fine-tuning.

- 0.1.8.

update random-crop, solved the problem of abnormal image as output while abnornal crop size.
