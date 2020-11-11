import cv2
import numpy as np


def decorate(f):
    def fun(image, bboxes, **kwargs):
        h, w, _ = image.shape
        try:
            ds_image, ds_bboxes = f(image.copy(), np.array(bboxes).copy(), height=h, width=w, **kwargs)
        except:
            return None, []
        if len(ds_bboxes) < 1:
            return None, []
        ds_bboxes[:, 2] = np.clip(ds_bboxes[:, 2], 0, w - 1).astype(np.int)
        ds_bboxes[:, 3] = np.clip(ds_bboxes[:, 3], 0, h - 1).astype(np.int)
        mask = (ds_bboxes[:, 2] > ds_bboxes[:, 0]) * (ds_bboxes[:, 3] > ds_bboxes[:, 1])
        # mask *= (ds_bboxes[:, 2] < w) * (ds_bboxes[:, 3] < h)
        if mask.sum() == 0:
            return None, []
        else:
            return ds_image, ds_bboxes[mask]

    return fun


@decorate
def Flip(image, bboxes, direction='horizon', **kwargs):
    """
    flip image and its bounding-boxes with 3 directions: horizon, vertical, horizon_vertical

    :param image: loaded image from cv2.imread()
    :param bboxes: context like [[x1, y1, x2, y2, label], ...]
    :param direction: assert in [horizon, vertical, horizon_vertical]
    :return: fliped image and its bboxes
    """

    h, w = kwargs['height'], kwargs['width']
    if direction == 'horizon':
        bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]
        dst = cv2.flip(image, 1)
    elif direction == 'vertical':
        bboxes[:, [1, 3]] = h - bboxes[:, [3, 1]]
        dst = cv2.flip(image, 0)
    elif direction == 'horizon_vertical':
        bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]
        bboxes[:, [1, 3]] = h - bboxes[:, [3, 1]]
        dst = cv2.flip(image, -1)
    else:
        raise ValueError(
            "('direction' must in [horizon, vertical, horizon_vertical])\tINCORRECT 'direction': {}".format(direction))

    return dst, bboxes


@decorate
def RandomCropByBoxes(image, bboxes, crop_size, thresh_ratio=0.4, **kwargs):
    h, w = kwargs['height'], kwargs['width']
    if isinstance(crop_size, int):
        if crop_size >= h or crop_size >= w:
            ratio = crop_size * 1.6 / min(w, h)
            image = cv2.resize(image, (0, 0), fx=ratio, fy=ratio)
            bboxes[:, :4] *= ratio
            h, w, _ = image.shape
        dsize = [crop_size, crop_size]
    elif isinstance(crop_size, (tuple, list)):
        if len(crop_size) == 2:
            if max(crop_size) >= min(w, h):
                ratio = max(crop_size) * 1.6 / min(w, h)
                image = cv2.resize(image, (0, 0), fx=ratio, fy=ratio)
                bboxes[:, :4] = ratio * bboxes[:, :4]
                h, w, _ = image.shape
            dsize = list(crop_size)
        else:
            raise ValueError('Expect format is (w, h), but input shape: {}'.format(crop_size))
    else:
        raise ValueError('Expect type is int, list, tuple, but input crop_size type: {}'.format(type(crop_size)))

    if len(bboxes) == 1:
        center_box = bboxes[0]
    elif len(bboxes) > 1:
        center_box = bboxes[np.random.random_integers(0, len(bboxes) - 1, [1])[0]]
    else:
        raise ValueError('EMPTY BBOXES: {}'.format(bboxes))

    w_cbox, h_cbox = center_box[2] - center_box[0], center_box[3] - center_box[1]

    if w_cbox >= dsize[0]:
        # print('[WARNING] BAD CROP WIDTH: %d.' % dsize[0])
        dsize[0] = min(int(w_cbox * 1.5), w)

    if h_cbox >= dsize[1]:
        # print('[WARNING] BAD CROP HEIGHT: %d.' % h)
        dsize[1] = min(int(h_cbox * 1.5), h)

    if (center_box[2] - dsize[0] >= center_box[0]) or (center_box[3] - dsize[1] >= center_box[1]):
        return None, []
    if (center_box[0] <= 0) or (center_box[1] <= 0):
        return None, []
    start_x = int(np.random.randint(center_box[2] - dsize[0], center_box[0], [1]).clip(0,  center_box[0])[0])
    start_y = int(np.random.randint(center_box[3] - dsize[1], center_box[1], [1]).clip(0,  center_box[1])[0])
    end_x = int(np.clip([start_x + dsize[0]], center_box[2], w)[0])
    end_y = int(np.clip([start_y + dsize[1]], center_box[3], h)[0])

    image = image[start_y: end_y, start_x: end_x, :]

    nboxes = bboxes.copy()
    nboxes[:, [0, 2]] = nboxes[:, [0, 2]].clip(start_x, end_x) - start_x
    nboxes[:, [1, 3]] = nboxes[:, [1, 3]].clip(start_y, end_y) - start_y
    mask = (nboxes[:, [2, 3]] - nboxes[:, [0, 1]]) / (bboxes[:, [2, 3]] - bboxes[:, [0, 1]])
    area = np.prod(mask, axis=1)
    mask = area > thresh_ratio
    nboxes = nboxes[mask]
    return image, nboxes.astype(np.int)


@decorate
def RandomContrast(image, bboxes, low=0.5, high=1.5, **kwargs):
    alpha = np.random.uniform(low, high)
    im = image.astype(np.float)
    im *= alpha
    return im.clip(min=0, max=255).astype(np.uint8), bboxes


@decorate
def RandomSaturation(image, bboxes, low=0.5, high=1.5, **kwargs):
    im = image.astype(np.float)
    im[:, :, 1] *= np.random.uniform(low, high)
    return im.astype(np.uint8), bboxes


@decorate
def RandomHue(image, bboxes, delta=18.0, **kwargs):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float)
    image[:, :, 0] += np.random.uniform(-delta, delta)
    image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
    image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
    return image.astype(np.uint8), bboxes


@decorate
def RandomSwap(image, bboxes, **kwargs):
    channels = [0, 1, 2]
    np.random.shuffle(channels)
    image[:, :, [0, 1, 2]] = image[:, :, channels]
    return image, bboxes


@decorate
def HistEqualize(image, bboxes, **kwargs):
    channels = np.split(image, image.shape[-1], axis=-1)
    equalimage = np.concatenate([np.expand_dims(cv2.equalizeHist(ch), axis=-1) for ch in channels], axis=-1)
    return equalimage, bboxes


@decorate
def GrayScale(image, bboxes, **kwargs):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.repeat(np.expand_dims(gray, axis=-1), 3, axis=-1), bboxes


@decorate
def ImagePyramid(image, bboxes, num_levels=4, **kwargs):
    h, w = kwargs['height'], kwargs['width']
    bboxes = np.array(bboxes)
    for i in range(-num_levels, 0):
        alpha = 2 ** i
        dsize = (int(w * alpha), int(h * alpha))
        bbox = bboxes.copy()
        bbox = bbox[:, :4] * alpha
        yield cv2.resize(image, dsize, interpolation=cv2.INTER_CUBIC), bbox.astype(np.int)


@decorate
def Rotation(image, bboxes, angle=0, scale=1, thresh_ratio=0.4, **kwargs):
    boxes = bboxes[:, :4]
    labels = bboxes[:, -1]
    h, w = kwargs['height'], kwargs['width']
    thresh_w = np.min(bboxes[:, 2] - bboxes[:, 0]) * thresh_ratio
    thresh_h = np.min(bboxes[:, 3] - bboxes[:, 1]) * thresh_ratio
    transM = np.zeros((4, 5))
    Mat = cv2.getRotationMatrix2D((0.5 * (w - 1), 0.5 * (h - 1)), angle=angle, scale=scale)
    dst = cv2.warpAffine(image, Mat, (0, 0))
    transM[:2, :3] = Mat
    transM[-2:, -2:] = Mat[:2, :2]
    transM[-2:, 2] = Mat[:, 2]
    coord13 = np.hstack((boxes[:, :2], np.ones((boxes.shape[0], 1)), boxes[:, -2:]))
    coord24 = np.hstack((boxes[:, [0, 3]], np.ones((boxes.shape[0], 1)), boxes[:, [2, 1]]))
    boxes13 = np.dot(transM, coord13.transpose((1, 0))).transpose((1, 0)).astype(np.int)
    boxes24 = np.dot(transM, coord24.transpose((1, 0))).transpose((1, 0)).astype(np.int)
    boxes = np.hstack((boxes13, boxes24))
    xs = boxes[:, [0, 2, 4, 6]]
    ys = boxes[:, [1, 3, 5, 7]]

    boxes = np.vstack((
        np.clip(np.min(xs, axis=-1), 0, w),
        np.clip(np.min(ys, axis=-1), 0, h),
        np.clip(np.max(xs, axis=-1), 0, w),
        np.clip(np.max(ys, axis=-1), 0, h),
        labels
    )).transpose((1, 0))

    boxes = boxes[boxes[:, 3] - boxes[:, 1] > thresh_h]
    boxes = boxes[boxes[:, 2] - boxes[:, 0] > thresh_w]

    return dst, boxes


@decorate
def FilterTransform(image, bboxes, filter='', **kwargs):
    def gauss(**kwargs):
        image = kwargs['image']
        ksize = kwargs['ksize']
        ds = cv2.GaussianBlur(image, ksize, sigmaX=0)
        return ds
    def avg(**kwargs):
        image = kwargs['image']
        ksize = kwargs['ksize']
        ds = cv2.blur(image, ksize)
        return ds
    def median(**kwargs):
        image = kwargs['image']
        ksize = kwargs['ksize']
        ds = cv2.medianBlur(image, ksize)
        return ds
    def bilateral(**kwargs):
        image = kwargs['image']
        d = kwargs['d']
        sigmaColor = kwargs['sigmaColor']
        sigmaSpace = kwargs['sigmaSpace']
        ds = cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)
        return ds

    def kernelfilter(**kwargs):
        image = kwargs['image']
        kernel = kwargs['kernel']
        ds = cv2.filter2D(image, -1, kernel)
        return ds

    fun = {
        'gauss': gauss,
        'average': avg,
        'median': median,
        'bilateral': bilateral,
        'customized': kernelfilter,
    }

    return fun[filter](image=image, **kwargs), bboxes


@decorate
def NoiseAdd(image, bboxes, noise='salt', **kwargs):
    def salt(image, **kwargs):
        ratio = kwargs['salt_ratio']
        assert (ratio > 0) and (ratio < 1)
        h, w = kwargs['height'], kwargs['width']
        mask = np.zeros(h * w, dtype=np.bool)
        num = int(ratio * mask.shape[0])
        mask[:num] = True
        np.random.shuffle(mask)
        mask = mask.reshape([h, w])
        noise = np.random.randint(0, 255, [num, 3], dtype=np.uint8)
        image[mask] = noise
        return image

    def gauss(image, **kwargs):
        scale = kwargs['scale']
        # h, w, _ = image.shape
        noise = np.random.normal(0, scale, image.shape).astype(np.uint8)
        image += noise
        return np.clip(image, 0, 255)

    fun = {'salt': salt, 'gauss': gauss}
    return fun[noise](image, **kwargs), bboxes


@decorate
def PerspectiveTransform(image, bboxes, direction, scale=0.8, **kwargs):
    # assert direction in ['left', 'right', 'top', 'bottom']
    boxes = bboxes[:, :4]
    h, w = kwargs['height'], kwargs['width']
    hscale = (1 - scale) * 0.5
    if direction == 'left':
        pt1 = [[0, int(hscale * h)], [0, int(h - hscale * h) - 1], [w - 1, h - 1], [w - 1, 0]]
    elif direction == 'right':
        pt1 = [[0, 0], [0, h - 1], [w - 1, int(h - hscale * h) - 1], [w - 1, int(hscale * h)]]
    elif direction == 'top':
        pt1 = [[int(hscale * w), 0], [0, h - 1], [w - 1, h - 1], [int(w - hscale * w) - 1, 0]]
    elif direction == 'bottom':
        pt1 = [[0, 0], [int(hscale * w), h - 1], [int(w - hscale * w) - 1, h - 1], [w - 1, 0]]
    else:
        raise ValueError('Undefined direction: {}'.format(direction))

    pt0 = np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]], dtype=np.float32)

    Matrix = cv2.getPerspectiveTransform(pt0, np.array(pt1, dtype=np.float32))

    cbox = np.array([boxes[:, idist] for idist in [[0, 1], [2, 3], [0, 3], [2, 1]]])
    ones = np.ones((4, len(boxes), 1))
    cbox = np.concatenate((cbox, ones), axis=-1)

    dst = cv2.warpPerspective(image, Matrix, (0, 0))
    # 11, 22; 12, 21
    boxes_mat = np.dot(cbox, Matrix.transpose((1, 0)))
    boxes_mat = boxes_mat[..., :2] / boxes_mat[..., [-1, -1]]
    result = np.vstack((
        np.min(boxes_mat[[0, 2], :, 0], axis=0), np.min(boxes_mat[[0, 3], :, 1], axis=0),
        np.max(boxes_mat[[1, 3], :, 0], axis=0), np.max(boxes_mat[[1, 2], :, 1], axis=0),
        bboxes[:, -1]
    )).transpose((1, 0))

    return dst, result.astype(np.int)


def ImagesFusion(f_image, bg_image, bboxes=None, dscale=0.3):
    fg_image = f_image.copy()
    fh, fw, _ = fg_image.shape
    bh, bw, _ = bg_image.shape
    fscale = min(float(bh) / fh, float(bw) / fw)
    fscale = fscale * np.random.uniform(dscale, 0.96, [1])[0]
    fg_image = cv2.resize(fg_image, dsize=(0, 0), fx=fscale, fy=fscale)
    fh, fw, _ = fg_image.shape
    start_y = np.random.random_integers(0, bh - fh - 1, [1])[0]
    start_x = np.random.random_integers(0, bw - fw - 1, [1])[0]
    bg_image[start_y:start_y + fh, start_x:start_x + fw] = fg_image
    bboxes = np.array(bboxes)
    boxes = bboxes[:, :-1] * fscale
    boxes[:, [0, 2]] += start_x
    boxes[:, [1, 3]] += start_y
    result = np.hstack((boxes, np.expand_dims(bboxes[:, -1], axis=-1)))
    return bg_image, result.astype(np.int)


