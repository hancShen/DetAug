import threading
from .utils import *
from multiprocessing import Pool, Manager, Process


class AugThreding(threading.Thread):
    def __init__(self, fun, args_dict, **kwargs):
        super(AugThreding, self).__init__()

        self.fun = fun
        if kwargs:
            self.args = args_dict.update(kwargs)
        else:
            self.args = args_dict
        self._result = None
        self.setDaemon(True)

    def run(self):
        if isinstance(self.args, dict):
            self._result = self.fun(**self.args)
        else:
            self._result = self.fun(*self.args)

    def result(self):
        return self._result


FuncMap = {
    'rotate': Rotation,
    'flip': Flip,
    'random_crop': RandomCropByBoxes,
    'random_hue': RandomHue,
    'random_swap': RandomSwap,
    'random_contrast': RandomContrast,
    'random_saturation': RandomSaturation,
    'gray_scale': GrayScale,
    'filter_transform': FilterTransform,
    'add_noise': NoiseAdd,
    'fusion': ImagesFusion,
    'perspective_transform': PerspectiveTransform,
    'hist_equalize': HistEqualize,
    }


def GroupAug(image, bboxes, args_dic):
    general_args = {'image': image.copy(), 'bboxes': bboxes}
    for f, arg_list in args_dic.items():
        func = FuncMap[f]
        for arg in arg_list:
            garg = general_args.copy()
            inargs = {**garg, **arg}
            img, ann = func(**inargs)
            yield img, ann
