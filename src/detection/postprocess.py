
import numpy as np








LP_CHAR_ID_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                   10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                   20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                   30, 31, 32, 33, 34, 35, 36]

OBJ_THRESH = 0.25
NMS_THRESH = 0.45

class Postprocess:

    def __call__(self, input_data, data_branch):
        """
        Using dunder method to execute Postprocess class to post-process input data
        """
        boxes, scores, clss_conf = [], [], []
        pair_per_branch = len(input_data)//data_branch
        for i in range(data_branch):
            boxes.append(self.box_process(input_data[pair_per_branch*i]))
            clss_conf.append(input_data[pair_per_branch*i+1])
            scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))


        boxes = [self.sp_flatten(_v) for _v in boxes]
        clss_conf = [self.sp_flatten(_v) for _v in clss_conf]
        scores = [self.sp_flatten(_v) for _v in scores]

        boxes = np.concatenate(boxes)
        clss_conf = np.concatenate(clss_conf)
        scores = np.concatenate(scores)

        boxes, classes, scores = self.filter_boxes(boxes, scores, clss_conf)

        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]
            keep = self.nms_boxes(b, s)

            if len(keep) != 0:
                nboxes.append(b[keep])
                nclasses.append(c[keep])
                nscores.append(s[keep])

        if not nclasses and not nscores:
            return None, None, None

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)

        return boxes, classes, scores

    def sp_flatten(self, _in):
        ch = _in.shape[1]
        _in = _in.transpose(0,2,3,1)
        return _in.reshape(-1, ch)


    def filter_boxes(self, boxes: np.ndarray, box_conf: np.ndarray, box_cls_probs: np.ndarray):
        """
        Filter boxes with threshold.
        """
        box_conf = box_conf.reshape(-1)
        # candidate, class_num = box_cls_probs.shape

        cls_max_score = np.max(box_cls_probs, axis=-1)
        classes = np.argmax(box_cls_probs, axis=-1)

        _class_pos = np.where(cls_max_score* box_conf >= OBJ_THRESH)
        scores = (cls_max_score* box_conf)[_class_pos]

        boxes = boxes[_class_pos]
        classes = classes[_class_pos]

        return boxes, classes, scores

    def nms_boxes(self, boxes: np.ndarray, scores: np.ndarray):
        """
        Suppress non-maximal boxes.
        # Returns
            keep: ndarray, index of effective boxes.
        """
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:

            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= NMS_THRESH)[0]
            order = order[inds + 1]

        keep = np.array(keep)
        return keep

    def box_process(self, position):
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=1)
        stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)

        position = dfl(position)
        box_xy  = grid +0.5 -position[:,0:2,:,:]
        box_xy2 = grid +0.5 +position[:,2:4,:,:]
        xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

        return xyxy