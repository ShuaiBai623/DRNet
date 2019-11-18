import matplotlib
import importlib
matplotlib.use('TkAgg')
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 as cv
import time
import copy
# from pysot.datasets import DatasetFactory
# from pysot.utils.region import vot_overlap, vot_float2str
import numpy as np

def get_axis_aligned_bbox(region):
    """ convert region to (cx, cy, w, h) that represent by axis aligned box
    """
    nv = region.size
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x+w/2
        cy = y+h/2
    return cx, cy, w, h

class BaseTracker:
    """Base class for all trackers."""

    def __init__(self, params):
        self.params = params

    def initialize(self, image, state, class_info=None):
        """Overload this function in your tracker. This should initialize the model."""
        raise NotImplementedError

    def track(self, image):
        """Overload this function in your tracker. This should track in the frame and update the model."""
        raise NotImplementedError

    def track_sequence(self, sequence):
        """Run tracker on a sequence."""

        # Initialize
        image = self._read_image(sequence.frames[0])

        times = []
        start_time = time.time()
        self.initialize(image, sequence.init_state)
        init_time = getattr(self, 'time', time.time() - start_time)
        times.append(init_time)
        if self.params.visualization:
            self.init_visualization()
            self.visualize(image, sequence.init_state)

        # Track
        tracked_bb = [sequence.init_state]
        for frame in sequence.frames[1:]:
            image = self._read_image(frame)

            start_time = time.time()
            state = self.track(image)
            times.append(time.time() - start_time)

            tracked_bb.append(state)

            if self.params.visualization:
                self.visualize(image, state)

        return tracked_bb, times
    def track_sequence_vot(self, sequence):
        """Run tracker on a sequence."""
        frame_counter = 0
        pred_bboxes = []
        times = []
        start_time = time.time()
        lost_number = 0
        iou_avg = 0
        count = 0
        for idx, frame in enumerate(sequence.frames):
        # Initialize
            gt_bbox = sequence.ground_truth_rect[idx,:]
            # print(frame)
            # im = cv.imread(frame) 
            if len(gt_bbox) > 4:
                gt_x_all = gt_bbox[[0, 2, 4, 6]]
                gt_y_all = gt_bbox[[1, 3, 5, 7]]
                x1 = int(np.min(gt_x_all))
                y1 = int(np.min(gt_y_all))
                x2 = int(np.max(gt_x_all))
                y2 = int(np.max(gt_y_all))
                tmp_box=[x1,y1,x2,y2]
            else:
                tmp_box=[int(gt_bbox[0]),int(gt_bbox[1]),int(gt_bbox[0]+gt_bbox[2]-1),int(gt_bbox[1]+gt_bbox[3]-1)]
            if len(gt_bbox) == 4:
                gt_bbox = [gt_bbox[0], gt_bbox[1],
                gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]] 
            # tmp_box = np.int0(gt_bbox)
            # cv.rectangle(im, (tmp_box[0], tmp_box[1]),(tmp_box[2], tmp_box[3]), (0, 0, 255), 2)
            if idx == frame_counter:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                image = self._read_image(frame)
                tracker_module = importlib.import_module('pytracking.tracker.{}'.format('drnet_mft2'))
                tracker_class = tracker_module.get_tracker_class()
                param_module = importlib.import_module('pytracking.parameter.{}.{}'.format('drnet_mft2', 'default_vot_mft'))
                params = param_module.parameters()
                self.tracker = tracker_class(params)

                self.tracker.initialize(image, list(gt_bbox))
                init_time = getattr(self, 'time', time.time() - start_time)
                times.append(init_time)
                pred_bbox = gt_bbox
                pred_bboxes.append(1)
                iou_avg+=1.
                count += 1

            elif idx > frame_counter:
                image = self._read_image(frame)
                start_time = time.time()
                pred_bbox = self.tracker.track(image)
                times.append(time.time() - start_time)
                overlap = vot_overlap(pred_bbox, gt_bbox, (image.shape[1], image.shape[0]))
                # print(pred_bbox)
                # print((int(pred_bbox[0]), int(pred_bbox[1])), (int(pred_bbox[0]+pred_bbox[2]), int(pred_bbox[1]+pred_bbox[3])))
                # cv.rectangle(im, (int(pred_bbox[0]), int(pred_bbox[1])), (int(pred_bbox[0]+pred_bbox[2]), int(pred_bbox[1]+pred_bbox[3])), (0, 255, 0), 2)
                
                if overlap > 0:
                    # not lost
                    pred_bboxes.append(pred_bbox)
                    iou_avg+=overlap
                    count += 1
                else:
                    print('---------------lost-----')
                    # lost object
                    pred_bboxes.append(2)
                    frame_counter = idx + 5 # skip 5 frames
                    lost_number += 1
            # cv.imwrite('/mnt/lustre/baishuai/look/1/'+frame.split('/')[-1],im)

        print('Video: {:12s}  Lost: {:d} avg_iou: {:.2f}'.format(sequence.name,lost_number,iou_avg/float(count)))
        return pred_bboxes, times

    def track_sequence_vot_se50(self, sequence):
        """Run tracker on a sequence."""
        frame_counter = 0
        pred_bboxes = []
        times = []
        start_time = time.time()
        lost_number = 0
        for idx, frame in enumerate(sequence.frames):
        # Initialize
            gt_bbox = sequence.ground_truth_rect[idx,:] 
            if len(gt_bbox) == 4:
                gt_bbox = [gt_bbox[0], gt_bbox[1],
                gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]] 
            if idx == frame_counter:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                image = self._read_image(frame)
                tracker_module = importlib.import_module('pytracking.tracker.{}'.format('eco'))
                tracker_class = tracker_module.get_tracker_class()
                param_module = importlib.import_module('pytracking.parameter.{}.{}'.format('eco', 'default'))
                params = param_module.parameters()
                self.tracker = tracker_class(params)
                self.tracker.initialize(image, list(gt_bbox))
                init_time = getattr(self, 'time', time.time() - start_time)
                times.append(init_time)
                pred_bbox = gt_bbox
                pred_bboxes.append(1)

            elif idx > frame_counter:
                image = self._read_image(frame)
                start_time = time.time()
                pred_bbox = self.tracker.track(image)
                times.append(time.time() - start_time)
                overlap = vot_overlap(pred_bbox, gt_bbox, (image.shape[1], image.shape[0]))
                if overlap > 0:
                    # not lost
                    pred_bboxes.append(pred_bbox)
                else:
                    # lost object
                    pred_bboxes.append(2)
                    frame_counter = idx + 5 # skip 5 frames
                    lost_number += 1

        print('Video: {:12s}  Lost: {:d}'.format(sequence.name,lost_number))
        return pred_bboxes, times

    def track_sequence_vot_res50(self, sequence):
        """Run tracker on a sequence."""
        frame_counter = 0
        pred_bboxes = []
        times = []
        start_time = time.time()
        lost_number = 0
        for idx, frame in enumerate(sequence.frames):
        # Initialize
            gt_bbox = sequence.ground_truth_rect[idx,:] 
            if len(gt_bbox) == 4:
                gt_bbox = [gt_bbox[0], gt_bbox[1],
                gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]] 
            if idx == frame_counter:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                image = self._read_image(frame)
                tracker_module = importlib.import_module('pytracking.tracker.{}'.format('drnet'))
                tracker_class = tracker_module.get_tracker_class()
                param_module = importlib.import_module('pytracking.parameter.{}.{}'.format('drnet', 'default_vot'))
                params = param_module.parameters()
                self.tracker = tracker_class(params)
                self.tracker.initialize(image, list(gt_bbox))
                init_time = getattr(self, 'time', time.time() - start_time)
                times.append(init_time)
                pred_bbox = gt_bbox
                pred_bboxes.append(1)

            elif idx > frame_counter:
                image = self._read_image(frame)
                start_time = time.time()
                pred_bbox = self.tracker.track(image)
                times.append(time.time() - start_time)
                overlap = vot_overlap(pred_bbox, gt_bbox, (image.shape[1], image.shape[0]))
                if overlap > 0:
                    # not lost
                    pred_bboxes.append(pred_bbox)
                else:
                    # lost object
                    pred_bboxes.append(2)
                    frame_counter = idx + 5 # skip 5 frames
                    lost_number += 1

        print('Video: {:12s}  Lost: {:d}'.format(sequence.name,lost_number))
        return pred_bboxes, times
    def track_webcam(self):
        """Run tracker with webcam."""

        class UIControl:
            def __init__(self):
                self.mode = 'init'  # init, select, track
                self.target_tl = (-1, -1)
                self.target_br = (-1, -1)
                self.mode_switch = False

            def mouse_callback(self, event, x, y, flags, param):
                if event == cv.EVENT_LBUTTONDOWN and self.mode == 'init':
                    self.target_tl = (x, y)
                    self.target_br = (x, y)
                    self.mode = 'select'
                    self.mode_switch = True
                elif event == cv.EVENT_MOUSEMOVE and self.mode == 'select':
                    self.target_br = (x, y)
                elif event == cv.EVENT_LBUTTONDOWN and self.mode == 'select':
                    self.target_br = (x, y)
                    self.mode = 'track'
                    self.mode_switch = True

            def get_tl(self):
                return self.target_tl if self.target_tl[0] < self.target_br[0] else self.target_br

            def get_br(self):
                return self.target_br if self.target_tl[0] < self.target_br[0] else self.target_tl

            def get_bb(self):
                tl = self.get_tl()
                br = self.get_br()

                bb = [tl[0], tl[1], br[0] - tl[0], br[1] - tl[1]]
                return bb

        ui_control = UIControl()
        cap = cv.VideoCapture(0)
        display_name = 'Display: ' + self.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        cv.setMouseCallback(display_name, ui_control.mouse_callback)

        if hasattr(self, 'initialize_features'):
            self.initialize_features()

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            frame_disp = frame.copy()

            if ui_control.mode == 'track' and ui_control.mode_switch:
                ui_control.mode_switch = False
                init_state = ui_control.get_bb()
                self.initialize(frame, init_state)

            # Draw box
            if ui_control.mode == 'select':
                cv.rectangle(frame_disp, ui_control.get_tl(), ui_control.get_br(), (255, 0, 0), 2)
            elif ui_control.mode == 'track':
                state = self.track(frame)
                state = [int(s) for s in state]
                cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                             (0, 255, 0), 5)

            # Put text
            font_color = (0, 0, 0)
            if ui_control.mode == 'init' or ui_control.mode == 'select':
                cv.putText(frame_disp, 'Select target', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 1)
                cv.putText(frame_disp, 'Press q to quit', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
            elif ui_control.mode == 'track':
                cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
                cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
                cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ui_control.mode = 'init'

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

    def reset_tracker(self):
        pass

    def press(self, event):
        if event.key == 'p':
            self.pause_mode = not self.pause_mode
            print("Switching pause mode!")
        elif event.key == 'r':
            self.reset_tracker()
            print("Resetting target pos to gt!")

    def init_visualization(self):
        # plt.ion()
        self.pause_mode = False
        self.fig, self.ax = plt.subplots(1)
        self.fig.canvas.mpl_connect('key_press_event', self.press)
        plt.tight_layout()

    def visualize(self, image, state):
        self.ax.cla()
        self.ax.imshow(image)
        rect = patches.Rectangle((state[0], state[1]), state[2], state[3], linewidth=1, edgecolor='r', facecolor='none')
        self.ax.add_patch(rect)

        if hasattr(self, 'gt_state') and False:
            gt_state = self.gt_state
            rect = patches.Rectangle((gt_state[0], gt_state[1]), gt_state[2], gt_state[3], linewidth=1, edgecolor='g',
                                     facecolor='none')
            self.ax.add_patch(rect)
        self.ax.set_axis_off()
        self.ax.axis('equal')
        plt.draw()
        plt.pause(0.001)

        if self.pause_mode:
            plt.waitforbuttonpress()

    def _read_image(self, image_file: str):
        return cv.cvtColor(cv.imread(image_file), cv.COLOR_BGR2RGB)

