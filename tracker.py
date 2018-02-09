import cv2
from feature_extraction import rectangle_read, get_next_frame, first_segmentation_mask
from particle_filter import ParticleFilter

class Tracker:

    def __init__(self):
        self.tracker = None
        self.particle_filter = ParticleFilter()
        self.histo = None
        self.points = []
        self.frame_number = 0

    def init_tracker(self, frame, bbox):
        tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
        tracker_type = tracker_types[0]

        if tracker_type == tracker_types[0]:
            self.tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            self.tracker = cv2.TrackerMIL_create()
     #   if tracker_type == 'KCF':
     #       self.tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            self.tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            self.tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            self.tracker = cv2.TrackerGOTURN_create()


        # Initialize tracker with first frame and bounding box

        ok = self.tracker.init(frame, bbox)
        self.particle_filter = ParticleFilter()
        self.particle_filter = ParticleFilter.init_particles(self.particle_filter,region=bbox, particlesPerObject=100)
        img = frame[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        self.histo = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        self.points.append(ParticleFilter.get_particle_center(self.particle_filter))
        self.frame_number += 1

        return self

    def next_frame(self, frame):

        ok, bbox = self.tracker.update(frame)

        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        w = frame.shape[0]
        h = frame.shape[1]
        self.particle_filter.transition(w, h)
        self.particle_filter.update_weight(frameHSV, self.histo)
        self.particle_filter.normalize_weights()
        self.particle_filter.resample()
        self.points.append(ParticleFilter.get_particle_center(self.particle_filter))
        self.frame_number += 1

        return ok, bbox

    def show_results(self, frame, bbox):

        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        frame = first_segmentation_mask(frame, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        return frame


