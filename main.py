import cv2
from feature_extraction import get_first_bbox, get_next_frame
import numpy as np
from tracker import Tracker

rect = get_first_bbox()
output_bbox = []
bbox = (rect[0], rect[1], rect[2], rect[3])
output_bbox.append(bbox)
frame_count = 1
ok, frame_count, frame = get_next_frame(frame_count)
print('frame', frame.shape)
tracker = Tracker()
tracker = Tracker.init_tracker(tracker, frame, bbox)
while True:
    # Read a new frame
    ok, frame_count, frame = get_next_frame(frame_count)
    if not ok:
        print('end')
        break
    # Update tracker
    ok, bbox = tracker.next_frame(frame)
    # Draw bounding box
    if ok:
        # Tracking success
        output_bbox.append(bbox)
        frame = tracker.show_results(frame=frame, bbox=bbox)
    else:
        # Tracking failure
        pass
        # cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    # Display result
    cv2.imshow("Tracking", frame)
    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
def get_ouput_bbox():
    return output_bbox
