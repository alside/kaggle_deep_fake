import cv2
import random
import glob
from joblib import Parallel, delayed
from dsfd.detect import DSFDDetector

detector = DSFDDetector()

sources = glob.glob("../data/dfdc_train_part_00/dfdc_train_part_0/*")


def crop_video(source):
    print(source)
    fname = source.split("/")[-1].split(".")[0]
    target = random.choice(["FAKE", "REAL"])  # TODO: get from metadata
    faces_img_name = lambda x: f"../croped_faces/{fname}_{x}_{target}.jpg"  # TODO: get rid of lambda
    cam = cv2.VideoCapture(source)
    rec_shift = 150
    iface = 0
    while True:
        ret, img = cam.read()
        if img is None:
            break
        try:
            detections = detector.detect_face(img, confidence_threshold=.7, shrink=0.1)
        except:
            print("Broke wtf: ", img.shape)
            break
        for detection in detections:
            x_min, y_min, x_max, y_max, probability = detection
            if iface % 20 == 0:
                y_min_shifted = int(max([y_min - rec_shift, 0]))
                y_max_shifted = int(min([y_max + rec_shift, img.shape[0]]))
                x_min_shifted = int(max([x_min - rec_shift, 0]))
                x_max_shifted = int(min([x_max + rec_shift, img.shape[1]]))
                cv2.imwrite(faces_img_name(iface),
                            img[y_min_shifted:y_max_shifted,
                                x_min_shifted:x_max_shifted]
                            )
                print(f"written {faces_img_name(iface)}")
            iface += 1


results = Parallel(n_jobs=5)(delayed(crop_video)(i) for i in sources)
# for source in sources:
#     crop_video(source)
cv2.destroyAllWindows()
