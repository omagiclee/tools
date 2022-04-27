import sys
import time
from pathlib import Path

import cv2

savep = Path("Res")
if not savep.exists():
    savep.mkdir(parents=True)



stime = time.time()
cap = cv2.VideoCapture(sys.argv[1])

cnt = 1
while True:
    success, im = cap.read()
    if success:
        cnt += 1
        cv2.imwrite("{}/{:04d}.jpg".format(savep, cnt), im)
    else:
        break

cap.release()
etime = time.time()
print('duration: {}s'.format(etime - stime))