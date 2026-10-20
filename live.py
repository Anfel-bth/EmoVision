import cv2
import numpy as np
from collections import deque
import predictor

labels = predictor.labels
history = deque(maxlen=7)          # average the last 7 frames so labels don't flicker

cap = cv2.VideoCapture(0)          # try 1 if you get a black window or an error

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)     # mirror view
    results = predictor.analyze(frame)

    # smooth the main (largest) face
    main = None
    if results:
        main = max(results, key=lambda r: r["box"][2] * r["box"][3])
        history.append(np.array([main["scores"][l] for l in labels]))
        avg = np.mean(history, axis=0)
        main["emotion"] = labels[int(avg.argmax())]
        main["confidence"] = float(avg.max())
        main["scores"] = dict(zip(labels, avg))
    else:
        history.clear()

    for r in results:
        x, y, w, h = r["box"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)

        cv2.putText(frame, f"{r['emotion']} {r['confidence']:.0%}",
                    (x, max(y - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 200, 0), 2)

    # confidence bars for the main face
    if main:
        for i, l in enumerate(labels):
            yy = 25 + i * 22
            s = main["scores"][l]
            cv2.putText(frame, l, (10, yy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)
            cv2.rectangle(frame, (90, yy - 12), (90 + int(s * 150), yy), (0, 200, 0), -1)

    cv2.imshow("EmoVision (press q to quit)", frame)
    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
        break

cap.release()
cv2.destroyAllWindows()