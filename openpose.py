import sys
import numpy as np
import cv2


# 모델 & 설정 파일
model = 'pose_iter_440000.caffemodel'
config = 'pose_deploy_linevec.prototxt.txt'

# 포즈 점 개수, 점 연결 개수, 연결 점 번호 쌍
nparts = 18
npairs = 17
pose_pairs = [(1, 2), (2, 3), (3, 4),  # 왼팔
              (1, 5), (5, 6), (6, 7),  # 오른팔
              # (1, 8), (8, 9), (9, 10),  # 왼쪽다리
              # (1, 11), (11, 12), (12, 13),  # 오른쪽다리
              (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]  # 얼굴

# 테스트 이미지 파일
cv2.namedWindow('VIDEO')
vcap = cv2.VideoCapture(0)

# 네트워크 생성
net = cv2.dnn.readNet(model, config)

if net.empty():
    print('Net open failed!')
    sys.exit()

while True:
    k = cv2.waitKey(1) & 0xff
    if vcap.isOpened():
            ret, raw = vcap.read()
            if not ret:
                break

            if raw is None:
                continue

            # 블롭 생성 & 추론
            blob = cv2.dnn.blobFromImage(raw, 1/255., (368, 368)) # scale, size
            net.setInput(blob)
            out = net.forward()  # out.shape=(1, 57, 46, 46)

            h, w = raw.shape[:2]

            # 검출된 점 추출
            points = []
            for i in range(nparts):
                heatMap = out[0, i, :, :]

                '''
                heatImg = cv2.normalize(heatMap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                heatImg = cv2.resize(heatImg, (w, h))
                heatImg = cv2.cvtColor(heatImg, cv2.COLOR_GRAY2BGR)
                heatImg = cv2.addWeighted(img, 0.5, heatImg, 0.5, 0)
                cv2.imshow('heatImg', heatImg)
                cv2.waitKey()
                '''

                _, conf, _, point = cv2.minMaxLoc(heatMap)
                x = int(w * point[0] / out.shape[3])
                y = int(h * point[1] / out.shape[2])

                points.append((x, y) if conf > 0.1 else None)  # heat map threshold=0.1

            # 검출 결과 영상 만들기
            for pair in pose_pairs:
                p1 = points[pair[0]]
                p2 = points[pair[1]]

                if p1 is None or p2 is None:
                    continue

                cv2.line(raw, p1, p2, (0, 255, 0), 3, cv2.LINE_AA)
                cv2.circle(raw, p1, 4, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(raw, p2, 4, (0, 0, 255), -1, cv2.LINE_AA)

            raw =cv2.flip(raw,1)
            # 추론 시간 출력
            t, _ = net.getPerfProfile()
            label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
            cv2.putText(raw, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 1, cv2.LINE_AA)


            cv2.imshow('VIDEO',raw)
            if k == 27: # ESCAPE
                break
vcap.release()
cv2.destroyAllWindows()
