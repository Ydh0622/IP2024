import numpy as np
import cv2

def run_main():
    # 동영상 파일 경로 설정
    video_path = "C:\\Users\\유덕현\\Desktop\\해양데이터\\coin.mp4"
    cap = cv2.VideoCapture(video_path)

    # 동영상 파일 확인
    if not cap.isOpened():
        print("Error: 동영상을 열 수 없습니다. 경로를 확인하세요.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("동영상이 끝났습니다.")
            # 동영상이 끝나도 자동 종료되지 않도록 설정
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 동영상 처음으로 되돌리기
            continue

        # 전체 화면 또는 코인이 있는 영역으로 ROI 설정
        roi = frame[0:720, 0:1280]  # 전체 화면을 사용하거나 적절히 조정
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 블러링 및 적응형 이진화
        gray_blur = cv2.GaussianBlur(gray, (9, 9), 0)  # 블러 크기를 줄임
        thresh = cv2.adaptiveThreshold(
            gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 9, 2  # 임계값을 약간 수정
        )

        # 모폴로지 닫힘 연산
        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=4)

        # 외곽선 검출
        cont_img = closing.copy()
        contours, hierarchy = cv2.findContours(
            cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 외곽선 필터링 및 타원 그리기
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000 or area > 5000:  # 면적 범위 확장
                continue

            if len(cnt) < 5:  # 충분한 점이 없으면 건너뜀
                continue

            # 타원을 외곽선에 맞게 그리기
            ellipse = cv2.fitEllipse(cnt)
            cv2.ellipse(roi, ellipse, (0, 255, 0), 2)

        # 결과 표시
        cv2.imshow("Morphological Closing", closing)
        cv2.imshow("Adaptive Thresholding", thresh)
        cv2.imshow('Contours', roi)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(100) & 0xFF == ord('q'):  # 속도 줄이기 위해 대기 시간 100ms로 설정
            break

    # 리소스 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_main()
