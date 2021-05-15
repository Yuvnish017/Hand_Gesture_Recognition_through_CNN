import cv2 as cv

background = None
accumulated_weight = 0.5

# Creating the dimensions for the ROI...
ROI_top = 100
ROI_bottom = 300
ROI_right = 150
ROI_left = 350


def cal_accum_avg(frame, accumulated_weight):
    global background

    if background is None:
        background = frame.copy().astype("float")
        return None

    cv.accumulateWeighted(frame, background, accumulated_weight)


def segment_hand(frame, threshold=25):
    global background

    diff = cv.absdiff(background.astype("uint8"), frame)

    _, thresholded = cv.threshold(diff, threshold, 255, cv.THRESH_BINARY)

    # Grab the external contours for the image
    contours, hierarchy = cv.findContours(thresholded.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None
    else:

        hand_segment_max_cont = max(contours, key=cv.contourArea)

        return (thresholded, hand_segment_max_cont)


cam = cv.VideoCapture(0)

num_frames = 0
element = 0
num_imgs_taken = 0

while True:
    ret, frame = cam.read()

    # flipping the frame to prevent inverted image of captured frame...
    #frame = cv.flip(frame, 1)

    frame_copy = frame.copy()

    roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]

    gray_frame = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    gray_frame = cv.GaussianBlur(gray_frame, (9, 9), 0)

    if num_frames < 200:
        cal_accum_avg(gray_frame, accumulated_weight)
        if num_frames <= 199:
            cv.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT", (80, 400), cv.FONT_HERSHEY_SIMPLEX, 0.9,
                       (0, 0, 255), 2)

    # Time to configure the hand specifically into the ROI...
    elif num_frames <= 600:

        hand = segment_hand(gray_frame)

        cv.putText(frame_copy, "Adjust hand...Gesture for" + str(element), (200, 400), cv.FONT_HERSHEY_SIMPLEX, 1,
                   (0, 0, 255), 2)

        # Checking if the hand is actually detected by counting the number of contours detected...
        if hand is not None:
            thresholded, hand_segment = hand

            # Draw contours around hand segment
            cv.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0), 1)

            cv.putText(frame_copy, str(num_frames) + "For" + str(element), (70, 45), cv.FONT_HERSHEY_SIMPLEX, 1,
                       (0, 0, 255), 2)

            # Also display the thresholded image
            cv.imshow("Thresholded Hand Image", thresholded)

    else:

        # Segmenting the hand region...
        hand = segment_hand(gray_frame)

        # Checking if we are able to detect the hand...
        if hand is not None:

            # unpack the thresholded img and the max_contour...
            thresholded, hand_segment = hand

            # Drawing contours around hand segment
            cv.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0), 1)

            cv.putText(frame_copy, str(num_frames), (70, 45), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv.putText(frame_copy, str(num_imgs_taken) + 'images' + "For" + str(element), (200, 400),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Displaying the thresholded image
            cv.imshow("Thresholded Hand Image", thresholded)
            if num_imgs_taken <= 600:
                cv.imwrite(str(num_imgs_taken + 300) + '.jpg', thresholded)

            else:
                break
            num_imgs_taken += 1
        else:
            cv.putText(frame_copy, 'No hand detected...', (200, 400), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Drawing ROI on frame copy
    cv.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255, 128, 0), 3)

    cv.putText(frame_copy, "DataFlair hand sign recognition_ _ _", (10, 20), cv.FONT_ITALIC, 0.5, (51, 255, 51), 1)

    # increment the number of frames for tracking
    num_frames += 1

    # Display the frame with segmented hand
    cv.imshow("Sign Detection", frame_copy)

    # Closing windows with Esc key...(any other key with ord can be used too.)
    k = cv.waitKey(1) & 0xFF

    if k == 27:
        break

# Releasing the camera & destroying all the windows...

cv.destroyAllWindows()
cam.release()
