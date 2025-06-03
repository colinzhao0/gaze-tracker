import cv2
import mediapipe as mp
import numpy as np
import math
import time
from scipy.spatial import distance as dist
import csv
import tkinter as tk

mp_face = mp.solutions.face_mesh
cap = cv2.VideoCapture(0)

LEFT_IRIS_CENTER_INDEX = 473
RIGHT_IRIS_CENTER_INDEX = 468

LEFT_EYE_TOP = 27
LEFT_EYE_BOT = 23
RIGHT_EYE_TOP = 257
RIGHT_EYE_BOT = 253

LEFT_EYE_OUTER_INDEX = 263 
LEFT_EYE_INNER_INDEX = 362
RIGHT_EYE_OUTER_INDEX = 133
RIGHT_EYE_INNER_INDEX = 33

distance_px_right = 0
distance_px_left = 0

mm_to_px = 3.7795275591  # 1 mm = 3.7795275591 px (at 96 DPI)

filename = "gazetracking_data.csv"
with open(filename, 'w+') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['gaze_x', 'gaze_y', 'isBlinking'])

def estimate_distance(iris_left, iris_right, img_w, focal_length_px=1920, real_iris_mm=11.7):
    # iris_left, iris_right: landmark objects with .x, .y
    # img_w: image width in pixels
    # focal_length_px: estimated focal length in pixels (default: 960 for 1080p webcam)
    # real_iris_mm: average human iris diameter in mm

    # Calculate iris diameter in pixels
    x1 = int(iris_left.x * img_w)
    x2 = int(iris_right.x * img_w)
    iris_diameter_px = abs(x2 - x1)

    if iris_diameter_px == 0:
        return None

    # Estimate distance (in mm)
    distance_mm = (real_iris_mm * focal_length_px) / iris_diameter_px
    return distance_mm * 3.780/3 + 2000

def get_angle_x(iris_x, eye_outer_x, eye_inner_x, callibration_constant, sensitivity=2):
    # Calculate the angle of the iris relative to the eye center using arcsin
    # eye_outer_x: outer eye corner (temporal side)
    # eye_inner_x: inner eye corner (nasal side)
    width = eye_inner_x - eye_outer_x
    # Move inner eye point 5% of width closer to outer eye point
    adj_eye_inner_x = eye_inner_x - 0.05 * width
    width = adj_eye_inner_x - eye_outer_x  # recalculate width after moving inner point
    middle_x = (eye_outer_x + adj_eye_inner_x) / 2
    dist_from_mid = (iris_x - middle_x) * 2  # Multiply by 2 to get full range [-width, width]
    # Clamp to avoid domain errors in asin
    norm = np.clip(dist_from_mid / width, -1.0, 1.0)
    # Human eye can rotate about ±50 degrees horizontally
    max_eye_angle = 50  # degrees
    angle = np.arcsin(norm) * (180 / np.pi)  # Convert to degrees
    angle = np.clip(angle, -max_eye_angle, max_eye_angle)

    return angle * sensitivity - callibration_constant  # Return callibrated angle with sensitivity adjustment

def get_angle_y(iris_y, eye_upper_y, eye_lower_y, callibration_deg, sensitivity=3):
    # Calculate the angle of the iris relative to the vertical center of the eye using arcsin
    height = eye_lower_y - eye_upper_y
    middle_y = (eye_upper_y + eye_lower_y) / 2
    dist_from_mid = (iris_y - middle_y) * 2  # Multiply by 2 to get full range [-height, height]
    # Clamp to avoid domain errors in asin
    norm = np.clip(dist_from_mid / height, -1.0, 1.0)
    # Assume human eye can rotate about ±30 degrees vertically
    max_eye_angle = 30  # degrees
    angle = np.arcsin(norm) * (180 / np.pi)  # Convert to degrees
    angle = np.clip(angle, -max_eye_angle, max_eye_angle)
    return angle * sensitivity - callibration_deg

def getEAR(eye_inner, eye_outer, eye_lower_outer, eye_lower_inner, eye_upper_outer, eye_upper_inner):
    hor_dist = abs(eye_inner.x - eye_outer.x)
    vert_dist1 = eye_lower_outer.y - eye_upper_outer.y
    vert_dist2 = eye_lower_inner.y - eye_upper_inner.y
    return (vert_dist1 + vert_dist2)/hor_dist
    pass

def isBlinking(ear, threshold=.45):
    return ear <= threshold

def getGazePos(distance_px, gaze_angle, iris_center, img_dim):
    # Calculate gaze position in pixels given distance, angle, iris center, and image dimension (width or height)
    return int(distance_px * math.tan(math.radians(gaze_angle)) + iris_center * img_dim)

def get_eye_landmarks(mesh, side="left"):
    if side == "left":
        return {
            "IRIS_CENTER": mesh.landmark[LEFT_IRIS_CENTER_INDEX],
            "EYE_OUTER": mesh.landmark[LEFT_EYE_OUTER_INDEX],
            "EYE_INNER": mesh.landmark[LEFT_EYE_INNER_INDEX],
            "IRIS_OUTER": mesh.landmark[474],
            "IRIS_INNER": mesh.landmark[476],
            "EYE_UPPER": mesh.landmark[LEFT_EYE_TOP],
            "EYE_LOWER": mesh.landmark[LEFT_EYE_BOT],
            "EYE_UPPER_OUTER": mesh.landmark[387],
            "EYE_UPPER_INNER": mesh.landmark[385],
            "EYE_LOWER_OUTER": mesh.landmark[373],
            "EYE_LOWER_INNER": mesh.landmark[380],
        }
    else:
        return {
            "IRIS_CENTER": mesh.landmark[RIGHT_IRIS_CENTER_INDEX],
            "EYE_OUTER": mesh.landmark[RIGHT_EYE_OUTER_INDEX],
            "EYE_INNER": mesh.landmark[RIGHT_EYE_INNER_INDEX],
            "IRIS_OUTER": mesh.landmark[469],
            "IRIS_INNER": mesh.landmark[471],
            "EYE_UPPER": mesh.landmark[RIGHT_EYE_TOP],
            "EYE_LOWER": mesh.landmark[RIGHT_EYE_BOT],
            "EYE_UPPER_OUTER": mesh.landmark[160],
            "EYE_UPPER_INNER": mesh.landmark[158],
            "EYE_LOWER_OUTER": mesh.landmark[144],
            "EYE_LOWER_INNER": mesh.landmark[153],
        }

def run_calibration(face_mesh, cap, img_w, img_h):
    # Calibration points: center, left, right, up, down (relative to screen)
    points = [
        (img_w // 2, img_h // 2),  # center
        (int(img_w * 0.15), img_h // 2),  # left
        (int(img_w * 0.85), img_h // 2),  # right
        (img_w // 2, int(img_h * 0.15)),  # up
        (img_w // 2, int(img_h * 0.85)),  # down
    ]
    gaze_x_angles = []
    gaze_y_angles = []
    instructions = [
        "Look at the CENTER dot and press SPACE",
        "Look at the LEFT dot and press SPACE",
        "Look at the RIGHT dot and press SPACE",
        "Look at the TOP dot and press SPACE",
        "Look at the BOTTOM dot and press SPACE",
    ]
    for idx, (px, py) in enumerate(points):
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame_disp = frame.copy()
            cv2.circle(frame_disp, (px, py), 30, (0, 0, 255), -1)
            cv2.putText(frame_disp, instructions[idx], (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.imshow("Calibration", frame_disp)
            key = cv2.waitKey(1)
            if key == 27:  # ESC to abort
                cv2.destroyWindow("Calibration")
                return None
            if key == 32:  # SPACE to record
                results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.multi_face_landmarks:
                    mesh = results.multi_face_landmarks[0]
                    left_eye = get_eye_landmarks(mesh, "left")
                    right_eye = get_eye_landmarks(mesh, "right")
                    # Use left eye for calibration
                    gaze_x_angle = get_angle_x(
                        left_eye["IRIS_CENTER"].x, left_eye["EYE_OUTER"].x, left_eye["EYE_INNER"].x, 0, 1
                    )
                    gaze_y_angle = get_angle_y(
                        left_eye["IRIS_CENTER"].y, left_eye["EYE_UPPER"].y, left_eye["EYE_LOWER"].y, 0, 1
                    )
                    gaze_x_angles.append(gaze_x_angle)
                    gaze_y_angles.append(gaze_y_angle)
                    break

    cv2.destroyWindow("Calibration")
    # Calculate calibration constants and sensitivities
    # Center dot is index 0, left 1, right 2, up 3, down 4
    center_x = gaze_x_angles[0]
    left_x = gaze_x_angles[1]
    right_x = gaze_x_angles[2]
    center_y = gaze_y_angles[0]
    up_y = gaze_y_angles[3]
    down_y = gaze_y_angles[4]
    # Sensitivity: how much angle changes per dot distance (assume ±30 deg for x, ±20 deg for y)
    angle_sens_x = 30 / abs(right_x - left_x) if abs(right_x - left_x) > 1e-3 else 2
    angle_sens_y = 20 / abs(up_y - down_y) if abs(up_y - down_y) > 1e-3 else 2
    callibration_constant_x = center_x * angle_sens_x
    callibration_constant_y = center_y * angle_sens_y
    return callibration_constant_x, angle_sens_x, callibration_constant_y, angle_sens_y

with mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:

    # Initialize smoothed gaze position
    smoothed_gaze_x = None
    smoothed_gaze_y = None
    smoothing_factor = 0.2  # 0 < smoothing_factor <= 1, lower is smoother

    # Calibration
    # Get a frame to determine image size
    while True:
        ret, frame = cap.read()
        if ret:
            img_h, img_w = frame.shape[:2]
            break
    # Run calibration
    calib = run_calibration(face_mesh, cap, img_w, img_h)
    if calib is None:
        print("Calibration aborted.")
        cap.release()
        cv2.destroyAllWindows()
        exit()
    callibration_constant_x, angle_sens_x, callibration_constant_y, angle_sens_y = calib

    while True:
        ret, frame = cap.read()
        img_h, img_w = frame.shape[:2]

        if not ret:
            break
    
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            mesh = results.multi_face_landmarks[0]

            left_eye = get_eye_landmarks(mesh, "left")
            right_eye = get_eye_landmarks(mesh, "right")

            # draw points on eye
            points = [
                left_eye["IRIS_CENTER"], left_eye["EYE_OUTER"], left_eye["EYE_INNER"],
                right_eye["IRIS_CENTER"], right_eye["EYE_OUTER"], right_eye["EYE_INNER"],
                left_eye["EYE_UPPER"], left_eye["EYE_LOWER"], right_eye["EYE_UPPER"], right_eye["EYE_LOWER"],
            ]
            for pt in points:
                x = int(pt.x * img_w)
                y = int(pt.y * img_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            # get x pos of gaze for each eye then average
            distance_l_prev = distance_px_left
            distance_r_prev = distance_px_right
            distance_px_left = estimate_distance(left_eye["IRIS_OUTER"], left_eye["IRIS_INNER"], img_w)
            distance_px_right = estimate_distance(right_eye["IRIS_OUTER"], right_eye["IRIS_INNER"], img_w)

            # reduce fluctuation of distance while maintaing accuracy
            distance_px_left = (distance_px_left + distance_l_prev)//2
            distance_px_right = (distance_px_right + distance_r_prev)//2

            callibration_constant_x = 9
            angle_sens_x = 2.5
            gaze_x_angle_left = get_angle_x(left_eye["IRIS_CENTER"].x, left_eye["EYE_OUTER"].x, left_eye["EYE_INNER"].x, callibration_constant_x, angle_sens_x)
            gaze_x_angle_right = get_angle_x(right_eye["IRIS_CENTER"].x, right_eye["EYE_OUTER"].x, right_eye["EYE_INNER"].x, callibration_constant_x, angle_sens_x)

            callibration_constant_y = 15
            angle_sens_y = 2
            gaze_y_angle_left = get_angle_y(right_eye["IRIS_CENTER"].y, right_eye["EYE_UPPER"].y, right_eye["EYE_LOWER"].y, callibration_constant_y, angle_sens_y)
            gaze_y_angle_right = get_angle_y(right_eye["IRIS_CENTER"].y, right_eye["EYE_UPPER"].y, left_eye["EYE_LOWER"].y, callibration_constant_y, angle_sens_y)

            gaze_x_left = getGazePos(distance_px_left, gaze_x_angle_left, left_eye["IRIS_CENTER"].x, img_w)
            gaze_x_right = getGazePos(distance_px_right, gaze_x_angle_right, right_eye["IRIS_CENTER"].x, img_w)
            gaze_y_left = getGazePos(distance_px_left, gaze_y_angle_left, left_eye["IRIS_CENTER"].y, img_h)
            gaze_y_right = getGazePos(distance_px_left, gaze_y_angle_right, right_eye["IRIS_CENTER"].y, img_h)

            gaze_x = int((gaze_x_left + gaze_x_right) / 2)
            if gaze_x <= img_w * 2 and gaze_x >= img_w * -1:
                gaze_x = np.clip(gaze_x, 0, img_w)

            gaze_y = int((gaze_y_left + gaze_y_right) / 2)
            if gaze_y <= img_h * 2 and gaze_y >= img_h * -1:
                gaze_y = np.clip(gaze_y, 0, img_h)

            #detect blinks
            leftBlinking = isBlinking(getEAR(
                left_eye["EYE_INNER"], left_eye["EYE_OUTER"],
                left_eye["EYE_LOWER_INNER"], left_eye["EYE_LOWER_OUTER"],
                left_eye["EYE_UPPER_INNER"], left_eye["EYE_UPPER_OUTER"]
            ))
            rightBlinking = isBlinking(getEAR(
                right_eye["EYE_INNER"], right_eye["EYE_OUTER"],
                right_eye["EYE_LOWER_INNER"], right_eye["EYE_LOWER_OUTER"],
                right_eye["EYE_UPPER_INNER"], right_eye["EYE_UPPER_OUTER"]
            ))
            blinking = leftBlinking and rightBlinking

            if blinking:
                gaze_x = gaze_y = None

            # Smooth the gaze tracker movement
            if gaze_x is not None and gaze_y is not None:
                if smoothed_gaze_x is None or smoothed_gaze_y is None:
                    smoothed_gaze_x = gaze_x
                    smoothed_gaze_y = gaze_y
                else:
                    smoothed_gaze_x = int(smoothed_gaze_x + smoothing_factor * (gaze_x - smoothed_gaze_x))
                    smoothed_gaze_y = int(smoothed_gaze_y + smoothing_factor * (gaze_y - smoothed_gaze_y))
            else:
                smoothed_gaze_x = None
                smoothed_gaze_y = None

            # Display info on frame
            frame = cv2.flip(frame, 1) #flip image
            cv2.putText(frame, f'Hor. Gaze Angle L: {gaze_x_angle_left:.1f} deg', (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, f'Hor. Gaze Angle R: {gaze_x_angle_right:.1f} deg', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, f'Vert. Gaze Angle L: {gaze_y_angle_left:.1f} deg', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, f'Vert. Gaze Angle R: {gaze_y_angle_right:.1f} deg', (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, f'Distance L: {distance_px_left:.1f} px', (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, f'Distance R: {distance_px_right:.1f} px', (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            if not blinking:
                cv2.putText(frame, f'Gaze X: {gaze_x} px', (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.putText(frame, f'Gaze Y: {gaze_y} px', (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, f'Window Width: {img_w} px', (30, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, f'Blinking: {blinking}', (30, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            overlay = frame.copy()
            # Draw the smoothed gaze point
            if smoothed_gaze_x is not None and smoothed_gaze_y is not None:
                cv2.circle(overlay, (smoothed_gaze_x, smoothed_gaze_y), 75, (255, 0, 0), -1)  # Draw gaze point
            alpha = 0.3  # Transparency factor.
            frame_new = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            #add csv data to file
            with open(filename, 'a') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([gaze_x, gaze_y, blinking])

        cv2.imshow('Eyetracker', frame_new)
        cv2.waitKey(0) #& 0xFF == ord('q')
        
cap.release()
cv2.destroyAllWindows()
