from datetime import datetime, timedelta
import mysql.connector
import cv2
import numpy as np
import face_recognition
import os

# Dictionary to keep track of the last attendance time for each student
last_attendance_time = {}


def get_current_course():
    db = None
    try:
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="attendancedb"
        )
        cursor = db.cursor()
        now = datetime.now()
        current_time = now.time()
        day = now.weekday()  # Monday is 0 and Sunday is 6

        sql = """
        SELECT course_code FROM timetable 
        WHERE day = %s AND start_time <= %s AND end_time >= %s
        """
        cursor.execute(sql, (day, current_time, current_time))
        result = cursor.fetchone()

        if result:
            return result[0]
        else:
            return None

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

    finally:
        if db is not None and db.is_connected():
            cursor.close()
            db.close()


def markAttendance(student_no, record_type, course_code):
    db = None
    try:
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="attendancedb"
        )
        cursor = db.cursor()

        now = datetime.now()
        dtString = now.strftime('%Y-%m-%d %H:%M:%S')

        # Insert into the MySQL database
        sql = "INSERT INTO attendance (date, time, type, student_no, course_code) VALUES (%s, %s, %s, %s, %s)"
        val = (now.date(), dtString, record_type, student_no, course_code)
        cursor.execute(sql, val)
        db.commit()

        print(f"Attendance {record_type} marked for {student_no} at {dtString} for course_code {course_code}")

        # Update the last attendance time
        last_attendance_time[student_no] = (now, record_type)

    except mysql.connector.Error as err:
        print(f"Error: {err}")

    finally:
        if db is not None and db.is_connected():
            cursor.close()
            db.close()


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def load_images_from_folder(folder):
    images = []
    student_numbers = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            student_numbers.append(os.path.splitext(filename)[0])
    return images, student_numbers


# Load images from a folder
folder_path = './images'
images, student_numbers = load_images_from_folder(folder_path)

if not images:
    print("No images found in the folder.")
else:
    encodeListKnown = findEncodings(images)
    print('Encoding Complete')

    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            break

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                student_no = student_numbers[matchIndex]
                now = datetime.now()

                # Determine if clocking in or clocking out
                if student_no in last_attendance_time:
                    last_time, last_type = last_attendance_time[student_no]
                    if last_type == "ClockIn":
                        record_type = "ClockOut"
                    else:
                        record_type = "ClockIn"
                else:
                    record_type = "ClockIn"

                # Check if the student's last attendance was recorded more than 5 minutes ago
                if student_no not in last_attendance_time or now - last_attendance_time[student_no][0] > timedelta(minutes=5):
                    course_code = get_current_course()
                    if course_code:
                        markAttendance(student_no, record_type, course_code)
                        y1, x2, y2, x1 = faceLoc
                        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, student_no, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Webcam', img)

        # Check if the window close button is clicked
        if cv2.getWindowProperty('Webcam', cv2.WND_PROP_VISIBLE) < 1:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
