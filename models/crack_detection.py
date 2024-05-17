import cv2
import torch
import csv
import time

# Function to compare two cracks based on area and coordinates
def calculate_conveyor_health(num_holes, avg_hole_size, avg_thinning):
    N_crit = 5       # Critical number of holes
    S_crit = 2000     # Critical average hole size. 
    T_crit = 30      # Critical average thinning (%)

    # Weights
    W_S = 0.38
    W_N = 0.35
    W_T = 0.27

    # Calculate health index
    health_index = 100 - (
        (min(num_holes / N_crit, 1) * W_N) +
        (min(avg_hole_size / S_crit, 1) * W_S) +
        (min(avg_thinning / T_crit, 1) * W_T)
    ) * 100

    return max(health_index, 0)  # Ensure health doesn't go below 0

# Example usage
num_holes = 6
avg_thinning = 12.5

def is_different_crack(crack1, crack2, threshold=0.1):
    area1, (x1, y1) = crack1
    area2, (x2, y2) = crack2

    area_diff = abs(area1 - area2) / float(max(area1, area2))
    coord_diff = max(abs(x1 - x2), abs(y1 - y2)) / max(frame_width, frame_height)

    return area_diff > threshold or coord_diff > threshold

# Load the custom YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
speed = 0.1
# Initialize the video capture object to use the default camera
cap = cv2.VideoCapture(1)
csv_file = open('cracks.csv', 'a+', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['S.No', 'Area', 'Time', 'Location'])
s_no = 1
start_time = time.time()
last_logged_crack = None  # Store the last logged crack details

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read the first frame to get the dimensions
ret, frame = cap.read()
if not ret:
    print("Error: Couldn't read frame.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Define the dimensions of the frame
frame_height, frame_width = frame.shape[:2]

# Define the ROI as a small vertical strip in the center
strip_width = 50
roi_x1 = frame_width // 2 - strip_width // 2
roi_y1 = 0
roi_width = strip_width
roi_height = frame_height
num_holes = 0
avg_hole_size = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't read frame.")
        break

    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x1 + roi_width, roi_y1 + roi_height), (255, 0, 0), 2)
    results = model(frame)

    for detection in results.pred[0]:
        *xyxy, conf, cls = detection
        if results.names[int(cls)] == 'crack':
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Crack", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if (x1 >= roi_x1 and x2 <= roi_x1 + roi_width) and (y1 >= roi_y1 and y2 <= roi_y1 + roi_height):
                crack_area = (x2 - x1) * (y2 - y1)
                crack_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                current_crack = (crack_area, crack_center)
                if last_logged_crack is None or is_different_crack(last_logged_crack, current_crack):
                    current_time = time.time() - start_time
                    csv_writer.writerow([s_no, crack_area, (current_time), (current_time*speed)])
                    last_logged_crack = current_crack
                    s_no += 1
                    print("New crack found and logged")
                    print(current_time)

    cv2.imshow('YOLOv5n Live Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
health_status = calculate_conveyor_health(num_holes, avg_hole_size, avg_thinning)
print(f"Conveyor Belt Health: {health_status}%")
csv_file.close()
