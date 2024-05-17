### This code will provide real time edge detection using Canny edge Detection

import cv2
import numpy as np

# custom canny edge detection code
def canny_edge_detection(frame): 
	# Convert the frame to grayscale for edge detection 
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
	
	# Apply Gaussian blur to reduce noise and smoothen edges 
	blurred = cv2.GaussianBlur(src=gray, ksize=(3, 5), sigmaX=0.5) 
	
	# Perform Canny edge detection 
	edges = cv2.Canny(blurred, 70, 135) 
	
	return blurred, edges


# 	# Open the default webcam
# cap = cv2.VideoCapture(0)

    ## now we need to remove all the horizonatal edges and keep only vertical edge 

def extract_vertical_edges(edges):
    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # Create a binary mask to extract vertical lines
    vertical_mask = np.zeros_like(edges)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if np.abs(angle) > 80:  # Consider lines with an angle more than 45 degrees as vertical
                cv2.line(vertical_mask, (x1, y1), (x2, y2), 255, 2)

    # Bitwise AND operation to extract vertical edges
    vertical_edges = cv2.bitwise_and(edges, vertical_mask)

    return vertical_edges

# check if the dislodgement is there or not, the principle is that all the edges should lie between the upper and lower bound

def dislodgement(vertical_edges, left_bound_x, right_bound_x, window_height):
    # Extract the specified window from the vertical edges
    window = vertical_edges[:window_height, left_bound_x:right_bound_x]

    # Calculate the sum of pixel values along the horizontal axis
    horizontal_projection = np.sum(window, axis=1)

    # Check if any value in the horizontal projection is outside the specified bounds
    dislodged = any(value > 0 for value in horizontal_projection)

    return dislodged



capture = cv2.VideoCapture(0)
window_height = 600  # Set your window height
while True:
    ret, frame = capture.read()
    if not ret:
        break
    blurred_frame, edges = canny_edge_detection(frame)
    vertical_edges = extract_vertical_edges(edges)

    left_bound_x = 100  # Set your left bound x-coordinate
    right_bound_x = 600  # Set your right bound x-coordinate
    is_dislodged = dislodgement(vertical_edges, left_bound_x, right_bound_x, window_height)


    cv2.imshow('Edges', edges)
    cv2.imshow('Vertical Edges', vertical_edges)
    cv2.line(frame,(left_bound_x,0),(left_bound_x,window_height),(0,255,0),4) # draws the left_bound
    cv2.line(frame, (right_bound_x, 0), (right_bound_x,window_height), (0,255,0), 4) # draws the right_bound


    if is_dislodged:
        cv2.putText(frame,"Belt is stable.",(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
        print("Belt is stable.")
    else:
        cv2.putText(frame,"Dislodgement detected!!",(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
        print("Dislodgement detected!!")

    cv2.imshow('Original Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()