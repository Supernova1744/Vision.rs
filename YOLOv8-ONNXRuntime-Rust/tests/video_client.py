import sys
import cv2
import grpc
import random
import result_pb2 as yolo_pb2
import result_pb2_grpc as yolo_pb2_grpc


def draw_bboxes(frame, bboxes, class_names = [], class_colors={}):
    """
    Draws bounding boxes and associated text (class name, id, and confidence) on the frame.
    
    Args:
        frame (numpy.ndarray): The image frame to draw on.
        bboxes (list): List of bounding box objects with attributes:
                       xmin, ymin, width, height, id, confidence, class_id.
        class_names (list): List of class names corresponding to class IDs.
        class_colors (dict, optional): Dictionary mapping class IDs to BGR colors.
                                       If None, random colors will be generated.
    
    Returns:
        numpy.ndarray: The frame with bounding boxes and labels drawn.
    """

    for bbox in bboxes:
        # Define top-left and bottom-right corners of the rectangle
        top_left = (int(bbox.xmin), int(bbox.ymin))
        bottom_right = (int(bbox.xmin + bbox.width), int(bbox.ymin + bbox.height))

        # Get the color for the current class
        if bbox.id not in class_colors:
            class_colors[bbox.id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        color = class_colors[bbox.id]

        # Draw rectangle on the frame
        cv2.rectangle(frame, top_left, bottom_right, color, 2)

        # Prepare label with class name, id, and confidence
        class_name = class_names[bbox.id] if len(class_names) else bbox.id
        label = f"{class_name}: ID {bbox.id}, Conf: {bbox.confidence:.2f}"

        # Calculate text size for proper placement
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1
        )

        # Define the position for the label background rectangle
        text_bg_top_left = (int(bbox.xmin), int(bbox.ymin) - text_height - 10)
        text_bg_bottom_right = (int(bbox.xmin) + text_width + 10, int(bbox.ymin))

        # Draw filled rectangle as the background for the label
        cv2.rectangle(frame, text_bg_top_left, text_bg_bottom_right, color, -1)

        # Put the label text inside the background rectangle
        cv2.putText(
            frame,
            label,
            (int(bbox.xmin) + 5, int(bbox.ymin) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),  # White text
            1,
            lineType=cv2.LINE_AA
        )

    return frame, class_colors

def run():
    # Create a channel and stub to connect to the server.
    channel = grpc.insecure_channel('localhost:50051')
    stub = yolo_pb2_grpc.YOLOServiceStub(channel)
    
    # Read the image file as bytes.
    video_path = sys.argv[1]

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    frames_bytes = []
    class_colors = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Encode the frame as JPEG
        success, encoded_image = cv2.imencode('.jpg', frame)
        if not success:
            print("Warning: Failed to encode frame, skipping.")
            continue

        # Convert the encoded image to bytes
        frame_bytes = encoded_image.tobytes()
    
        # Create the request with a list of images (here, just one).
        request = yolo_pb2.ProcessImagesRequest(images=[frame_bytes])
        
        # Call the ProcessImages RPC.
        response = stub.ProcessImages(request)
        
        # Print the results.
        print("Received YOLO results:")
        for i, result in enumerate(response.results):
            print(f"\nResult for image {i}:")
            if result.HasField("probs"):
                print("Embedding data:", result.probs.data)
                print("Embedding shape:", result.probs.shape)
            if result.bboxes:
                print("Bounding boxes:")
                frame, class_colors = draw_bboxes(frame=frame, bboxes=result.bboxes, class_colors=class_colors)
                for bbox in result.bboxes:
                    print(f"  Bbox: xmin={bbox.xmin}, ymin={bbox.ymin}, "
                        f"width={bbox.width}, height={bbox.height}, "
                        f"id={bbox.id}, confidence={bbox.confidence}")
            if result.keypoints:
                print("Keypoints:")
                for kp_set in result.keypoints:
                    for pt in kp_set.points:
                        print(f"  Point: x={pt.x}, y={pt.y}, confidence={pt.confidence}")
            if result.masks:
                print("Masks (byte lengths):", [len(mask) for mask in result.masks])
            cv2.imshow("Video Stream", frame)

            # Wait for 30ms and check if 'q' is pressed to quit.
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
    # Release resources.
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()
