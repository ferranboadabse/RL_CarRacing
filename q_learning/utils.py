import os
import cv2
import json
import pandas as pd

def read_json_files_from_folder(folder_path='results'):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                file_data = json.load(file)
                file_data['filename'] = filename
                data.append(file_data)
    return pd.DataFrame(data)


def make_video_frames(video_path):
    output_folder = 'results/frames_output'
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = int(total_frames / fps)

    frame_step = 30
    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_step == 0:
            # Calculate current second
            current_sec = int(frame_count / fps)
            timestamp_text = f"{current_sec} of {duration_sec}"

            # Put timestamp on frame (bottom-left corner)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_color = (255, 255, 255)  # white text
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(timestamp_text, font, font_scale, thickness)
            position = (10, 10 + text_height)

            # Add a black rectangle behind text for better visibility
            (text_width, text_height), baseline = cv2.getTextSize(timestamp_text, font, font_scale, thickness)
            cv2.rectangle(frame,
                        (position[0], position[1] - text_height - baseline),
                        (position[0] + text_width, position[1] + baseline),
                        (0, 0, 0),
                        thickness=cv2.FILLED)

            # Put the text on the frame
            cv2.putText(frame, timestamp_text, position, font, font_scale, font_color, thickness)

            frame_filename = os.path.join(output_folder, f'frame_{saved_count:03d}.png')
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Saved {saved_count} frames to '{output_folder}'")