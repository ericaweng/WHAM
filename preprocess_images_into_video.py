import cv2
import os
import numpy as np

def frames_to_video(input_folder, output_file, fps):
    images = sorted([img for img in os.listdir(input_folder) if img.endswith(".jpg")])
    frame = cv2.imread(os.path.join(input_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(input_folder, image)))

    cv2.destroyAllWindows()
    video.release()


# Example usage
def main():

    # save each individual image videos
    cam_nums = [0,2,4,6,8]
    for num in cam_nums:
        for split in ['train', 'test']:
            cam_num_path = f'../PoseFormer/datasets/jackrabbot/{split}/images/image_{num}'
            scenes = os.listdir(cam_num_path)
            for scene in scenes:
                print(f"split: {split}; scene: {scene}; cam_num: {num}")
                images_path = f'{cam_num_path}/{scene}'
                output_file = f'../pose_forecasting/viz/wham_input_vids/{scene}_image_{num}.mp4'
                print(f"saving to: {output_file}")
                if not os.path.exists(os.path.dirname(output_file)):
                    os.makedirs(os.path.dirname(output_file))
                frames_to_video(images_path, output_file, 15)

    # save stitched image video
    for split in ['train', 'test']:
        cam_num_path = f'../PoseFormer/datasets/jackrabbot/{split}/images/image_stitched/'
        scenes = os.listdir(cam_num_path)
        for scene in scenes:
            print(f"split: {split}; scene: {scene}")#; cam_num: {num}")
            images_path = f'{cam_num_path}/{scene}'
            output_file = f'../pose_forecasting/viz/wham_input_vids/{scene}.mp4'
            print(f"saving to: {output_file}")
            if not os.path.exists(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file))
            frames_to_video(images_path, output_file, 15)



if __name__ == "__main__":
    main()
