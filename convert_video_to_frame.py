import os
import skvideo.io
from skimage.transform import resize
from skimage.io import imsave

video_root_path = './data/videos'
size = (224, 224)

def video_to_frame(dataset, train_or_test):
    video_path = os.path.join(video_root_path, dataset, '{}_videos'.format(train_or_test))
    frame_path = os.path.join(video_root_path, dataset, '{}_frames'.format(train_or_test))
    print('video_path = ', video_path)
    print('frame_path = ', frame_path)
    os.makedirs(frame_path, exist_ok=True)

    for video_file in os.listdir(video_path):
        if video_file.lower().endswith(('.avi', '.mp4')):
            print('==> ' + os.path.join(video_path, video_file))
            vid_frame_path = os.path.join(frame_path, os.path.basename(video_file).split('.')[0])
            os.makedirs(vid_frame_path, exist_ok=True)
            vidcap = skvideo.io.vreader(os.path.join(video_path, video_file))
            count = 1
            for image in vidcap:
              image = resize(image, size, mode='reflect')
              imsave(os.path.join(vid_frame_path, '{:05d}.jpg'.format(count)), image)     # save frame as JPEG file
              count += 1

# convert
video_to_frame('avenue', 'training')
video_to_frame('avenue', 'testing')
