import threading
import time

import numpy as np

from src.io import video_to_array


def import_labels(f):
    ''' Read from a file all the labels from it '''
    lines = f.readlines()
    labels = []
    i = 0
    for l in lines:
        t = l.split('\t')
        assert int(t[0]) == i
        label = t[1].split('\n')[0]
        labels.append(label)
        i += 1
    return labels

def to_categorical(y, nb_classes=None):
    ''' Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.
    '''
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y

def generate_output(video_info, labels, length=16):
    ''' Given the info of the vide, generate a vector of classes corresponding the output for each
    clip of the video which features have been extracted.
    '''
    nb_frames = video_info['num_frames']
    duration = video_info['duration']
    last_first_name = nb_frames - length + 1

    clips = np.array([range(0, last_first_name, length),
                      range(length, nb_frames, length)]).T
    # Get annotations
    num_annotations = len(video_info['annotations'])
    intersection = np.zeros([num_annotations, clips.shape[0]])
    target_labels = np.zeros(num_annotations, dtype=int)
    for i, annotation in enumerate(video_info['annotations']):
        t1 = annotation['segment'][0] * nb_frames / duration
        t2 = annotation['segment'][1] * nb_frames / duration
        # Length of intersection of clips with GT
        it1 = np.maximum(clips[:, 0], t1)
        it2 = np.minimum(clips[:, 1], t2)
        intersection[i, :] = (it2 - it1 + 1.0).clip(0)
        target_labels[i] = labels.index(annotation['label'])
    # Assign label with max intersection (in the rare event of multiple
    # matching annotations per clip)
    idx = np.argmax(intersection, axis=0)
    clip_labels = target_labels[idx]
    # Background instances
    clip_labels[np.max(intersection, axis=0) < length/2] = 0
    instances = clip_labels.tolist()
    return instances


class VideoGenerator(object):

    def __init__(self, videos, stored_videos_path,
            stored_videos_extension, length, input_size):
        self.videos = videos
        self.total_nb_videos = len(videos)
        self.flow_generator = self._flow_index(self.total_nb_videos)
        self.lock = threading.Lock()
        self.stored_videos_path = stored_videos_path
        self.stored_videos_extension = stored_videos_extension
        self.length = length
        self.input_size = input_size

    def _flow_index(self, total_nb_videos):
        pointer = 0
        while pointer < total_nb_videos:
            pointer += 1
            yield pointer-1

    def next(self):
        with self.lock:
            index = next(self.flow_generator)
        t1 = time.time()
        video_id = self.videos[index]
        path = self.stored_videos_path + '/' + video_id + '.' + self.stored_videos_extension
        vid_array = video_to_array(path, start_frame=0,
                                   resize=self.input_size)
        if vid_array is not None:
            vid_array = vid_array.transpose(1, 0, 2, 3)
            nb_frames = vid_array.shape[0]
            nb_instances = nb_frames // self.length
            vid_array = vid_array[:nb_instances*self.length,:,:,:]
            vid_array = vid_array.reshape((nb_instances, self.length, 3,)+(self.input_size))
            vid_array = vid_array.transpose(0, 2, 1, 3, 4)
        t2 = time.time()
        print('Time to fetch {} video: {:.2f} seconds'.format(video_id, t2-t1))
        return video_id, vid_array

    def __next__(self):
        self.next()
