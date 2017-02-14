import argparse
import copy
import json
import os

import h5py
from progressbar import ProgressBar

from src.data import import_labels
from src.processing import activity_localization, get_classification, smoothing

SUBSETS = ['training', 'validation', 'testing']


def template_submission(subset):
    dict_template = dict(version="VERSION 1.3", results={},
                         external_data={"details": subset, "used": False})
    return dict_template


def process_prediction(experiment_id, predictions_path, output_path, smoothing_k, activity_threshold, subset=None, epoch=0):
    clip_length = 16.

    if subset == None:
        subsets = SUBSETS[1:]
    else:
        subsets = subset

    predictions_file = os.path.join(
        predictions_path,
        'predictions_{experiment_id}_{epoch}.hdf5'.format(experiment_id=experiment_id, epoch=epoch)
    )

    with open('dataset/labels.txt', 'r') as f:
        labels = import_labels(f)
    with open('dataset/videos.json', 'r') as f:
        videos_info = json.load(f)

    f_predictions = h5py.File(predictions_file, 'r')
    for subset in subsets:
        print('Generating results for {} subset...'.format(subset))
        subset_predictions = f_predictions[subset]

        progbar = ProgressBar(max_value=len(subset_predictions.keys()))
        results_classification = template_submission(subset)
        results_detection = template_submission(subset)

        count = 0
        progbar.update(0)
        for video_id in subset_predictions.keys():
            old_video_id = video_id[2:]
            prediction = subset_predictions[video_id][...]
            video_info = videos_info[old_video_id]
            fps = float(video_info['num_frames']) / video_info['duration']

            # Post processing to obtain the classification
            labels_idx, scores = get_classification(prediction, k=5)
            result_classification = []
            for idx, score in zip(labels_idx, scores):
                label = labels[idx]
                if score > 0:
                    result_classification.append({
                        'score': score,
                        'label': label
                    })
            results_classification['results'][old_video_id] = result_classification

            # Post Processing to obtain the detection
            prediction_smoothed = smoothing(prediction, k=smoothing_k)
            activities_idx, startings, endings, scores = activity_localization(
                prediction_smoothed,
                activity_threshold
            )
            result_detection = []
            for idx, s, e, score in zip(activities_idx, startings, endings, scores):
                label = labels[idx]
                result_detection.append({
                    'score': score,
                    'segment': [
                        s * clip_length / fps,
                        e * clip_length / fps
                    ],
                    'label': label
                })
            results_detection['results'][old_video_id] = result_detection

            count += 1
            progbar.update(count)
        progbar.finish()

        classification_output_file = os.path.join(
            output_path,
            'results_classification_{}_{}_{}.json'.format(experiment_id, subset, epoch)
        )
        detection_output_file = os.path.join(
            output_path,
            'results_detection_{}_{}_{}.json'.format(experiment_id, subset, epoch)
        )
        with open(classification_output_file, 'w') as f:
            json.dump(results_classification, f, sort_keys=True, indent=4)
        with open(detection_output_file, 'w') as f:
            json.dump(results_detection, f, sort_keys=True, indent=4)

    f_predictions.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Post-process the prediction of the RNN to obtain the classification and temporal localization of the videos activity')

    parser.add_argument('--id', dest='experiment_id', default=0, help='Experiment ID to track and not overwrite results')
    parser.add_argument('-p', '--predictions-path', type=str, dest='predictions_path', default='data/dataset', help='Path where the predictions file is stored (default: %(default)s)')
    parser.add_argument('-o', '--output-path', type=str, dest='output_path', default='data/dataset', help='Path where is desired to store the results (default: %(default)s)')

    parser.add_argument('-k', type=int, dest='smoothing_k', default=5, help='Smoothing factor at post-processing (default: %(default)s)')
    parser.add_argument('-t', type=float, dest='activity_threshold', default=.2, help='Activity threshold at post-processing (default: %(default)s)')

    parser.add_argument('-s', '--subset', type=str, dest='subset', default=None, nargs='+', choices=SUBSETS, help='Subset you want to predict the output (default: validation and testing)')
    parser.add_argument('-e', '--epoch', dest='epoch', default='0', help='Epoch identifier (no formatting behind scenes!)')

    args = parser.parse_args()

    process_prediction(**vars(args))
