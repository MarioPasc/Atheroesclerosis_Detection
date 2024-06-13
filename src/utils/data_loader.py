import os
import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, base_path):
        self.base_path = base_path
        self.selected_videos_path = os.path.join(base_path, 'selectedVideos')
        self.patients_folders = np.sort(os.listdir(self.selected_videos_path))

    def get_file_paths(self, patient_path, filename):
        file_path = os.path.join(patient_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                return [line.strip() for line in file]
        return []

    def get_selected_frames(self, video_path, patient, video):
        selected_frames_path = os.path.join(video_path, f'{patient}_{video}_selectedFrames.txt')
        selected_frames = []
        if os.path.isfile(selected_frames_path):
            with open(selected_frames_path, 'r') as file:
                selected_frames = [os.path.join(video_path, 'input', line.strip() + '.png') for line in file]
        return selected_frames

    def get_groundtruth_files(self, video_path, lesion_videos):
        groundtruth_files = []
        groundtruth_map = {}
        if os.path.basename(video_path) in lesion_videos:
            groundtruth_path = os.path.join(video_path, 'groundtruth')
            if os.path.isdir(groundtruth_path):
                groundtruth_files = [os.path.join(groundtruth_path, f) for f in os.listdir(groundtruth_path) if f.endswith('.txt')]
                groundtruth_map = {os.path.basename(gt_file).split('.')[0]: gt_file for gt_file in groundtruth_files}
        return groundtruth_map

    def process_data(self):
        patient_list = []
        video_list = []
        frame_list = []
        selected_frames_non_lesion_video_list = []
        selected_frames_lesion_video_list = []
        groundtruth_file_list = []
        lesion_list = []

        for patient in self.patients_folders:
            patient_path = os.path.join(self.selected_videos_path, patient)
            video_folders = np.sort(os.listdir(patient_path))

            lesion_txt_path = os.path.join(patient_path, 'lesionVideos.txt')
            non_lesion_txt_path = os.path.join(patient_path, 'nonlesionVideos.txt')
            lesion_videos = self.get_file_paths(patient_path, 'lesionVideos.txt')
            non_lesion_videos = self.get_file_paths(patient_path, 'nonlesionVideos.txt')

            for video in lesion_videos + non_lesion_videos:
                video_path = os.path.join(patient_path, video)
                frames_path = os.path.join(video_path, 'input')
                selected_frames = self.get_selected_frames(video_path, patient, video)
                groundtruth_map = self.get_groundtruth_files(video_path, lesion_videos)

                for frame in selected_frames:
                    frame_id = os.path.basename(frame).split('_')[-1].split('.')[0]
                    groundtruth_file = groundtruth_map.get(f'{patient}_{video}_{frame_id}', '')
                    has_lesion = groundtruth_file != ''

                    patient_list.append(patient)
                    video_list.append(video)
                    frame_list.append(frame_id)
                    if video in lesion_videos:
                        selected_frames_lesion_video_list.append(frame)
                        selected_frames_non_lesion_video_list.append("")
                    else:
                        selected_frames_lesion_video_list.append("")
                        selected_frames_non_lesion_video_list.append(frame)
                    groundtruth_file_list.append(groundtruth_file)
                    lesion_list.append(has_lesion)

        data = {
            'Patient': patient_list,
            'Video': video_list,
            'Frame': frame_list,
            'SelectedFramesNonLesionVideo': selected_frames_non_lesion_video_list,
            'SelectedFramesLesionVideo': selected_frames_lesion_video_list,
            'GroundTruthFile': groundtruth_file_list,
            'Lesion': lesion_list
        }
        df = pd.DataFrame(data)
        df['video_paciente'] = df['Patient'] + '_' + df['Video']
        return df

    def save_to_csv(self, df, output_csv_path):
        df.to_csv(output_csv_path, index=False)

