import ctypes
import importlib
import logging
import os
import re
import subprocess
from datetime import datetime
from datetime import timedelta
import moviepy.video.fx.all as vfx
import numpy as np
import pandas as pd
import pyautogui as pya
import requests
import yagmail
from moviepy.editor import VideoFileClip, AudioFileClip
from pydub import AudioSegment

# from utils.sensitives import url_dict, crest_gmail, crest_app_pw
from utils.sensitives import url_dict

def get_project_path():
    path = os.getcwd()
    while os.path.basename(path) != 'catalyst':
        path = os.path.dirname(path)
    # if in win, replace \ with /
    path = path.replace('\\', '/')
    return path


def get_logger(exp_name, module_name):
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s[%(process)d] - %(name)s - %(levelname)s - %(message)s')
    # create log file if not exist
    log_dir = get_dir('log_dir', exp_name=exp_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    file_handler = logging.FileHandler(get_log_path(exp_name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def complement_value(parameters: dict):
    sum_value = np.sum(list(parameters.values()))
    comp = 1 - sum_value
    # set to 0 if it's too small or even a negative value
    comp = comp if comp > 0.002 else 0
    return round(comp, 3)


def turn_capslock_off():
    if ctypes.WinDLL("User32.dll").GetKeyState(0x14):
        pya.press("capslock")


def log_and_print(to_log: bool, logger: logging.Logger, log_level: str, msg: str, force_print: bool = False):
    if to_log:
        level_dict = {'debug': logger.debug, 'info': logger.info, 'warning': logger.warning, 'error': logger.error}
        level_dict[log_level](msg)
    if force_print or log_level in ['warning', 'error']:
        print(msg)


def config_loader(exp_name, config_type):
    assert config_type in [
        'ot2_deck_configs',
        'ot2_run_configs',
        'database_configs',
        'robot_test_configs',
        'al_configs'
    ]

    # load default config
    default_config = importlib.import_module(f'projects.default.config_files.{config_type}').configs

    # load project config
    try:
        project_config = importlib.import_module(f'projects.{exp_name}.config_files.{config_type}').configs
    except ModuleNotFoundError:
        project_config = {}

    # update default config with project config
    config = {**default_config, **project_config}

    return config


def get_tasks(exp_name, task_name):
    tasks = importlib.import_module(f'projects.{exp_name}.config_files.ot2_tasks_configs').configs[task_name]
    task_df = pd.DataFrame(columns=tasks['elements'], data=tasks['task_list'])
    print(f'{task_df}\n\n'
          f'Please double check task list as above')
    return task_df


def get_dir(dir_name, exp_name=None):
    assert dir_name in ['log_dir', 'icons_dir']
    if dir_name == 'log_dir':
        return os.path.expanduser(f"~/PycharmProjects/catalyst/projects/{exp_name}/logs")
    elif dir_name == 'icons_dir':
        return os.path.expanduser(f"~/PycharmProjects/catalyst/robotic_testing/icons")


def get_log_path(exp_name):
    return os.path.expanduser(f"~/PycharmProjects/catalyst/projects/{exp_name}/logs/{exp_name}.log")


def normalize_task(task: pd.DataFrame):
    task_norm = task.copy()
    for task_index, task_recipe in task_norm.iterrows():
        # skip if all 0
        if any(task_recipe) != 0:
            task_norm.iloc[task_index] = np.round(task_norm.iloc[task_index] / task_norm.iloc[task_index].sum(), 3)
    return task_norm


def get_latest_file_name(dir_path):
    """
    Get the last modified file name only in a directory
    """
    files = os.listdir(dir_path)
    paths = [os.path.join(dir_path, basename) for basename in files]
    full_path = max(paths, key=os.path.getmtime)
    return os.path.splitext(os.path.basename(full_path))[0]


def hyperlapse_video(file_name, input_path, output_path, input_type='mkv', output_type='mp4', acceleration_factor=100,
                     audio=False):
    """
    Hyperlapse a video to speed up the video by acceleration_factor
    """
    input_file = f'{os.path.join(input_path, file_name)}.{input_type}'
    output_file = f'{os.path.join(output_path, file_name)}_{acceleration_factor}x.{output_type}'

    clip = VideoFileClip(input_file)
    audio_clip = None

    if audio:
        audio_clip = clip.audio
        audio_clip.write_audiofile("temp_audio.wav")
        sound = AudioSegment.from_file("temp_audio.wav")
        sound = sound.speedup(playback_speed=acceleration_factor)
        sound.export("temp_audio_speedup.wav", format="wav")
        audio_clip = AudioFileClip("temp_audio_speedup.wav")
    else:
        clip = clip.without_audio()

    clip = clip.set_fps(30)

    final = clip.fx(vfx.speedx, acceleration_factor)

    if audio:
        final = final.set_audio(audio_clip)

    final.write_videofile(output_file, codec='h264')

    return output_file


# def email(email_address, subject, contents: list or str):
#     yag = yagmail.SMTP(crest_gmail, crest_app_pw)
#     yag.send(email_address, subject, contents)


def post_to(target, endpoint, data):
    response = requests.post(f'{url_dict[target]}/{endpoint}', json=data)
    return response


def print_gpt_process(content, gpt_process):
    assert gpt_process in ['msg_received', 'msg_replied', 'func_called', 'func_completed']
    mapping = {
        'msg_received': 'MESSAGE RECEIVED',
        'msg_replied': 'MESSAGE REPLIED',
        'func_called': 'FUNCTION CALLED',
        'func_completed': 'FUNCTION COMPLETED',
    }
    print(f"\n****{mapping[gpt_process]}****\n\n"
          f"{content}\n\n"
          f"-----------------------\n")


def get_counterpart(var):
    counterparts = {'start': 'stop', 'stop': 'start'}
    return counterparts.get(var, 'Invalid input')


def adb_pull_dir(adb_path, android_dir, local_dir):
    # Make sure android_dir ends with a /
    if not android_dir.endswith('/'):
        android_dir += '/'

    # Get a list of all files in the Android directory
    adb_ls_output = subprocess.check_output([adb_path, "shell", f"ls {android_dir}"]).decode("utf-8").strip()
    android_files = adb_ls_output.split("\n")

    # Remove hidden files and directories (starting with '.')
    android_files = [f for f in android_files if not f.startswith('.')]

    # Check which files already exist in the local directory
    local_files = os.listdir(local_dir)

    # Pull files that don't exist locally
    for file in android_files:
        if file not in local_files:
            subprocess.run([adb_path, "pull", f"{android_dir}{file}", f"{local_dir}/{file}"])


def adb_pull_files_after_timestamp(
        adb_path: str,
        android_dir: str,
        android_phone_id: str,
        local_dir: str,
        time_threshold: datetime,
        time_traceback: int = 180,
):
    """
    retrieve files from android phone after a certain time stamp minus a time traceback
    :param adb_path: adb.exe path
    :param android_dir: the directory on the android phone to retrieve files from
    :param android_phone_id: the id of the android phone
    :param local_dir: the local directory to store the retrieved files
    :param time_threshold: the time stamp to retrieve files after
    :param time_traceback: in seconds, retrieve files after time_threshold - time_traceback
    :return: a list of retrieved files
    """
    # Ensure the Android directory path ends with a '/'
    if not android_dir.endswith('/'):
        android_dir += '/'

    # Attempt to get a list of all files in the Android directory, sorted by last modification time
    try:
        cmd = [adb_path, '-s', android_phone_id, "shell", 'ls', '-lt', f'\"{android_dir}\"']
        adb_ls_output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode(
            "utf-8").strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to execute ADB command to list files: {e.output.decode('utf-8')}")

    # Parse the output of the 'ls' command to get file details
    lines = adb_ls_output.split("\n")
    retrieved_files = []

    for line in lines:
        parts = line.split()
        if len(parts) < 6:
            continue

        file_name = parts[-1]

        # Use regex to extract date and time from the file name
        match = re.match(r'(\d{8})_(\d{6})_', file_name)
        if match:
            date_str, time_str = match.groups()
            file_date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
            try:
                file_date = datetime.strptime(file_date_str, '%Y-%m-%d %H:%M:%S')
            except ValueError as e:
                print(f"Failed to parse date from file name {file_name}: {e}")
                continue  # Skip files that don't have the correct date format in the name

            # Check if file is newer than the timestamp minus the traceback time
            if file_date > time_threshold - timedelta(seconds=time_traceback) and file_name.endswith('.jpg'):
                file_name = parts[-1]
                file_path = f"{android_dir}{file_name}"

                # Pull the file if it's not in the local directory
                if file_name not in os.listdir(local_dir):
                    try:
                        subprocess.run(
                            [adb_path, "-s", android_phone_id, "pull", file_path, f"{local_dir}/{file_name}"],
                            check=True)
                        retrieved_files.append(f"{local_dir}/{file_name}")
                    except subprocess.CalledProcessError as e:
                        raise RuntimeError(f"Failed to execute ADB pull command for {file_name}: {e}")

    return retrieved_files


def get_latest_file_path(file_dir, time_threshold: datetime = None):
    latest_time = None
    latest_file = None

    # List all files in the directory
    try:
        files = os.listdir(file_dir)
    except FileNotFoundError:
        print(f"The directory {file_dir} was not found.")
        return None

    # Loop through each file to find the one with the latest time stamp after the provided threshold
    for file_name in files:
        full_path = os.path.join(file_dir, file_name)

        # Check if it's a file
        if not os.path.isfile(full_path):
            continue

        # Get the modification time of the file
        file_time = datetime.fromtimestamp(os.path.getmtime(full_path))

        # Compare the modification time with the provided time stamp, if any
        if time_threshold is None or file_time > time_threshold:
            if latest_time is None or file_time > latest_time:
                latest_time = file_time
                latest_file = full_path

    return latest_file
