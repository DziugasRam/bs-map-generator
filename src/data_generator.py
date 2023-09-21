import tensorflow as tf
import numpy as np
from tqdm import tqdm
import glob
import random
from data_loader import full_load_map
from concurrent.futures import ProcessPoolExecutor
import traceback


def data_generator_multi_process(map_folders):
    map_folders = [map_folder.decode('UTF-8') for map_folder in map_folders]
    max_workers = 18
    items_in_queue = max_workers * 5
    queued_maps = items_in_queue
    cancel = False
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        map_tasks = [(executor.submit(full_load_map, map_folder), map_folder) for map_folder in map_folders[:items_in_queue]]
        while len(map_tasks) > 0:
            map_task, map_folder = map_tasks.pop(0)
            try:
                if cancel:
                    map_task.cancel()
                    continue
                results = map_task.result()
                for result in results:
                    x_context_prev_audio, x_context_prev_notes, x_context_audio, y_context_notes, z_timing_counts, z_note_counts, z_note_pos_counts, z_acc_prediction, z_speed_prediction = result
                    yield (x_context_prev_audio), (x_context_prev_notes), (x_context_audio), z_timing_counts, z_note_counts/20, z_note_pos_counts/10, z_acc_prediction, z_speed_prediction, (y_context_notes)
            except InterruptedError as ke:
                cancel = True
            except Exception as exc:
                pass
                # if str(exc) != "'_version'" and str(exc) != 'not v2':
                    # print(map_folder)
                    # print(exc)
                    # traceback.print_exc()
            finally:
                if not cancel:
                    queued_maps += 1
                    if queued_maps < len(map_folders):
                        map_tasks.append((executor.submit(full_load_map, map_folders[queued_maps]), map_folders[queued_maps]))
                        
def create_ds_for_files(map_folders, batch_size, cache_name, cache=False, shuffle=False):
    ds = tf.data.Dataset.from_generator(data_generator_multi_process, args=[map_folders], output_signature=(
        tf.TensorSpec(shape=(None, 2, 87, 129), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 2, 40, 25), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 2, 87, 129), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.int32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 2, 40, 25), dtype=tf.float32),
        # tf.TensorSpec(shape=(None, 1025, 44), dtype=tf.float32),
        # tf.TensorSpec(shape=(None, 35), dtype=tf.float32),
    ))
    ds = ds.flat_map(lambda x1, x2, x3, x4, x5, x6, x7, x8, y: tf.data.Dataset.from_tensor_slices((x1, x2, x3, x4, x5, x6, x7, x8, y)))
    ds = ds.prefetch(20000)

    if cache:
        # ds = ds.cache()
        ds = ds.cache(f"./dataset_cache/{cache_name}")
    if shuffle:
        ds = ds.shuffle(25000, reshuffle_each_iteration=True)
        # ds = ds.shuffle(len([v for v in ds]), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(256)
    return ds


def create_datasets(batch_size=64, val_split=0.1, random_seed=1470258369):
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    maps = [path.replace("\\", "/") for path in glob.glob("../data/maps/*")]
    random.shuffle(maps)
    # I currently cache the entire dataset, since the data loading part is quite compute intensive. Added a limit of 50 maps to avoid running out of ram on a test run.
    # maps = maps[:50]
    
    train_ds = create_ds_for_files(maps[int(len(maps)*val_split):], batch_size, "train", False, True)
    val_ds = create_ds_for_files(maps[:int(len(maps)*val_split)], batch_size, "val", False, False)
    return train_ds, val_ds