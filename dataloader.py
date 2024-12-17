import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import zarr

def open_zarr_datasets(zarr_path):
    zarr_root = zarr.open(zarr_path, mode="r")
    frames = zarr_root["base_frames"]  # (num_segments, T, C, H, W) 
    embeddings = zarr_root["clip_emb"]  # (num_videos, emb_dim)
    segment_mapping = zarr_root["segment_to_video"]  # (num_segments,)
    # print(f"frames.shape: {frames.shape}")
    return frames, embeddings, segment_mapping

def data_loader_worker(input_queue, output_queue, zarr_path, pin_memory):
    frames, embeddings, segment_mapping = open_zarr_datasets(zarr_path)
    frame_memory = None
    embedding_memory = None

    try:
        while True:
            batch_segments = input_queue.get()
            if batch_segments is None:
                break
            
            # this is so we don't initialize locally and then in shared. we pre-allocate space
            frame_shape = (len(batch_segments), frames.shape[1], frames.shape[2], frames.shape[3], frames.shape[4])
            emb_shape = (len(batch_segments), embeddings.shape[1])
            
            # initialze embeddings directly in shared memory to prevent copying
            if frame_memory is None:
                frame_memory = shared_memory.SharedMemory(create=True, size=np.prod(frame_shape) * frames.dtype.itemsize)
            if embedding_memory is None:
                embedding_memory = shared_memory.SharedMemory(create=True, size=np.prod(emb_shape) * embeddings.dtype.itemsize)

            frame_batch = np.ndarray(frame_shape, dtype=frames.dtype, buffer=frame_memory.buf)
            embedding_batch = np.ndarray(emb_shape, dtype=embeddings.dtype, buffer=embedding_memory.buf)

            # retrieve segments from zarr and store in shared memory
            for i, seg_id in enumerate(batch_segments):
                frame_batch[i] = frames[seg_id]
                embedding_batch[i] = embeddings[segment_mapping[seg_id]]

            output_queue.put((frame_memory.name, frame_shape, frames.dtype, embedding_memory.name, emb_shape, embeddings.dtype))

    finally:
        # clean up
        for memory in (frame_memory, embedding_memory):
            if memory:
                memory.close()
                memory.unlink()

class VideoDataLoader:
    def __init__(self, zarr_path, segment_ids, batch_size=8, num_workers=2, prefetch_batches=2, pin_memory=False, shuffle=True):
        self.zarr_path = zarr_path
        self.segment_ids = np.array(segment_ids)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_batches = prefetch_batches
        self.pin_memory = pin_memory
        self.shuffle = shuffle

        frames, embeddings, _ = open_zarr_datasets(zarr_path)
        _, self.time_steps, self.channels, self.height, self.width = frames.shape
        _, self.embedding_size = embeddings.shape
        self.num_batches = (len(self.segment_ids) + batch_size - 1) // batch_size

        # Set up multiprocessing queues for async prefetching
        self.input_queue = mp.Queue(maxsize=prefetch_batches)
        self.output_queue = mp.Queue()

        self.workers = []
        for _ in range(num_workers):
            worker = mp.Process(target=data_loader_worker, args=(self.input_queue, self.output_queue, zarr_path, pin_memory), daemon=True)
            self.workers.append(worker)

        for worker in self.workers:
            worker.start()

        self.current_batch = 0
        if shuffle:
            np.random.shuffle(self.segment_ids)
        self.prefetch_data()

    # Prefetch next batch
    def prefetch_data(self):
        for _ in range(self.prefetch_batches):
            if self.current_batch < self.num_batches:
                start_idx = self.current_batch * self.batch_size
                end_idx = start_idx + self.batch_size
                # add segment ids to queue for workers to retrieve
                self.input_queue.put(self.segment_ids[start_idx:end_idx])
                self.current_batch += 1

    def __iter__(self):
        self.current_batch = 0
        if self.shuffle:
            np.random.shuffle(self.segment_ids)
        self.prefetch_data()
        return self

    def __next__(self):
        if self.current_batch >= self.num_batches and self.output_queue.empty():
            raise StopIteration

        # get shared memory info from output queue  
        frame_name, frame_shape, frame_dtype, emb_name, emb_shape, emb_dtype = self.output_queue.get()

        frame_memory = shared_memory.SharedMemory(name=frame_name)
        embedding_memory = shared_memory.SharedMemory(name=emb_name)

        frames = np.ndarray(frame_shape, dtype=frame_dtype, buffer=frame_memory.buf)
        embeddings = np.ndarray(emb_shape, dtype=emb_dtype, buffer=embedding_memory.buf)

        frame_memory.close()
        embedding_memory.close()

        # next batch
        if self.current_batch < self.num_batches:
            start_idx = self.current_batch * self.batch_size
            end_idx = start_idx + self.batch_size
            self.input_queue.put(self.segment_ids[start_idx:end_idx])
            self.current_batch += 1

        return frames, embeddings

    def shutdown(self):
        # Clean up
        for queue in (self.input_queue, self.output_queue):
            while not queue.empty():
                try:
                    queue.get_nowait()
                except:
                    pass

        for _ in self.workers:
            self.input_queue.put(None)
        
        for worker in self.workers:
            worker.join()
        
        self.input_queue.close()
        self.output_queue.close()
