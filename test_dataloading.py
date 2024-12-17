import zarr
import numpy as np
import time
from dataloader import VideoDataLoader
import time
import numpy as np
import zarr

def create_dummy_dataset(output_path, num_segments=50000, num_videos=100, temporal_dim=20, channels=4, height=32, width=32, embedding_dim=512):
    # create a dummy dataset to test dataloader
    store = zarr.DirectoryStore(output_path)
    root = zarr.group(store=store, overwrite=True)
    
    compressor = zarr.Blosc(cname='zstd', clevel=5, shuffle=1)
    
    frames = root.create_dataset(
        'base_frames',
        shape=(num_segments, temporal_dim, channels, height, width),
        chunks=(1, temporal_dim, channels, height, width),
        dtype=np.float16,
        compressor=compressor
    )

    for i in range(num_segments):
        segment_frames = np.random.randn(temporal_dim, channels, height, width).astype(np.float16)
        frames[i] = segment_frames
        
        if i % 100 == 0:
            print(f"Progress: {i}/{num_segments} segments")

    clip_emb = root.create_dataset(
        'clip_emb',
        shape=(num_videos, embedding_dim),
        chunks=(1, embedding_dim),
        dtype=np.float16,
        compressor=compressor
    )
    clip_emb[:] = np.random.randn(num_videos, embedding_dim).astype(np.float16)
    
    segment_to_video = root.create_dataset(
        'segment_to_video',
        shape=(num_segments,),
        dtype=np.int64,
        compressor=compressor
    )
    segment_to_video[:] = np.random.randint(0, num_videos, size=num_segments)
    return output_path

def test_dataloader(zarr_path, num_segments = 5000, batch_size = 32, num_workers = 4):
    segment_ids = np.arange(num_segments)
    loader = VideoDataLoader(zarr_path=zarr_path, segment_ids=segment_ids, batch_size=batch_size, num_workers=num_workers, prefetch_batches=2, shuffle=True)
    start_time = time.time()
    num_batches = 0
    
    try:
        for i, (latents, clip_emb) in enumerate(loader):
            num_batches += 1
            
        total_samples = num_batches * batch_size
        elapsed = time.time() - start_time
        print(f"{total_samples} samples in {elapsed} seconds")
        print(f"{total_samples/elapsed} samples/second")
        
    finally:
        loader.shutdown()

if __name__ == "__main__":
    DATASET_PATH = "dummy_dataset.zarr"
    num_segments = 5000 # modify
    create_dummy_dataset(output_path=DATASET_PATH, num_segments=num_segments, \
                          num_videos=100, temporal_dim=20, channels=4, height=32, width=32, embedding_dim=512)
    # import cProfile
    # import pstats
    # import io
    # pr = cProfile.Profile()
    # pr.enable()
    test_dataloader( zarr_path=DATASET_PATH, num_segments=num_segments, batch_size=4,num_workers=2)
    # pr.disable()
    
    # s = io.StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    # ps.print_stats(20)  # Print the top 20 time-consuming functions
    # print(s.getvalue())
