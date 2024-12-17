# VideoDataLoader

### (All code and experiments were done to accomodate for CPUs. I did not have access to a GPU (even on Colab due to region restrictions I believe) so I could not experiment with direct-to-GPU methods or large chunk/batch sizes)

### Results:
- Processed 5000 segments in 1.48 seconds (Dummy Zarr Dataset made by random tensor values of desired shapes)
- dummy_dataset.zarr (5000 segments + text embeddings + index mapping) is 723M (my CPU constrains brings a tradeoff: chunking for 8 takes 1 - 1.5s more in loading but reduces memory by 60 MB. Chunk/batch size is a hyperparameter)
- Memory-speed tradeoff may behave differently when optimized for GPU transfer / large scale video training. (like pinned memory, vectorized operations, optimal chunks to minimize I/O processes)

- Average throughput: 2860.83 samples/second (low: 1816.13, high: 3369.24)
### 1. Video Processing Pipeline
- We break down each video into 5-second segments, sampled at 4 FPS (20 frames per segment)
- We can change the max number of segments per video.
- We uniformly sample frames from each video to create a dataset of segments. This is because videos have a lot of temporal redundancy.
- Random cropping to 256x256 pixels. I wrote a function manually so we do not use libraries which convert frames to PIL images and so we can have the same random crop per segment. This decreases latency.
- Each frame is encoded through Stable Diffusion's VAE to get compact latent representations of 32x32x4
- Text descriptions are converted to CLIP embeddings once per video.
- A map is made from segments to videos which is used to map segments to appropriate CLIP embeddings without having duplicates.
- Without mapping: (example video/segment numbers)
  Would need: 5000 segments × 512 dimensions × 2 bytes (float16) = ~5.12 MB for CLIP embeddings
  With mapping:
    CLIP embeddings: 100 videos × 512 dimensions × 2 bytes = ~102 KB
    Mapping array: 5000 segments × 8 bytes (int64) = ~40 KB
    Total: ~142 KB
- Stored all embeddings in float16

### 2. Storage Format
We use Zarr for our data compression (0.25 compression ratio on small data sample. Would increase for larger data sizes)
- `base_frames`: VAE latents for each video segment (shape: num_segments × 20 frames × latent_dims)
- `clip_emb`: CLIP embeddings of action descriptions (shape: num_videos × 512)
- `segment_to_video`: Mapping from segments back to their source videos (shape: num_segments,)

Why Zarr?
- Hierarchical structure where you can organize data into groups and subgroups. This allows for logical separation like base_frames, clip_emb, and segment_to_video, each as separate datasets
- We use Blosc with the Zstandard algorithm (zstd). zstd is known for its high compression ratio and speed, which is beneficial for numerical data
- Memory-mapped reading for efficient loading (inherent in zarr so no need for manual memmap with npy. It is equivalent and has zero-copy)
- Zarr's chunking mechanism is crucial as it allows random acess / easy indexing which is very beneficial for training purposes.
- Can make chunk = training batch size so we read small portions of data without loading the entire dataset into memory.
- Very beneficial for our training task. I set chunk to 1 which allows much faster loading but slightly lower compression. I also did 1 due to CPU constraints. Ideally chunk size should be batch size so we load a batch at a time at train-time. This would be optimized for the GPU to reduce I/O processes. Higher chunk = stronger compression.
- Although at CPU level I was operating at low chunk size (1), the chunking was very beneficial since I did not have to load everything into memory and could operate on 1 batch at a time efficiently.
- Chunks can be distributed across multiple nodes or accessed in parallel, improving performance in distributed systems/GPU/Large scale training.

### 3. Data Loading
- Utilize multiple CPU cores for data loading, reducing the latency between training steps
- Used shared memory to share data across processes without copying, which significantly reduces memory overhead and speeds up inter-process communication.
- Prefetching workers and queue to acoide idle times during training.
- Zarr's chunking allows for efficient random access and reading only what's necessary for each batch, reducing I/O operations. 
- Can have shuffling, pin memory (for GPU transfer) and compatible with PyTorch due to iterator features.
- Only create memory blocks once, pass memory references rather than copy data, allow workers to have independent connection to zarr
- Each worker process:
    -  Opens its own connection to the Zarr dataset
    -  Waits for batch segment IDs from the input queue
    -  Processes the data using shared memory
    -  Sends the shared memory information back through the output queue

### 4. Training Loop
Have implemented sample training run without model definition since I was limitied by my CPU (would hang on model initialization). Commented out model training etc. to do just loading speed profiling (equivalent to test_dataloading.py) and got (batch size = 1 segment):

Total training time: 2.96 seconds
Average time per segment: 0.30 ms
Segment per second: 3382.57

### Ideas for improvement
- I did uniform sampling sped up by closest keyframe decoding. Better sampling methods may be implemented by detected large differences in frames. Only keyframe sampling may be helpful. My implementation has an hidden benefit, though it allows maximum segments curated per video, increasing data quantity since it will also have a different random crop.
- Like MP4, I was inspired by inter-frame encoding like delta compression. I implemented it but did not get much better results. My hypothesis is that the temporal changes withing neighboring frames are more closer in nature in pixel-space than in VAE latent-space. Another reason may be that I uniformly sampled frames, and did not take neighboring ones. Further experiments might show smaller delta's even in VAE latent space. Also, delta encoding is most memory efficient with aggressive compression. Larger chunk sizes would cause easier pattern detection since delta's are zero to near-zero.
- Experimenting with chunk sizing, compression level (clevel) of zarr
- Very task dependent. If it is video classification, we can use much more sparser temporal sampling.
- bf16 if using H100s
- Different compression formats or vector quantization (dependent on downstream task)
