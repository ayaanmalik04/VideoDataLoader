### Results:
- Processed 5000 samples in 1.48 seconds (Dummy Zarr Dataset made by random tensor values of desired shapes)
- dummy_dataset.zarr (5000 segments + text embeddings + index mapping) is 723M (my CPU constrains brings a tradeoff: chunking for 8 takes 1 - 1.5s more in loading but reduces memory by 60 MB. chunk/batch size is a hyperparameter)
- Memory-speed tradeoff may behave differently when optimized for GPU transfer / large scale video training.

- Average throughput: 3369.24 samples/second
### 1. Video Processing Pipeline
- We break down each video into 5-second segments, sampled at 4 FPS (20 frames per segment)
- You can change the max number of segments per video.
- We uniformly sample frames from each video to create a dataset of segments. This is because videos have a lot of temporal redundancy.
- We us seek to decode keyframes close to uniformly sampled frames for faster and memory efficient decoding.
- Random cropping to 256x256 pixels. I wrote a function manually so we do not use libraries which convert frames to PIL images and so we can have the same random crop per segment. This decreases latency.
- Each frame is encoded through Stable Diffusion's VAE to get compact latent representations of 32x32x4
- Text descriptions are converted to CLIP embeddings once per video.
- A map is made from segments to videos. which is used to map segments to appropriate CLIP embeddings without having duplicates.
- Stored all embeddings in float16

### 2. Storage Format
We use Zarr for our data compression
- `base_frames`: VAE latents for each video segment (shape: num_segments × 20 frames × latent_dims)
- `clip_emb`: CLIP embeddings of action descriptions (shape: num_videos × 512)
- `segment_to_video`: Mapping from segments back to their source videos

Why Zarr?
-  hierarchical structure where you can organize data into groups and subgroups. This allows for logical separation like base_frames, clip_emb, and segment_to_video, each as separate datasets under a root group or even in different groups if needed.
- We use Blosc with the Zstandard algorithm (zstd).zstd is known for its high compression ratio and speed, which is particularly beneficial for numerical data
- Memory-mapped reading for efficient loading (inherent in zarr so need for manual memmap with npy. It is equivalent and has zero-copy)
- Zarr's chunking mechanism is crucial as it allows random acess / easy indexing.
- Can make chunk = training batch size so we read small portions of data without loading the entire dataset into memory.
- Very beneficial for our training task. I set chunk to 1 which allows much faster loading but slightly more memory. I also did 1 due to CPU constraints. Ideally chunk size should be batch size so we load a batch at a time. Higher chunk = stronger compression.
Load Balancing: Chunks can be distributed across multiple  nodes or accessed in parallel, improving performance in distributed systems.


### 3. Data Loading
- Utilize multiple CPU cores for data loading, reducing the latency between training steps
- Used shared memory to share data across processes without copying, which significantly reduces memory overhead and speeds up inter-process communication.
- Prefetching workers and queue to acoide idle times during training.
- Zarr's chunking allows for efficient random access and reading only what's necessary for each batch, reducing I/O operations. 
- Can have shuffling, pin memory (for GPU transfer) and compatible with PyTorch due to iterator features.

### 4. Training Loop
Have implemented sample training run without model definition.