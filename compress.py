# import os
import av
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor
import zarr
from numcodecs import Blosc
from tqdm import tqdm
from diffusers import AutoencoderKL
from transformers import CLIPTokenizer, CLIPModel
import json

class VideoProcessor:
    def __init__(self, config):
        self.config = config
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").cpu().eval()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cpu().eval()
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.to_tensor = Compose([ToTensor()])

    def encode_frames(self, frames, batch_size=20):
        # print(frames.shape)
        latents_list = []
        num_frames = frames.size(0)
        
        for i in range(0, num_frames, batch_size):
            batch = frames[i:i + batch_size]
            with torch.no_grad():
                latents = self.vae.encode(batch).latent_dist.sample() * self.vae.config.scaling_factor
            latents_list.append(latents.half().cpu().numpy())
        
        return np.concatenate(latents_list, axis=0)

    def get_clip_embedding(self, text):
        tokens = self.tokenizer(text, return_tensors="pt") 
        with torch.no_grad():
            embedding = self.clip.get_text_features(**tokens)
        # print(embedding.shape)
        return embedding.half().numpy()
    
    def random_crop(self, img, size, top, left):
        height, width, _ = img.shape
        target_h, target_w = size

        if height == target_h and width == target_w:
            return img

        return img[top:top + target_h, left:left + target_w, :]
    
    def get_random_crop_pos(self, frame_shape):
        height, width, _ = frame_shape
        crop_size = self.config['final_size']
        top = np.random.randint(0, height - crop_size + 1)
        left = np.random.randint(0, width - crop_size + 1)
        return top, left
    
    def process_segments_per_video(self, frames):
        num_segments = frames.shape[0]
        cropped_frames = []

        for segment_idx in range(num_segments):
            # random crop position per segment
            top, left = self.get_random_crop_pos(frames[segment_idx, 0].shape)

            # crop all frames in the current segment with same random crop position
            cropped_segment_frames = []
            for frame in frames[segment_idx]:
                frame_np = frame.cpu().numpy()
                crop_size = (self.config['final_size'], self.config['final_size'])
                cropped = self.random_crop(frame_np, crop_size, top, left)
                cropped_segment_frames.append(cropped)

            tensor_frames = torch.stack([self.to_tensor(frame) for frame in cropped_segment_frames])
            normalized = tensor_frames.float() / 255.0
            
            # VAE
            latents = self.encode_frames(normalized)
            cropped_frames.append(latents)
        return np.concatenate(cropped_frames, axis=0)



class VideoDataset(Dataset):
    def __init__(self, json_path, config):
        with open(json_path, 'r') as f:
            self.videos = json.load(f)
        self.config = config
        self.frames_per_segment = config['frames_per_segment']

    def __len__(self):
        return len(self.videos)

    def extract_frames(self, video_path):
        video = av.open(video_path)
        stream = video.streams.video[0]
        max_segments = self.config['max_segments'] 
        fps = self.config['fps']
        seg_duration = self.config['segment_duration']
        frames_per_seg = self.frames_per_segment
        total_frames = int(stream.frames)

        # Get linearly spaced frame indices
        frame_indices = []
        for seg in range(max_segments):
            start_frame = seg * seg_duration * fps
            if start_frame >= total_frames:
                break
                
            end_frame = min(total_frames, start_frame + frames_per_seg)
            indices = np.linspace(start_frame, end_frame-1, frames_per_seg, dtype=int)
            frame_indices.append(indices)
        
        if not frame_indices:
            return np.array([])
            
        frame_indices = np.concatenate(frame_indices)
        frames = []
        
        # batch process frames
        container = av.open(video_path)
        container.streams.video[0].thread_type = "AUTO" # multithreading
        
        for frame in container.decode(video=0):
            if frame.index in frame_indices:
                frames.append(frame.to_rgb().to_ndarray())
            if len(frames) == len(frame_indices):
                break
                
        frames = np.array(frames)
        height, width, channels = frames[0].shape
        num_segments = len(frame_indices) // frames_per_seg
        frames = frames[:num_segments * frames_per_seg] # prevent incomplete segment
        frames = frames.reshape(-1, frames_per_seg, height, width, channels)
        return frames

    def __getitem__(self, idx):
        video = self.videos[idx]
        frames = self.extract_frames(video['path'])
        return frames, video['idx'], video['action']

def setup_zarr_store(root, config):
    compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE) # compression type and strength

    frames = root.create_dataset(
        'base_frames',
        shape=(config['max_segments'], config['frames_per_segment'], config['final_size'] // 8, config['final_size'] // 8),
        chunks=(1, 20, config['final_size'] // 8, config['final_size'] // 8),
        dtype='float16',
        compressor=compressor
    ) # (segments, frames, h, w)

    with open(config['json_path']) as f:
        num_videos = len(set(v['idx'] for v in json.load(f)))
    # print(num_videos)

    embeddings = root.create_dataset(
        'clip_emb',
        shape=(num_videos, 512),
        chunks=(1, 512),
        dtype='float16',
        compressor=compressor
    ) # (videos, embedding_dim)

    mapping = root.create_dataset(
        'segment_to_video',
        shape=(config['max_segments'],),
        dtype='int64',
        chunks=(config['max_segments'],)
    ) # (segments,)

    return frames, embeddings, mapping


def main():
    store = zarr.DirectoryStore(config['output_zarr'])
    root = zarr.group(store, overwrite=True)
    frames_data, embeddings_data, mapping_data = setup_zarr_store(root, config)

    processor = VideoProcessor(config)
    dataset = VideoDataset(config['json_path'], config)
    loader = DataLoader(dataset, batch_size=1, num_workers=4)

    processed_videos = {}

    seg_count = 0
    for frames, video_id, caption in tqdm(loader, desc="Processing videos"):
        frames = frames[0]
        # print(frames.shape)
        latents = processor.process_segments_per_video(frames)
        if video_id not in processed_videos:
            embedding = processor.get_clip_embedding(caption)
            embeddings_data[video_id.item()] = embedding.squeeze()
            processed_videos[video_id.item()] = True

        for i in range(latents.shape[0]):
            frames_data[seg_count] = latents[i].numpy()
            mapping_data[seg_count] = video_id
            seg_count += 1

def test_frame_sampling():
    dataset = VideoDataset(config['json_path'], config)
    print(dataset.__getitem__(0)[0].shape)
    
def test_recon():
    # Create and save compressed data
    store = zarr.DirectoryStore(config['output_zarr'])
    root = zarr.group(store, overwrite=True)

    test_latents = torch.randn(20, 32, 32).half()
    frames_data, embeddings_data, mapping_data = setup_zarr_store(root, config)

    frames_data[0] = test_latents.numpy()
    mapping_data[0] = 0

    test_embedding = torch.randn(512).half().numpy()
    embeddings_data[0] = test_embedding

    del root
    del store

    # Read from compressed data
    store = zarr.DirectoryStore(config['output_zarr'])
    root = zarr.group(store)

    saved_latents = torch.from_numpy(root['base_frames'][0])
    saved_embedding = torch.from_numpy(root['clip_emb'][0])
    saved_mapping = root['segment_to_video'][0]

    # Recon
    if torch.allclose(saved_latents, test_latents, atol=1e-4) and \
        torch.allclose(saved_embedding, torch.from_numpy(test_embedding), atol=1e-4) \
            and saved_mapping == 0:
        print("Reconstruction test passed")
        print(f"Compression ratio: {test_latents.numpy().nbytes / root['base_frames'].nbytes}")
    else:
        print("Reconstruction test failed")

if __name__ == "__main__":
    config = {
        'json_path': "./data.json",
        'output_zarr': "./dataset.zarr", 
        'fps': 4,
        'segment_duration': 5,
        'final_size': 256,
        'batch_size': 4,
        'frames_per_segment': 20,
        'max_segments': 4
    }
    test_frame_sampling() # should output (4, 20, H, W, 3) (uncropped) or (max_segments, frames_per_segment, h, w, c). Cropping done in VideoProcessor
    test_recon()
