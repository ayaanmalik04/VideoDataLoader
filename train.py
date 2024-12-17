import torch
import torch.nn as nn
import numpy as np
import time
from dataloader import VideoDataLoader, open_zarr_datasets
import psutil 

# dummy model to convert frames to CLIP text embedding
class DummyModel(nn.Module):
    def __init__(self, frames_dim=20*32*32*3, embedding_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(frames_dim, frames_dim), nn.ReLU(), nn.Linear(frames_dim, embedding_dim))
        self.criterion = nn.MSELoss()
        self.half()
        
    def forward(self, frames, clip_embeddings):
        # [B, T, H, W, C]
        batch_size = frames.shape[0]
        x = frames.reshape(batch_size, -1)
        x = self.encoder(x)
        loss = self.criterion(x, clip_embeddings)
        return loss

class Trainer:
    def __init__(self, zarr_path, num_epochs=2):
        print(f"Initializing training loop with {num_epochs} epochs...")
        self.model = DummyModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.zarr_path = zarr_path
        self.num_epochs = num_epochs
        
        frames, _, _ = open_zarr_datasets(zarr_path)
        self.num_segments = len(frames)
        self.segment_ids = np.arange(self.num_segments)

        self.dataloader = VideoDataLoader(zarr_path=zarr_path, segment_ids=self.segment_ids, batch_size=1, num_workers=4, prefetch_batches=2, pin_memory=False, shuffle=True)

    def train(self):
        total_time = 0
        total_batches = 0
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_start = time.time()
            
            for batch_idx, (frames, clip_embeddings) in enumerate(self.dataloader):
                batch_start = time.time()
                frames = torch.from_numpy(frames).half()
                clip_embeddings = torch.from_numpy(clip_embeddings).half()
                
                self.optimizer.zero_grad()
                loss = self.model(frames, clip_embeddings)
                loss.backward()
                self.optimizer.step()
                
                batch_time = time.time() - batch_start
            
            epoch_time = time.time() - epoch_start
            total_time += epoch_time
            total_batches += len(self.dataloader)
            
        avg_time_per_batch = total_time / total_batches
        print(f"{total_time:.2f}")
        print(f"{avg_time_per_batch*1000:.2f}")
        print(f"{1.0/avg_time_per_batch:.2f}")
                
        self.dataloader.shutdown()

if __name__ == "__main__":
    trainer = Trainer(zarr_path="dummy_dataset.zarr", num_epochs=2)
    trainer.train()