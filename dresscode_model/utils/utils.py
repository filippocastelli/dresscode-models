from pathlib import Path
import torch

def save_checkpoint(model: torch.Module, save_path: Path, use_cuda: bool = True):

    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True)

    torch.save(model.cpu().state_dict(), save_path)
    if use_cuda:
        model.cuda()


def load_checkpoint(model: torch.Module, checkpoint_path: Path, use_cuda: bool = True):

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
    
    log = model.load_state_dict(torch.load(checkpoint_path), strict=False)
    if use_cuda:
        model.cuda()