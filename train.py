import os
import time
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from utils import logging_utils
from utils.YParams import YParams
from utils import get_data_loader  # Updated to accept a 'distributed' flag.
from utils.loss import l2_loss
from utils.metrics import weighted_rmse
from utils.plots import generate_images
from networks import vit

# Configure logging
logging_utils.config_logger()

def train(params, args):
    import torch.distributed as dist

    # Use the rank from the distributed process group, if initialized.
    rank = dist.get_rank() if dist.is_initialized() else 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    # Enable cuDNN autotuner for fixed input sizes.
    torch.backends.cudnn.benchmark = True

    # Initialize data loaders with distributed samplers.
    logging.info("Initializing data loaders")
    train_data_loader, _ = get_data_loader(params, params.train_data_path, train=True, distributed=True)
    val_data_loader, _ = get_data_loader(params, params.valid_data_path, train=False, distributed=True)
    logging.info("Data loaders initialized")

    # Create model and move it to the designated GPU.
    model = vit.ViT(params).to(device)
    # Wrap model with DistributedDataParallel (DDP).
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = optim.AdamW(model.parameters(), lr=params.lr, betas=(0.9, 0.95))

    logging.info("Model architecture:\n%s", model)

    # Set up learning rate scheduler.
    if params.lr_schedule == "cosine":
        if params.warmup > 0:
            lr_scale = lambda x: min((x + 1) / params.warmup,
                                     0.5 * (1 + np.cos(np.pi * x / params.num_iters)))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.num_iters)
    else:
        scheduler = None

    # Set up AMP scaler for mixed precision training.
    scaler = torch.cuda.amp.GradScaler()

    logging.info("Starting Training Loop...")

    # Only rank 0 performs initial logging.
    if rank == 0:
        model.eval()
        with torch.no_grad():
            inp, tar = next(iter(train_data_loader))
            inp, tar = inp.to(device, non_blocking=True), tar.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                gen = model(inp)
                tr_loss = l2_loss(gen, tar)

            inp, tar = next(iter(val_data_loader))
            inp, tar = inp.to(device, non_blocking=True), tar.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                gen = model(inp)
                val_loss = l2_loss(gen, tar)
                val_rmse = weighted_rmse(gen, tar)

            args.tboard_writer.add_scalar("Loss/train", tr_loss.item(), 0)
            args.tboard_writer.add_scalar("Loss/valid", val_loss.item(), 0)
            args.tboard_writer.add_scalar("RMSE/valid", val_rmse.cpu().numpy()[0], 0)

    params.num_epochs = params.num_iters // len(train_data_loader)
    iters = 0
    start_time = time.time()

    for epoch in range(params.num_epochs):
        # Update the sampler epoch for proper shuffling.
        if hasattr(train_data_loader.sampler, "set_epoch"):
            train_data_loader.sampler.set_epoch(epoch)

        torch.cuda.reset_peak_memory_stats(device)

        start_epoch = time.time()
        model.train()
        epoch_losses = []
        epoch_sample_count = 0  # Count the number of samples processed in this epoch.

        for data in train_data_loader:
            iters += 1
            inp, tar = data
            inp = inp.to(device, non_blocking=True)
            tar = tar.to(device, non_blocking=True)
            epoch_sample_count += inp.shape[0]

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                gen = model(inp)
                loss = l2_loss(gen, tar)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(loss.item())

            if scheduler is not None:
                scheduler.step()

        epoch_time = time.time() - start_epoch
        avg_loss = np.mean(epoch_losses)
        throughput = epoch_sample_count / epoch_time  # samples per second

        max_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        if rank == 0:
            logging.info("Epoch %d: Loss = %.6f, Time = %.2f sec, Throughput = %.2f samples/sec, Max GPU Mem = %.2f MB",
                         epoch+1, avg_loss, epoch_time, throughput, max_mem_mb)
            args.tboard_writer.add_scalar("Loss/train", avg_loss, iters)
            args.tboard_writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], iters)
            args.tboard_writer.add_scalar("Throughput/train", throughput, iters)
            args.tboard_writer.add_scalar("GPU_Memory/Max", max_mem_mb, iters)

            # Validation loop.
            model.eval()
            val_losses = []
            val_rmse_total = 0.0
            valid_steps = 0
            with torch.no_grad():
                for data in val_data_loader:
                    inp, tar = data
                    inp = inp.to(device, non_blocking=True)
                    tar = tar.to(device, non_blocking=True)
                    with torch.cuda.amp.autocast():
                        gen = model(inp)
                        loss_val = l2_loss(gen, tar)
                        rmse_val = weighted_rmse(gen, tar)
                    val_losses.append(loss_val.item())
                    val_rmse_total += rmse_val.cpu().numpy()[0]
                    valid_steps += 1
            avg_val_loss = np.mean(val_losses)
            avg_val_rmse = val_rmse_total / valid_steps
            logging.info("Validation: Loss = %.6f, RMSE = %.6f", avg_val_loss, avg_val_rmse)
            args.tboard_writer.add_scalar("Loss/valid", avg_val_loss, iters)
            args.tboard_writer.add_scalar("RMSE/valid", avg_val_rmse, iters)

    total_time = time.time() - start_time
    if rank == 0:
        logging.info("Training completed in %.2f sec", total_time)


if __name__ == "__main__":
    import torch.distributed as dist

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default="00", type=str, help="tag for current experiment")
    parser.add_argument("--yaml_config", default="./config/ViT.yaml", type=str, help="path to yaml config")
    parser.add_argument("--config", default="base", type=str, help="name of config in YAML")
    parser.add_argument("--local_batch_size", default=None, type=int, help="local batch size")
    parser.add_argument("--num_iters", default=None, type=int, help="number of iterations")

    args = parser.parse_args()
    run_num = args.run_num
    params = YParams(os.path.abspath(args.yaml_config), args.config)

    # Determine world size and local rank from environment variables.
    # If WORLD_SIZE > 1, initialize the distributed process group.
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
    else:
        rank = 0

    # Set local batch size based on provided arguments or config.
    if args.local_batch_size is not None:
        params.local_batch_size = args.local_batch_size
        params.update({"global_batch_size": args.local_batch_size * world_size})
    else:
        # Assume global_batch_size is specified in the config.
        params.local_batch_size = params.global_batch_size // world_size

    # Only rank 0 sets up experiment directories and TensorBoard logging.
    if rank == 0:
        expDir = os.path.join(params.expdir, args.config, run_num)
        if not os.path.isdir(expDir):
            os.makedirs(expDir)
        logging_utils.log_to_file(log_filename=os.path.join(expDir, "out.log"))
        params.log()
        args.tboard_writer = SummaryWriter(log_dir=os.path.join(expDir, "logs/"))
        params.experiment_dir = os.path.abspath(expDir)
    else:
        args.tboard_writer = None

    train(params, args)

    if rank == 0:
        logging.info("DONE")

