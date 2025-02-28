import os
import time
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import logging
from utils import logging_utils
from utils.YParams import YParams
from utils import get_data_loader  # simpler, non-distributed data loader
from utils.loss import l2_loss
from utils.metrics import weighted_rmse
from utils.plots import generate_images
from networks import vit

# Configure logging
logging_utils.config_logger()

def train(params, args):
    # Enable cuDNN autotuner for fixed input sizes
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize data loaders
    logging.info("Initializing data loaders")
    train_data_loader, _ = get_data_loader(params, params.train_data_path, train=True)
    val_data_loader, _ = get_data_loader(params, params.valid_data_path, train=False)
    logging.info("Data loaders initialized")

    # Create model and move to device
    model = vit.ViT(params).to(device)
    # Wrap the model with torch.compile for further optimizations
    model = torch.compile(model)

    # Using the built-in fused Adam optimizer for improved performance
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params.lr,
        betas=(0.9, 0.95),
        fused=True  # Enable fused kernel updates to reduce overhead
    )

    logging.info("Model architecture:\n%s", model)

    # Learning rate scheduler (cosine annealing)
    if params.lr_schedule == "cosine":
        if params.warmup > 0:
            lr_scale = lambda x: min((x + 1) / params.warmup,
                                     0.5 * (1 + np.cos(np.pi * x / params.num_iters)))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.num_iters)
    else:
        scheduler = None

    # Set up AMP scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    logging.info("Starting Training Loop...")

    # Log initial loss on train and validation to TensorBoard
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
        # Reset peak memory stats at the beginning of each epoch.
        torch.cuda.reset_peak_memory_stats(device)

        start_epoch = time.time()
        model.train()
        epoch_losses = []
        epoch_sample_count = 0  # Track number of samples processed this epoch

        for data in train_data_loader:
            iters += 1
            inp, tar = data
            inp = inp.to(device, non_blocking=True)
            tar = tar.to(device, non_blocking=True)
            epoch_sample_count += inp.shape[0]  # Count samples in current batch

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

        # Get the max memory allocated on the GPU during this epoch (in MB)
        max_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        logging.info("Epoch %d: Loss = %.6f, Time = %.2f sec, Throughput = %.2f samples/sec, Max GPU Mem = %.2f MB",
                     epoch+1, avg_loss, epoch_time, throughput, max_mem_mb)
        args.tboard_writer.add_scalar("Loss/train", avg_loss, iters)
        args.tboard_writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], iters)
        args.tboard_writer.add_scalar("Throughput/train", throughput, iters)
        args.tboard_writer.add_scalar("GPU_Memory/Max", max_mem_mb, iters)

        # Validation
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
    logging.info("Training completed in %.2f sec", total_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default="00", type=str, help="tag for current experiment")
    parser.add_argument("--yaml_config", default="./config/ViT.yaml", type=str, help="path to yaml config")
    parser.add_argument("--config", default="base", type=str, help="name of config in YAML")
    parser.add_argument("--local_batch_size", default=None, type=int, help="local batch size")
    parser.add_argument("--num_iters", default=None, type=int, help="number of iterations")

    args = parser.parse_args()
    run_num = args.run_num
    params = YParams(os.path.abspath(args.yaml_config), args.config)

    if args.num_iters:
        params.update({"num_iters": args.num_iters})
    if args.local_batch_size:
        params.local_batch_size = args.local_batch_size
        params.update({"global_batch_size": args.local_batch_size})
    else:
        params.local_batch_size = params.global_batch_size

    # Set up experiment directory and logging
    expDir = os.path.join(params.expdir, args.config, run_num)
    if not os.path.isdir(expDir):
        os.makedirs(expDir)
    logging_utils.log_to_file(log_filename=os.path.join(expDir, "out.log"))
    params.log()
    args.tboard_writer = SummaryWriter(log_dir=os.path.join(expDir, "logs/"))

    params.experiment_dir = os.path.abspath(expDir)
    train(params, args)
    logging.info("DONE")

