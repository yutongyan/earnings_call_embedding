# Using CUHK SCRP for LLM Fine-Tuning

A practical guide to running Llama 3 SFT experiments on the SCRP High Performance Computing Cluster at CUHK Economics.

## Cluster Overview

SCRP uses Slurm for job scheduling. GPU nodes relevant to this project:

| Partition | Node | GPUs | RAM | Access |
|-----------|------|------|-----|--------|
| `a100` | scrp-node-13 | 8x A100 80GB | 2 TB | Faculty + RPg students |
| `a100` | scrp-node-21 | 8x A100 80GB | 2 TB | Faculty + RPg students |
| `a100` | scrp-node-24 | 8x A100 80GB | 2.3 TB | Faculty + RPg students |
| `gpu` | scrp-node-8,9 | 4x RTX 3090 24GB | — | All users |
| `gpu` | scrp-node-22 | 4x A100 + 1x L40S + 2x RTX 3090 | — | All users |
| `jrc` | scrp-node-11 | 2x A100 80GB | 747 GB | JRC members only |

Some nodes also have **NVIDIA H100 (Hopper) GPUs** — labeled as `a100` on the cluster for accounting. Use the `hopper` shortcut or `--constraint=hopper` to specifically request H100s over A100s. H100s have higher memory bandwidth and FP8 support, making them faster for LLM training.

## Quick Start

### Check your resource limits

```bash
qos
```

This shows your QoS tier:

| QoS | CPUs | GPUs | Max Duration |
|-----|------|------|-------------|
| c4g1 | 4 | 1 | 1 day |
| c16g1 | 16 | 1 | 5 days |
| c32g4 | 128 | 4 | 5 days |
| c32g8 | 128 | 8 | 5 days |
| c16-long | 16 | 1 | 30 days |

Default job duration on `a100` partition is **6 hours**. Always specify `-t` explicitly.

### Check node status

```bash
scrp-info
```

## Running GPU Jobs

### Using SCRP shortcuts (simplest)

```bash
# Request 1 Hopper GPU (preferred — auto-allocates 8 CPUs + 160GB RAM)
hopper python my_script.py

# Request 1 A100 GPU
a100 python my_script.py

# Request 1 RTX 3090
rtx3090 python my_script.py
```

### Using `compute` command (more control)

```bash
# 1x Hopper GPU with 16 CPUs for 24 hours
compute --gpus=a100 --constraint=hopper -c 16 --mem=160G -t 24:00:00 python train.py

# 2x Hopper GPUs for multi-GPU training
compute --gpus=a100:2 --constraint=hopper -c 32 --mem=320G -t 48:00:00 python train.py

# Interactive shell on Hopper node
compute --gpus=a100 --constraint=hopper -c 16 --mem=160G -t 6:00:00 bash

# Fallback to A100 if Hopper is busy
compute --gpus=a100 -c 16 --mem=160G -t 24:00:00 python train.py
```

Note: Hopper GPUs are labeled as `a100` on the cluster for accounting purposes. Use `--constraint=hopper` to specifically request Hopper over A100.

### Using `srun` directly (maximum flexibility)

```bash
# Interactive Hopper session for 24 hours
srun --pty -p a100 --gpus=1 --constraint=hopper -c 16 --mem=160G -t 24:00:00 bash

# 2x Hopper for 70B model training, 5 days
srun --pty -p a100 --gpus=2 --constraint=hopper -c 32 --mem=320G -t 5-0 bash

# Long-running job (30 days, faculty only)
srun --pty -p a100 --gpus=1 --constraint=hopper -c 16 --mem=160G -t 30-0 -q c16-long bash
```

## Batch Jobs (Recommended for Training)

Training runs should be submitted as batch jobs so they don't die if your SSH connection drops.

### Llama 3.1 8B LoRA Training Script

Create `train_8b.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=llama3-8b-sft
#SBATCH --partition=a100
#SBATCH --gpus=1
#SBATCH --constraint=hopper
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_7b_%j.out
#SBATCH --error=logs/train_7b_%j.err

# Load environment
source ~/miniconda3/bin/activate
conda activate llm

# Run training
python src/train.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --data_path data/train.json \
    --output_dir checkpoints/llama3-8b-sft \
    --lora_r 32 \
    --lora_alpha 64 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --learning_rate 2e-4 \
    --max_seq_length 4096 \
    --bf16
```

Submit:

```bash
mkdir -p logs
sbatch train_7b.sh
```

### Llama 3.1 70B QLoRA Training Script

Create `train_70b.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=llama2-70b-qlora
#SBATCH --partition=a100
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/train_70b_%j.out
#SBATCH --error=logs/train_70b_%j.err

source ~/miniconda3/bin/activate
conda activate llm

python src/train.py \
    --model_name meta-llama/Llama-3.1-70B-Instruct \
    --data_path data/train.json \
    --output_dir checkpoints/llama2-70b-qlora \
    --use_4bit \
    --lora_r 64 \
    --lora_alpha 128 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --max_seq_length 4096 \
    --bf16
```

### Multi-GPU with DeepSpeed

For 70B training across 4 GPUs:

```bash
#!/bin/bash
#SBATCH --job-name=llama2-70b-ds
#SBATCH --partition=a100
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/train_70b_ds_%j.out

source ~/miniconda3/bin/activate
conda activate llm

torchrun --nproc_per_node=4 src/train.py \
    --model_name meta-llama/Llama-3.1-70B-Instruct \
    --deepspeed configs/ds_config_zero3.json \
    --data_path data/train.json \
    --output_dir checkpoints/llama2-70b-ds \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --bf16
```

## Monitoring Jobs

```bash
# Your running jobs
scrp-queue

# All jobs on A100 partition
squeue -p a100

# Detailed job info
scontrol show job <job_id>

# Cancel a job
scancel <job_id>

# Watch GPU utilization on your node (from within the job)
watch -n 5 nvidia-smi
```

## Resource Limits and Tips

### A100 partition constraints
- **2 GPUs per user** (check with `qos`)
- **32 CPUs per job**
- **512 GB RAM per job**
- Default duration: **6 hours** — always set `-t`

### Storage

| Location | Use | Quota (RPg) |
|----------|-----|-------------|
| `~/` | Code, configs | 50 GB |
| `~/large-data` (-> /data/users/) | Datasets, models | 500 GB (BeeGFS distributed) |
| `~/archive` | Backups | 1 TB |
| Distributed storage | Large files | ~2 TB |

Check usage: `scrp-quota`

### WRDS Access

WRDS database queries work via PostgreSQL with credentials in `~/.pgpass`:

```bash
# Copy pgpass from your local machine
scp ~/.pgpass scrp-wenzhuoyue:~/.pgpass
chmod 600 ~/.pgpass

# Test
python3 -c "import wrds; db = wrds.Connection(wrds_username='yutongyancuhk'); print('OK'); db.close()"
```

No Duo push needed for database queries once `.pgpass` is configured.

### Environment setup (first time)

```bash
# Install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create LLM environment
conda create -n llm python=3.10
conda activate llm

# Install PyTorch + training stack
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft trl datasets accelerate bitsandbytes
pip install deepspeed  # for multi-GPU

# For data processing
pip install wrds scikit-learn
```

### Data transfer

```bash
# Upload data via parallel scp (split + reassemble for speed)
split -n 4 -d data.tar.gz data_part_
for i in 00 01 02 03; do scp data_part_$i scrp-wenzhuoyue:~/large/ & done
wait
ssh scrp-wenzhuoyue "cat ~/large/data_part_* > ~/large/data.tar.gz && tar xzf ~/large/data.tar.gz"

# Download from Dropbox directly on SCRP
ssh scrp-wenzhuoyue "wget -O ~/large/file.zip 'DROPBOX_URL&dl=1'"
```

## Recommended Workflow

1. **Develop locally** — write and debug code on your laptop/workstation
2. **Push to GitHub** — `git push origin main`
3. **Pull on SCRP** — `ssh scrp-wenzhuoyue && cd ~/large/project && git pull`
4. **Submit batch job** — `sbatch run_job.sh`
5. **Monitor** — `scrp-queue` and check `logs/`
6. **Pull results** — download outputs

### CPU-only jobs (data processing, regression)

```bash
#!/bin/bash
#SBATCH --job-name=process
#SBATCH --partition=scrp
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

source ~/miniconda3/bin/activate llm
python3 -u src/process_data.py
```

## Contact

SCRP High Performance Computing Cluster
Department of Economics, The Chinese University of Hong Kong
