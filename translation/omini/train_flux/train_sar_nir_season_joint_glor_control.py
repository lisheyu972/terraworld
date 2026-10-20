"""Joint SAR/NIR/season training on a single FLUX backbone."""
import argparse
import copy
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import shutil
import time

import lightning as L
from lightning.pytorch.strategies import DDPStrategy
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from .joint_task_data import TASKS, JointTaskDataset, JointTaskBatchSampler
from .sar_to_optical_data import SARToOpticalDataset
from .nir_to_optical_data import NIRToOpticalDataset
from .ssl4eo_season_data import SSL4EOSeasonDataset
from .train_frame_interp import EXPANDED_LORA_TARGET_MODULES
from .trainer import OminiModel


class TaskConditionEncoders(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoders = nn.ModuleDict({t: copy.deepcopy(encoder) for t in TASKS})
        self.active_task = TASKS[0]

    def forward(self, image):
        return self.encoders[self.active_task](image)


def task_glor_state(state, task):
    """Export exactly the ordinary single-task GLoR state dictionary."""
    prefix = f"condition_encoder.encoders.{task}."
    result = {}
    for key, value in state.items():
        if key.startswith(prefix):
            result['condition_encoder.' + key[len(prefix):]] = value
        elif not key.startswith('condition_encoder.encoders.'):
            result[key] = value
    return result


class JointModel(OminiModel):
    def init_lora(self, lora_path, lora_config):
        # Identical module insertion order across ranks, irrespective of hash seed.
        self.adapter_set = tuple(sorted(self.adapter_set))
        return super().init_lora(lora_path, lora_config)

    def __init__(self, *, joint_config, **kwargs):
        super().__init__(**kwargs)
        self.joint_config = copy.deepcopy(joint_config)
        self.glor_control.condition_encoder = TaskConditionEncoders(
            self.glor_control.condition_encoder)
        self.active_task = TASKS[0]
        self.task_updates = {t: 0 for t in TASKS}
        self.task_loss = {t: None for t in TASKS}

    def training_step(self, batch, batch_idx):
        task_ids = set(batch['task_id'])
        if len(task_ids) != 1 or not task_ids <= set(TASKS):
            raise ValueError(f"Expected one supported task per batch, got {task_ids}")
        self.active_task = next(iter(task_ids))
        # Assign a NEW list: checkpoint recomputation retains its forward's list.
        self.adapter_names = ['text', 'main', f'condition_{self.active_task}']
        self.glor_control.condition_encoder.active_task = self.active_task
        loss = super().training_step(batch, batch_idx)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {self.active_task} loss")
        self.task_loss[self.active_task] = float(loss.detach())
        return loss

    def on_before_optimizer_step(self, optimizer):
        # Omini scales inactive adapters to zero rather than disabling them.
        # Clear their zero gradients to avoid Adam momentum/decay changing a
        # different task's adapter. One task occupies the entire accumulation.
        inactive = [f'.condition_{t}.' for t in TASKS if t != self.active_task]
        for name, param in self.transformer.named_parameters():
            if 'lora_' in name and any(token in name for token in inactive):
                param.grad = None

    def save_joint(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        super().save_lora(str(path))
        (path/'config.yaml').write_text(yaml.safe_dump(self.joint_config, sort_keys=False))
        state = self.glor_control.state_dict()
        for task in TASKS:
            dest = path/task
            dest.mkdir()
            for source, target in [('text','text'),('main','main'),(f'condition_{task}','condition')]:
                src, dst = path/f'{source}.safetensors', dest/f'{target}.safetensors'
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
            torch.save({'glor_control': task_glor_state(state, task)}, dest/'glor_control.pt')
            cfg = copy.deepcopy(self.joint_config)
            task_cfg = cfg.pop('tasks')[task]
            cfg['model']['adapter_names'] = ['text', 'main', 'condition']
            cfg['train']['dataset'] = task_cfg['dataset']
            cfg['train']['condition_type'] = task_cfg['condition_type']
            (dest/'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        (path/'complete.json').write_text(json.dumps({'step': int(self.global_step),
            'task_updates': self.task_updates, 'format': 'joint-and-single-task-inference-weights',
            'optimizer_state_saved': False}, indent=2))


class JointDataModule(L.LightningDataModule):
    def __init__(self, datasets, config):
        super().__init__()
        self.datasets, self.config = datasets, config

    def train_dataloader(self):
        cfg = self.config
        self.batch_sampler = JointTaskBatchSampler(
            {t: len(d) for t,d in self.datasets.items()}, cfg['batch_size'],
            cfg['accumulate_grad_batches'], cfg['updates_per_epoch'],
            rank=self.trainer.global_rank, world_size=self.trainer.world_size,
            seed=cfg['seed'])
        return DataLoader(JointTaskDataset(self.datasets), batch_sampler=self.batch_sampler,
            num_workers=cfg['dataloader_workers'], pin_memory=True,
            persistent_workers=cfg['dataloader_workers'] > 0)


class JointProgress(L.Callback):
    def __init__(self, config):
        self.config = config
        self.last_step = 0
        self.last_save = -1
        self.started = time.monotonic()

    def on_fit_start(self, trainer, model):
        run_name = datetime.now().strftime('%Y%m%d-%H%M%S') if trainer.is_global_zero else None
        run_name = trainer.strategy.broadcast(run_name)
        self.run_dir = Path(self.config['train']['save_path'])/run_name
        if trainer.is_global_zero:
            self.run_dir.mkdir(parents=True, exist_ok=False)
            (self.run_dir/'config.yaml').write_text(yaml.safe_dump(self.config, sort_keys=False))
            print(f'Joint training output: {self.run_dir}', flush=True)

    def on_train_epoch_start(self, trainer, model):
        trainer.datamodule.batch_sampler.set_epoch(trainer.current_epoch)

    def save(self, trainer, model):
        # All ranks wait so checkpoint I/O cannot race the next collective.
        trainer.strategy.barrier()
        if trainer.is_global_zero and self.last_save != trainer.global_step:
            path = self.run_dir/'ckpt'/str(trainer.global_step)
            print(f'Saving joint and task exports: {path}', flush=True)
            model.save_joint(path)
        self.last_save = trainer.global_step
        trainer.strategy.barrier()

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        step = trainer.global_step
        if step == self.last_step:
            return
        self.last_step = step
        model.task_updates[model.active_task] += 1
        cfg = self.config['train']
        if trainer.is_global_zero and (step == 1 or step % cfg['print_every_n_steps'] == 0):
            losses = ', '.join(f'{t}={v:.5f}' for t,v in model.task_loss.items() if v is not None)
            print(f'[joint] step={step}/{cfg["max_steps"]} task={model.active_task} '
                  f'losses({losses}) updates={model.task_updates} '
                  f'elapsed={(time.monotonic()-self.started)/60:.1f}min', flush=True)
        if step % cfg['save_interval'] == 0:
            self.save(trainer, model)

    def on_train_end(self, trainer, model):
        if trainer.global_step and self.last_save != trainer.global_step:
            self.save(trainer, model)


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    cfg = config['train']
    for env, key in [('BATCH_SIZE','batch_size'), ('GRAD_ACCUM_STEPS','accumulate_grad_batches'),
        ('MAX_STEPS','max_steps'), ('SAVE_EVERY_N_STEPS','save_interval'),
        ('PRINT_EVERY_N_STEPS','print_every_n_steps'), ('NUM_WORKERS','dataloader_workers'),
        ('SEED','seed'), ('UPDATES_PER_EPOCH','updates_per_epoch')]:
        if env in os.environ:
            cfg[key] = int(os.environ[env])
    for key in ['batch_size','accumulate_grad_batches','max_steps','save_interval',
                'print_every_n_steps','updates_per_epoch']:
        if cfg[key] < 1:
            raise ValueError(f'{key} must be positive')
    if cfg['dataloader_workers'] < 0:
        raise ValueError('NUM_WORKERS must be nonnegative')
    if cfg['updates_per_epoch'] % 3:
        raise ValueError('UPDATES_PER_EPOCH must be divisible by three for equal task sampling')
    cfg['lora_config']['target_modules'] = EXPANDED_LORA_TARGET_MODULES
    for env,key in [('FLUX_PATH','flux_path'),('PRETRAINED_TRANSFORMER','transformer_path')]:
        if env in os.environ:
            config[key] = os.environ[env]
    if 'SAVE_PATH' in os.environ:
        cfg['save_path'] = os.environ['SAVE_PATH']
    for task in TASKS:
        if f'{task.upper()}_DATA_ROOT' in os.environ:
            config['tasks'][task]['dataset']['root'] = os.environ[f'{task.upper()}_DATA_ROOT']
    if 'SEASON_INDEX_PATH' in os.environ:
        config['tasks']['season']['dataset']['index_path'] = os.environ['SEASON_INDEX_PATH']
    if os.environ.get('OMINI_LORA_PATH') or cfg.get('lora_path'):
        raise ValueError('This entry point initializes fresh joint adapters; unset OMINI_LORA_PATH')
    expected = ['text','main',*[f'condition_{t}' for t in TASKS]]
    if config['model']['adapter_names'] != expected:
        raise ValueError(f'model.adapter_names must be {expected}')
    glor = config['glor_control']
    if not glor['enable'] or glor['task'] != 'super_resolution' or glor['disable_omini_condition']:
        raise ValueError('Joint training requires single-image GLoR and enabled Omini conditions')
    if config['tasks']['season']['dataset'].get('drop_text_prob', 0) != 0:
        raise ValueError('Season target is specified in text; drop_text_prob must be zero')
    return config


def check_paths(config):
    flux = Path(config['flux_path'])
    transformer = Path(config['transformer_path'])
    required = [flux/'model_index.json', transformer/'config.json']
    for component in ['vae','text_encoder','text_encoder_2','tokenizer','tokenizer_2','scheduler']:
        required.append(flux/component)
    index = transformer/'diffusion_pytorch_model.safetensors.index.json'
    if index.exists():
        required.extend(transformer/f for f in set(json.loads(index.read_text())['weight_map'].values()))
    else:
        required.append(transformer/'diffusion_pytorch_model.safetensors')
    required.extend(Path(config['tasks'][t]['dataset']['root']) for t in TASKS)
    required.append(Path(config['tasks']['season']['dataset']['index_path']))
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError('Missing model/data paths:\n'+'\n'.join(missing))


def make_datasets(config):
    constructors = {'sar': SARToOpticalDataset, 'nir': NIRToOpticalDataset,
                    'season': SSL4EOSeasonDataset}
    return {t: constructors[t](**config['tasks'][t]['dataset']) for t in TASKS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='train/config/sar_nir_season_joint_glor_control.yaml')
    parser.add_argument('--dry-run', action='store_true', help='Check all paths, datasets and one batch per task without loading FLUX')
    args = parser.parse_args()
    config = load_config(args.config)
    check_paths(config)
    cfg = config['train']
    L.seed_everything(cfg['seed'], workers=True)
    datasets = make_datasets(config)
    world = int(os.environ.get('WORLD_SIZE', os.environ.get('JOINT_WORLD_SIZE', '1')))
    print(f'Joint tasks={TASKS}; balanced 1:1:1; transformer={config["transformer_path"]}', flush=True)
    print(f'Batch/GPU={cfg["batch_size"]}; accumulation={cfg["accumulate_grad_batches"]}; '
          f'GPUs={world}; effective batch={cfg["batch_size"]*cfg["accumulate_grad_batches"]*world}; '
          f'total updates={cfg["max_steps"]} (~{cfg["max_steps"]/3:g}/task)', flush=True)
    if args.dry_run:
        from torch.utils.data import default_collate
        sampler = JointTaskBatchSampler({t:len(d) for t,d in datasets.items()}, cfg['batch_size'],
            1, 3, world_size=world, seed=cfg['seed'])
        combined = JointTaskDataset(datasets)
        for indices in sampler:
            batch = default_collate([combined[key] for key in indices])
            print(f'CHECK {batch["task_id"][0]}: samples={len(datasets[batch["task_id"][0]])} '
                  f'target={tuple(batch["image"].shape)} condition={tuple(batch["condition_0"].shape)}', flush=True)
        print('DRY RUN PASSED. No model loaded and no training started.', flush=True)
        return
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    model = JointModel(joint_config=config, flux_pipe_id=config['flux_path'],
        transformer_path=config['transformer_path'], lora_config=cfg['lora_config'],
        device='cuda', dtype=getattr(torch, config['dtype']), optimizer_config=cfg['optimizer'],
        model_config=config['model'], adapter_names=config['model']['adapter_names'],
        gradient_checkpointing=cfg['gradient_checkpointing'],
        glor_control_config=config['glor_control'], glor_task='super_resolution')
    model.training_config = cfg
    trainer = L.Trainer(accelerator='gpu', devices=world,
        precision='bf16-mixed' if config['dtype'] == 'bfloat16' else '32-true',
        strategy=DDPStrategy(find_unused_parameters=True, timeout=timedelta(hours=6)) if world>1 else 'auto',
        use_distributed_sampler=False, accumulate_grad_batches=cfg['accumulate_grad_batches'],
        max_steps=cfg['max_steps'], max_epochs=-1, gradient_clip_val=cfg['gradient_clip_val'],
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, callbacks=[JointProgress(config)])
    trainer.fit(model, datamodule=JointDataModule(datasets, cfg))


if __name__ == '__main__':
    main()
