import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """把一个epoch的数据完整过一遍。start_step为断点续训的起始步(正常训练为0)"""
    start_time = time.time()
    last_step = start_step  # 记录本epoch实际跑到的步号，供循环结束后判断有无残留梯度
    # ==================== 主循环：每个step处理一个batch ====================
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # ---- ① 准备：数据搬上GPU + 按余弦公式取当前学习率 ----
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)  # pad位置已被标成-100，算loss时自动忽略
        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)  # 用全局进度查余弦曲线，先大后小
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ---- ② 前向：半精度模式下算预测和损失 ----
        with autocast_ctx:
            res = model(input_ids, labels=labels)  # 传入labels后模型内部错位对比并算好交叉熵loss
            loss = res.loss + res.aux_loss  # 总损失 = 语言任务损失 + MoE负载均衡惩罚(非MoE时恒为0)
            loss = loss / args.accumulation_steps  # 除以累积份数：8份小梯度相加恰好等价于一个大batch

        scaler.scale(loss).backward()  # 放大loss防半精度下溢；反向传播只累加梯度，不改参数

        # ---- ③ 参数更新：攒够accumulation_steps份梯度才真正执行一次(唯一改变模型的地方) ----
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)  # 先除掉放大倍数，还原真实梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度过大则等比截断，防脏数据一步改坏模型
            scaler.step(optimizer)  # 梯度无inf/nan才执行 参数 -= lr * 梯度
            scaler.update()  # 根据刚才是否溢出，自适应调整放大系数
            optimizer.zero_grad(set_to_none=True)  # 清掉已消费的梯度，开始攒下一批

        # ---- ④ 定期报数：盯loss是否在降、lr曲线、预计还要多久 ----
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps  # .item()取出数值；乘回去还原真实loss(前向时被除了)
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss  # 从总损失里拆出纯语言任务部分
            current_lr = optimizer.param_groups[-1]['lr']  # 直接读optimizer当前生效值，保证日志不说谎
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60  # 平均每步耗时 × 剩余步数，换算成分钟
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # ---- ⑤ 定期存档：仅主进程执行(多卡时N个进程同写一个文件会损坏) ----
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()  # 切评估模式(关dropout)，存完再切回
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'  # 配置编进文件名，不同规模互不覆盖
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model  # 剥掉DDP外壳，否则键名带module.前缀
            raw_model = getattr(raw_model, '_orig_mod', raw_model)  # 再剥torch.compile外壳(没编译过则原样返回)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)  # 存档一：纯权重fp16+CPU张量，供推理/发布(开源发的就是这种)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')  # 存档二：resume包(optimizer状态+epoch+step)，供断点续训
            model.train()
            del state_dict

        del input_ids, labels, res, loss  # 立刻释放本batch显存，避免下一批加载时新旧两批同时占显存

    # ==================== 收尾：epoch结束时手里还有没凑满一份的零头梯度 → 补更新一次 ====================
    # 条件一：确实跑过至少一步；条件二：最后一步不是累积周期整数倍(是的话上面已正常更新过)
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()