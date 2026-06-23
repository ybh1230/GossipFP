import argparse
import copy
import logging
import os
import os.path as osp
import pprint
import random
import time
from datetime import datetime
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from tensorboardX import SummaryWriter

from dataset.transform import generate_unsup_data
from dataset.builder import get_loader

from model.model.model_helper import ModelBuilder
from model.gossip import GossipPrototypeBank
from utils.dist_helper import setup_distributed
from utils.loss_helper import (
    get_criterion,
    compute_pseudo_label_loss,
)
from utils.lr_helper import get_optimizer, get_scheduler
from utils.utils import (
    AverageMeter,
    init_log,
    intersectionAndUnion,
    load_state,
    set_random_seed,
)

from torch.cuda.amp import GradScaler, autocast


parser = argparse.ArgumentParser(description="Semi-Supervised Semantic Segmentation")
parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--port", default=None, type=int)


def to_status(m, status):
    if hasattr(m, 'batch_type'):
        m.batch_type = status

to_clean_status = partial(to_status, status='clean')
to_adv_status = partial(to_status, status='adv')
to_mix_status = partial(to_status, status='mix')
to_warm_status = partial(to_status, status='warm_up')


def main():
    global args, cfg
    args = parser.parse_args()
    seed = args.seed
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    scaler = GradScaler()

    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    cfg["exp_path"] = os.path.dirname(args.config)
    cfg["save_path"] = os.path.join(cfg["exp_path"], cfg["saver"]["snapshot_dir"])

    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True

    rank, word_size = setup_distributed(port=args.port)

    if rank == 0:
        logger.info("{}".format(pprint.pformat(cfg)))
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        tb_logger = SummaryWriter(
            osp.join(cfg["exp_path"], "log/events_seg/" + current_time)
        )
    else:
        tb_logger = None

    if args.seed is not None:
        print("set random seed to", args.seed)
        set_random_seed(args.seed)

    if not osp.exists(cfg["saver"]["snapshot_dir"]) and rank == 0:
        os.makedirs(cfg["saver"]["snapshot_dir"])

    # Create network
    num_classes = cfg["net"].get("num_classes", cfg["dataset"].get("num_cls"))
    model = ModelBuilder(num_classes)
    modules_back = [model.encoder]
    modules_head = [model.decoder]

    # Teacher model
    model_teacher = ModelBuilder(num_classes)

    model.cuda()
    model_teacher.cuda()

    sup_loss_fn = get_criterion(cfg)

    gossip_cfg = cfg.get("gossip", {})
    gossip_bank = GossipPrototypeBank(
        num_classes=num_classes,
        feat_dim=cfg["net"].get("fts_dim", 256),
        momentum=gossip_cfg.get("momentum", 0.99),
        confusion_momentum=gossip_cfg.get("confusion_momentum", 0.95),
        topk=gossip_cfg.get("topk", 3),
        temperature=gossip_cfg.get("temperature", 0.2),
        confusion_weight=gossip_cfg.get("confusion_weight", 0.5),
        min_pixels=gossip_cfg.get("min_pixels", 16),
        margin=gossip_cfg.get("margin", 0.2),
        ignore_label=cfg.get("ignore_label", 255),
    ).cuda()

    train_loader_sup, train_loader_unsup, val_loader = get_loader(cfg, seed=seed)

    # Optimizer and lr decay scheduler
    cfg_trainer = cfg["trainer"]
    cfg_optim = cfg_trainer["optimizer"]
    times = 10 if "pascal" in cfg["dataset"]["type"] else 1

    params_list = []
    for module in modules_back:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"])
        )
    for module in modules_head:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"] * times)
        )

    optimizer = get_optimizer(params_list, cfg_optim)

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    model_teacher = torch.nn.parallel.DistributedDataParallel(
        model_teacher,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
    
    for p in model_teacher.parameters():
        p.requires_grad = False

    best_prec = 0
    last_epoch = 0 #############################

    # auto_resume > pretrain
    if cfg["saver"].get("auto_resume", False):
        lastest_model = os.path.join(cfg["save_path"], "ckpt.pth")
        if not os.path.exists(lastest_model):
            "No checkpoint found in '{}'".format(lastest_model)
        else:
            print(f"Resume model from: '{lastest_model}'")
            best_prec, last_epoch = load_state(
                lastest_model, model, optimizer=optimizer, key="model_state"
            )
            checkpoint = torch.load(lastest_model, map_location="cpu")
            if "gossip_state" in checkpoint:
                gossip_bank.load_state_dict(checkpoint["gossip_state"], strict=False)

    elif cfg["saver"].get("pretrain", False):
        load_state(cfg["saver"]["pretrain"], model, key="model_state")

    optimizer_start = get_optimizer(params_list, cfg_optim)
    lr_scheduler = get_scheduler(
        cfg_trainer, len(train_loader_sup), optimizer_start, start_epoch=last_epoch
    )

    # Start to train model
    for epoch in range(last_epoch, cfg_trainer["epochs"]):
        # Training
        train(
            model,
            model_teacher,
            gossip_bank,
            optimizer,
            lr_scheduler,
            sup_loss_fn,
            train_loader_sup,
            train_loader_unsup,
            epoch,
            tb_logger,
            logger,
            scaler,
        )

        # Validation
        if cfg_trainer["eval_on"] and (epoch+1) % 4 == 0:
            if rank == 0:
                logger.info("start evaluation")


            prec_s = validate(model, val_loader, epoch, logger)

            if  epoch >= cfg["trainer"].get("gossip_start_epoch", 1):
                prec_t = validate(model_teacher, val_loader, epoch, logger)
                prec = max(prec_s, prec_t)
            else:
                prec = prec_s

            if rank == 0:
                state = {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "teacher_state": model_teacher.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_miou": best_prec,
                    "gossip_state": gossip_bank.state_dict(),
                }
                if prec > best_prec:
                    best_prec = prec
                    torch.save(
                        state, osp.join(cfg["saver"]["snapshot_dir"], "ckpt_best.pth")
                    )

                torch.save(state, osp.join(cfg["saver"]["snapshot_dir"], "ckpt.pth"))

                logger.info(
                    "\033[31m * Currently, the best val result is: {:.2f}\033[0m".format(
                        best_prec * 100
                    )
                )
                
                tb_logger.add_scalar("mIoU val", prec_s, epoch)

                if epoch >= cfg["trainer"].get("gossip_start_epoch", 1):
                    tb_logger.add_scalar("mIoU val teacher", prec_t, epoch)
                    

def train(
    model,
    model_teacher,
    gossip_bank,
    optimizer,
    lr_scheduler,
    sup_loss_fn,
    loader_l,
    loader_u,
    epoch,
    tb_logger,
    logger,
    scaler,
):
    ema_decay_origin = cfg["net"]["ema_decay"]
    gossip_start = cfg["trainer"].get("gossip_start_epoch", 1)
    gossip_weight = cfg.get("gossip", {}).get("loss_weight", 0.05)
    model.train()

    if epoch >= gossip_start + 1:
        model.apply(to_mix_status)
    else:
        model.apply(to_warm_status)

    model_teacher.apply(to_clean_status)

    loader_l.sampler.set_epoch(epoch)
    loader_u.sampler.set_epoch(epoch)
    loader_l_iter = iter(loader_l)
    loader_u_iter = iter(loader_u)
    assert len(loader_l) == len(
        loader_u
    ), f"labeled data {len(loader_l)} unlabeled data {len(loader_u)}, imbalance!"

    rank, world_size = dist.get_rank(), dist.get_world_size()

    sup_losses = AverageMeter(10)
    uns_losses = AverageMeter(10)
    gossip_losses = AverageMeter(10)
    active_class_meter = AverageMeter(10)
    seen_class_meter = AverageMeter(10)
    data_times = AverageMeter(10)
    batch_times = AverageMeter(10)
    learning_rates = AverageMeter(10)

    batch_end = time.time()
    for step in range(len(loader_l)):
        batch_start = time.time()
        data_times.update(batch_start - batch_end)

        i_iter = epoch * len(loader_l) + step
        lr = lr_scheduler.get_lr()
        learning_rates.update(lr[0])
        lr_scheduler.step()

        image_l, label_l = loader_l_iter.next()
        batch_size, h, w = label_l.size()
        image_l, label_l = image_l.cuda(), label_l.cuda()

        image_u, image_u_s, ignore_mask = loader_u_iter.next()
        image_u = image_u.cuda()
        image_u_s = image_u_s.cuda()
        ignore_mask = ignore_mask.cuda()
        
        dist.barrier()

        num_labeled = len(image_l)
        if epoch < cfg["trainer"]["sup_only_epoch"]:
            with autocast():
                # forward
                outs = model(image_l)
                pred = outs['pred']
                pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=True)
                
                sup_loss = sup_loss_fn(pred, label_l)
                unsup_loss = 0 * pred[num_labeled:].sum()

                model_teacher.train()
                _ = model_teacher(image_l)

            loss = sup_loss + unsup_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            gossip_loss = pred.sum() * 0.0
            gossip_stats = {
                "active_classes": 0,
                "seen_classes": int(gossip_bank.seen.sum().item()),
            }

        else:
            if epoch == cfg["trainer"].get("sup_only_epoch", 1):
            # copy student parameters to teacher
                with torch.no_grad():
                    for t_params, s_params in zip(
                        model_teacher.parameters(), model.parameters()
                    ):
                        t_params.data = s_params.data
            
            image_bi = torch.cat((image_l, image_u))
            
            # teacher model predicts on all data
            model_teacher.train()
            with torch.no_grad():

                outs_t = model_teacher(image_bi)
                pred_bi_t, fts_bi_t = outs_t['pred'].detach(), outs_t['fts']
                
                # Get predictions of original unlabeled data
                pred_u_t = pred_bi_t[num_labeled:]
                
                pred_u_t_large = F.interpolate(pred_u_t, (h, w), mode="bilinear", align_corners=True)
                prob_u_t_large = F.softmax(pred_u_t_large, dim=1)
                logits_u_t, label_u_t = torch.max(prob_u_t_large, dim=1)

                # label threshold
                thresh = cfg['trainer']['thresh']
                thresh_mask = logits_u_t.le(thresh).bool() * (ignore_mask != 255).bool()
                label_u_t[thresh_mask] = 255
                label_u_t[ignore_mask == 255] = 255

                label_bi_t = torch.cat((label_l.clone(), label_u_t.clone()), dim=0)
                gossip_stats = gossip_bank.update(fts_bi_t, label_bi_t, logits=pred_bi_t)

                if random.uniform(0,1) < 0.5:
                    image_u_aug, label_u_aug, ignore_mask_aug = generate_unsup_data(
                        image_u_s,
                        label_u_t.clone(),
                        ignore_mask.clone(),
                        mode='cutmix',
                    )
                else:
                    image_u_aug, label_u_aug, ignore_mask_aug = image_u_s, label_u_t.clone(), ignore_mask.clone()


            with torch.no_grad():
                label_u_aug[ignore_mask_aug==255] = 255

            image_tri = torch.cat((image_l, image_u_aug), dim=0)

            model.train()
            with autocast():
                if epoch >= gossip_start + 1:
                    model.apply(to_mix_status)
                    out_tri_s = model(image_tri, label_u_aug.clone(), gossip_bank, cfg, eps=cfg['adv']['eps'], adv=True)
                    pred_tri_s = out_tri_s['pred']
                    
                    pred_tri_s_large = F.interpolate(pred_tri_s, size=label_l.shape[1:], mode="bilinear", align_corners=True)

                    pred_l_s_large, pred_u_aug_s_large, pred_u_aug_s_large_pt = pred_tri_s_large.chunk(3)

                    sup_loss = sup_loss_fn(pred_l_s_large, label_l.clone())

                    unsup_loss = compute_pseudo_label_loss(
                                pred_u_aug_s_large,
                                label_u_aug.clone(),
                                ignore_mask_aug.clone())
                    
                    unsup_loss_pt = compute_pseudo_label_loss(
                                pred_u_aug_s_large_pt,
                                label_u_aug.clone(),
                                ignore_mask_aug.clone())

                    label_tri = torch.cat((label_l.clone(), label_u_aug.clone()), dim=0)
                    gossip_loss = gossip_bank.separation_loss(out_tri_s["fts_train"], label_tri)

                    loss = sup_loss + 0.5*unsup_loss + 0.5*unsup_loss_pt + gossip_weight*gossip_loss

                else:
                    model.apply(to_warm_status)
                    outs_bi_s = model(image_tri)
                    pred_bi_s = outs_bi_s['pred']
                   
                    pred_bi_s_large = F.interpolate(pred_bi_s, size=label_l.shape[1:], mode="bilinear", align_corners=True)

                    pred_l_s_large = pred_bi_s_large[:num_labeled]
                    pred_u_aug_s_large_pt = pred_bi_s_large[num_labeled:]
                    sup_loss = sup_loss_fn(pred_l_s_large, label_l.clone())

                    unsup_loss = compute_pseudo_label_loss(
                                pred_u_aug_s_large_pt,
                                label_u_aug.clone(),
                                ignore_mask_aug.clone())

                    gossip_loss = sup_loss * 0.0
                    loss = sup_loss + unsup_loss
        

            dist.barrier()
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
             # update teacher model with EMA
            with torch.no_grad():
                ##########################
                ema_decay = min(
                    1
                    - 1
                    / (
                        i_iter
                        - len(loader_l) * cfg["trainer"].get("sup_only_epoch", 1)
                        + 1
                    ),
                    ema_decay_origin,
                )
                #############################
                for t_params, s_params in zip(
                    model_teacher.parameters(), model.parameters()
                ):
                    t_params.data.copy_(
                        ema_decay * t_params.data + (1 - ema_decay) * s_params.data
                    )


        # gather all loss from different gpus
        reduced_sup_loss = sup_loss.clone().detach()
        dist.all_reduce(reduced_sup_loss)
        reduced_sup_loss /= world_size
        sup_losses.update(reduced_sup_loss.item())

        reduced_uns_loss = unsup_loss.clone().detach()
        dist.all_reduce(reduced_uns_loss)
        reduced_uns_loss /= world_size
        uns_losses.update(reduced_uns_loss.item())

        reduced_gossip_loss = gossip_loss.clone().detach()
        dist.all_reduce(reduced_gossip_loss)
        reduced_gossip_loss /= world_size
        gossip_losses.update(reduced_gossip_loss.item())

        active_class_meter.update(gossip_stats["active_classes"])
        seen_class_meter.update(gossip_stats["seen_classes"])

        batch_end = time.time()
        batch_times.update(batch_end - batch_start)

        if i_iter % 50 == 0 and rank == 0:
            logger.info(
                "[{}] "
                "Iter [{}/{}]\t"
                "Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                "Sup {sup_loss.val:.3f} ({sup_loss.avg:.3f})\t"
                "Uns {uns_loss.val:.3f} ({uns_loss.avg:.3f})\t"
                "Gossip {gossip_loss.val:.3f} ({gossip_loss.avg:.3f})\t"
                "SeenCls {seen_cls.val:.0f}\t".format(
                    cfg["dataset"]["n_sup"],
                    i_iter,
                    cfg["trainer"]["epochs"] * len(loader_l),
                    batch_time=batch_times,
                    sup_loss=sup_losses,
                    uns_loss=uns_losses,
                    gossip_loss=gossip_losses,
                    seen_cls=seen_class_meter,
                )
            )
            tb_logger.add_scalar("lr", learning_rates.val, i_iter)
            tb_logger.add_scalar("Sup Loss", sup_losses.val, i_iter)
            tb_logger.add_scalar("Uns Loss", uns_losses.val, i_iter)
            tb_logger.add_scalar("Gossip Loss", gossip_losses.val, i_iter)
            tb_logger.add_scalar("Gossip Active Classes", active_class_meter.val, i_iter)
            tb_logger.add_scalar("Gossip Seen Classes", seen_class_meter.val, i_iter)



def validate(
    model,
    data_loader,
    epoch,
    logger,
):
    model.eval()
    model.apply(to_clean_status)
    data_loader.sampler.set_epoch(epoch)

    num_classes, ignore_label = (
        cfg["net"]["num_classes"],
        cfg["dataset"]["ignore_label"],
    )
    rank, world_size = dist.get_rank(), dist.get_world_size()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    for step, batch in enumerate(data_loader):
        images, labels = batch
        images = images.cuda()
        labels = labels.long().cuda()

        with torch.no_grad():
            output = model(images)['pred']

        # get the output produced by model_teacher
        output = F.interpolate(
            output, labels.shape[1:], mode="bilinear", align_corners=True
        )
        output = output.data.max(1)[1].cpu().numpy()
        target_origin = labels.cpu().numpy()

        # start to calculate miou
        intersection, union, target = intersectionAndUnion(
            output, target_origin, num_classes, ignore_label
        )

        # gather all validation information
        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()
        reduced_target = torch.from_numpy(target).cuda()

        dist.all_reduce(reduced_intersection)
        dist.all_reduce(reduced_union)
        dist.all_reduce(reduced_target)

        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)

    if rank == 0:
        for i, iou in enumerate(iou_class):
            logger.info(" * class [{}] IoU {:.2f}".format(i, iou * 100))
        logger.info(" * epoch {} mIoU {:.2f}".format(epoch, mIoU * 100))

    return mIoU


if __name__ == "__main__":
    main()
