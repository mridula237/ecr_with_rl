import os
from pyexpat import model
import torch
from tqdm import tqdm
from .lib import emotion_one_hot_encode, emotion_label_to_id
from torch.optim import AdamW
from transformers import get_scheduler
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_

# === NEW: for DDP-safe logging and synchronization ===
import torch.distributed as dist

def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
# ======================================================


# LABEL_MAP = {"neutral": 0, "joy": 1, "sadness": 2, "anger": 3, "surprise": 4, "fear": 5, "disgust": 6} # this is inccorrect indexing
LABEL_MAP = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}
CLASS_WEIGHTS = {"neutral": 4.0, "surprise": 15.0, "fear": 15.0, "sadness": 3.0, "joy": 1.0, "disgust": 6.0, "anger": 3.0}

ID_CLASS_WEIGHTS = {LABEL_MAP[k]: v for k, v in CLASS_WEIGHTS.items()}


def evaluate_model(model, 
                   processor,
                   test_loader, 
                   epoch, 
                   include_audio=False,  
                   device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Testing Epoch {epoch + 1}"):
            inputs, labels, audio_embed = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if include_audio:
                inputs["audio_inputs"] = audio_embed.to(device) if audio_embed is not None else None        
            
            outputs = model(**inputs)
            logits = torch.softmax(outputs["logits"], dim=1)
            preds = torch.argmax(logits, dim=1)
            targets = emotion_label_to_id(labels, label_map=LABEL_MAP)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets)
        
        sample_weights = [ID_CLASS_WEIGHTS[int(t)] for t in all_targets]
        
        accuracy = accuracy_score(all_targets, all_preds)
        weighted_accuracy = accuracy_score(all_targets, all_preds, sample_weight= sample_weights)
        f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(all_targets, all_preds, sample_weight=sample_weights, average= "macro")
        
    return {
        "accuracy": accuracy,
        "weighted_accuracy": weighted_accuracy,
        "f1_score": f1,
        "weighted_f1_score": weighted_f1,
    }



def train_one_epoch(
    model,
    optimizer,
    lr_scheduler,
    train_loader,
    val_loader,
    epoch,
    include_audio=False,  
    loss_fn=torch.nn.CrossEntropyLoss(),
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    train_loss = 0
    model.train()

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
        inputs, labels, audio_embed = batch

        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        if include_audio:
            inputs["audio_inputs"] = audio_embed.to(device) if audio_embed is not None else None 
        
        target = emotion_one_hot_encode(labels, label_map=LABEL_MAP, device=device)

        outputs = model(**inputs)
        logits = outputs["logits"]
        
        loss = loss_fn(logits, target)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        train_loss += loss.item()

    val_loss = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}"):
            inputs, labels, audio_embed = batch

            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            if include_audio:
                inputs["audio_inputs"] = audio_embed.to(device) if audio_embed is not None else None 
            
            target = emotion_one_hot_encode(labels, device=device)
            outputs = model(**inputs)
            logits = outputs["logits"]
            loss = loss_fn(logits, target)
            val_loss += loss.item()

    return train_loss, val_loss


def train_model(
    model,
    processor,
    train_loader,
    val_loader,
    test_loader,
    loss_fn=torch.nn.CrossEntropyLoss(),
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    epochs=2,
    lr=1e-6,
    tensorboard_logs="tensorboard_logs",
    output_dir="ckpt",
    include_audio=False,  
    unfreeze_on_epoch=None,
):
    # if torch.cuda.device_count() > 1:
    #     raise RuntimeError("Multiple GPUs are not supported yet. Please use a single GPU for training.")

    model.to(device)
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    
    num_training_steps = epochs * len(train_loader)
    num_warmup_steps = 0  
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    avg_train_losses = []
    avg_val_losses = []
    acc_results = []
    f1_results = []
    wacc_results = []
    wf1_results = []

    writer = SummaryWriter(log_dir=tensorboard_logs, flush_secs=240)
    best_acc = 0.0

    for epoch in range(epochs):
        torch.cuda.empty_cache()

        # === NEW: set epoch for DistributedSampler ===
        if isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        # =============================================

        if unfreeze_on_epoch is not None and epoch == unfreeze_on_epoch - 1:
            print(f"Unfreezing the model at epoch {unfreeze_on_epoch}")
            print(f"Total trainable parameters before unfreezing: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
            for p in model.florence2.vision_tower.parameters():
                p.requires_grad = True  
            current_lr = optimizer.param_groups[0]["lr"] 
            davit_params = [p for p in model.florence2.vision_tower.parameters() if p.requires_grad]
            optimizer.add_param_group({"params": davit_params,"lr": current_lr * 0.1}) 
            if include_audio:
                for p in model.wav_model.feature_extractor.parameters():
                    p.requires_grad = True
                audio_params = [p for p in model.wav_model.feature_extractor.parameters() if p.requires_grad]
                optimizer.add_param_group({"params": audio_params, "lr": current_lr * 0.1})
            print(f"Total trainable parameters after unfreezing: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
            
        train_loss, val_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            epoch=epoch,
            loss_fn=loss_fn,
            include_audio=include_audio,
            device=device
        )

        eval_results = evaluate_model(model, processor, test_loader, epoch, include_audio=include_audio, device=device)

        avg_train_losses.append(train_loss / len(train_loader))
        avg_val_losses.append(val_loss / len(val_loader))
        acc_results.append(eval_results["accuracy"])
        f1_results.append(eval_results["f1_score"])
        wacc_results.append(eval_results["weighted_accuracy"])
        wf1_results.append(eval_results["weighted_f1_score"])

        # === NEW: only rank 0 writes TensorBoard ===
        if is_main_process():
            writer.add_scalar("Loss/train", avg_train_losses[-1], epoch)
            writer.add_scalar("Loss/val", avg_val_losses[-1], epoch)
            writer.add_scalar("Accuracy/test", eval_results["accuracy"], epoch)
            writer.add_scalar("F1/test", eval_results["f1_score"], epoch)   
            writer.add_scalar("Weighted Accuracy/test", eval_results["weighted_accuracy"], epoch)
            writer.add_scalar("Weighted F1/test", eval_results["weighted_f1_score"], epoch)
            writer.add_scalar("Learning Rate", lr_scheduler.get_last_lr()[0], epoch)
        # ===========================================

        print(f"Epoch {epoch + 1}/{epochs} - Accuracy: {eval_results['accuracy']:.4f}, Weighted Accuracy: {eval_results['weighted_accuracy']:.4f}")

        # === NEW: only rank 0 saves checkpoints ===
        if is_main_process() and acc_results[-1] >= best_acc:
            best_acc = acc_results[-1]
            model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
            ckpt_path = os.path.join(output_dir, f"model_epoch_{epoch+1}_ACC_{acc_results[-1]:.4f}.pt")
            torch.save(model_to_save.state_dict(), ckpt_path)
            processor_path = os.path.join(output_dir, f"processor_epoch_{epoch+1}_ACC_{acc_results[-1]:.4f}")
            processor.save_pretrained(processor_path)
        # ==========================================

    # === NEW: sync all ranks before exit ===
    if dist.is_initialized():
        dist.barrier()
    # =======================================

    return avg_train_losses, avg_val_losses, acc_results, f1_results, wacc_results, wf1_results
