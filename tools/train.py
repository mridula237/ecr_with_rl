import os
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler, AutoProcessor, Wav2Vec2FeatureExtractor
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.Florence2 import Florence2Classifier
from src.florence2_wav import Florence2WavClassifier
from src.florence2_wav_attn import Florence2WavAttnClassifier
from src.MELD import MELD, collate_fn
from src.IEMOCAP import IEMOCAP, collate_fn_iemocap
import functools
from src.train import train_model
import argparse
import torchvision.transforms.v2 as v2

transform = v2.RandomApply([
    v2.RandomHorizontalFlip(p=1.0),
    v2.RandomRotation(degrees=15),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
], p=0.5)

# Load the processor
MODEL_PATH="microsoft/Florence-2-large-ft"
processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    # model_revision='refs/pr/6',
    trust_remote_code=True,
)

wav_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-960h")

LABEL_MAP = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}
CLASS_WEIGHTS = {"neutral": 4.0, "surprise": 15.0, "fear": 15.0, "sadness": 3.0, "joy": 1.0, "disgust": 6.0, "anger": 3.0}
ID_CLASS_WEIGHTS = {LABEL_MAP[k]: v for k, v in CLASS_WEIGHTS.items()}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args: argparse.Namespace) -> None:
    # checks
    if not args.freeze_backbone and args.unfreeze_on_epoch is not None:
        raise ValueError("If --unfreeze_on_epoch is set, you must also specify --freeze_backbone.")

    # audio_model = Florence2WavClassifier() if args.include_audio else None
    # audio_model.to(DEVICE) if audio_model else None
    if args.use_cross_attn:
        if not args.include_audio:
            raise ValueError("Cross attention requires audio features. Please set --include_audio to True.")
        print("Using Florence-2 with Wav2Vec2 and Cross Attention")
        model = Florence2WavAttnClassifier(use_cross_attn=True)
    elif args.include_audio:
        print("Using Florence-2 with Wav2Vec2")
        model = Florence2WavAttnClassifier()
    else:
        print("Using Florence-2 without audio")
        model = Florence2Classifier(device=DEVICE)        
    
    # freeze the DaViT backbone
    if args.freeze_backbone:
        print("Freezing the Florence-2 vision tower backbone")
        for p in model.florence2.vision_tower.parameters():
            p.requires_grad = False          # torch way
            p.grad = None                    # save a bit of RAM
        
        if args.freeze_encoder > 0:
            print(f"Freezing first {args.freeze_encoder} encoder layers of Florence-2 language model")
            for layer_idx, layer in enumerate(model.florence2.language_model.model.encoder.layers):
                if layer_idx < args.freeze_encoder:
                    for param in layer.parameters():
                        param.requires_grad = False
        
            
        if args.include_audio:
            print(f"freezing Wav2Vec Feature Extractor model")
            for p in model.wav_model.feature_extractor.parameters():
                p.requires_grad = False
                p.grad = None  # save a bit of RAM
            
            if args.freeze_encoder > 0:
                print(f"freezing first {args.freeze_encoder} Wav2Vec2 transformer blocks")
                for idx, layer in enumerate(model.wav_model.encoder.layers):
                    if idx < args.freeze_encoder:     
                        for p in layer.parameters():
                            p.requires_grad = False


    # ✅ Wrap for multi-GPU training
    if torch.cuda.device_count() > 1:
        raise RuntimeError("Error: Multiple GPUs are detected, but this script does not support multi-GPU training yet.")


    model.to(DEVICE)

    # ✅ Hyperparameters
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr

    # ✅ Paths
    logs_dir = args.log_dir + args.exp_name
    os.makedirs(logs_dir, exist_ok=True)
    output_dir = args.ckpt
    os.makedirs(output_dir, exist_ok=True)
    
    if args.transform:
        print("Using data augmentation transforms")
    else:
        print("No data augmentation transforms applied")
        transform = None

    # ✅ Datasets and loaders
    if args.dataset == 'MELD':
        print("Using MELD dataset")
        train_dataset = MELD(args.data_train, transform=transform)
        valid_dataset = MELD(args.data_val, transform=transform)
        test_dataset = MELD(args.data_test, transform=transform)
        _collate_fn = functools.partial(collate_fn, processor=processor, wav_feature_extractor=wav_feature_extractor)
    elif args.dataset == 'IEMOCAP':
        print("Using IEMOCAP dataset")
        train_dataset = IEMOCAP(args.data_train, transform=transform)
        valid_dataset = IEMOCAP(args.data_val, transform=transform)
        test_dataset = IEMOCAP(args.data_test, transform=transform)
        _collate_fn = functools.partial(collate_fn_iemocap, processor=processor, wav_feature_extractor=wav_feature_extractor)   
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}. Supported datasets are 'MELD' and 'IEMOCAP'.")        
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )

    if args.weighted_loss:
        print("Using weighted loss based on class distribution")
        # Ensure the weight tensor matches the number of classes
        assert len(ID_CLASS_WEIGHTS) == len(LABEL_MAP), "Mismatch between class weights and number of classes in LABEL_MAP"
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([ID_CLASS_WEIGHTS[i] for i in range(len(LABEL_MAP))]).to(DEVICE))
    else:
        print("Using unweighted loss")
        loss_fn = torch.nn.CrossEntropyLoss()
    

    # ✅ Train
    avg_train_losses, avg_val_losses, acc_results, f1_results, wacc_results, wf1_results = train_model(
        model=model,
        processor=processor,
        train_loader=train_loader,
        val_loader=valid_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        device=DEVICE,
        epochs=EPOCHS,
        lr=LR,
        tensorboard_logs=logs_dir,
        output_dir=output_dir, 
        include_audio=args.include_audio,      
        unfreeze_on_epoch=args.unfreeze_on_epoch, 
    )

    print("\n✅ Training Completed")
    print(f"📉 Avg. Train Losses: {avg_train_losses}")
    print(f"📈 Avg. Val Losses: {avg_val_losses}")
    print(f"✅ Acc Results: {acc_results}")
    print(f"✅ F1 Results: {f1_results}")
    print(f"✅ Weighted Acc Results: {wacc_results}")
    print(f"✅ Weighted F1 Results: {wf1_results}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Florence‑2 training")
    parser.add_argument("--data_train", type=str, default="/data/shared/meld/meld_modified/meld_updated/train_processed/")
    parser.add_argument("--data_val", type=str, default="/data/shared/meld/meld_modified/meld_updated/valid_processed/")
    parser.add_argument("--data_test", type=str, default="/data/shared/meld/meld_modified/meld_updated/test_processed/")
    # parser.add_argument("--data_train", type=str, default="/data/shared/meld_dataset/meldraw/train/processed_videos")
    # parser.add_argument("--data_val", type=str, default="/data/shared/meld_dataset/meldraw/dev/processed_videos_valid")
    # parser.add_argument("--data_test", type=str, default="/data/shared/meld_dataset/meldraw/test/processed_videos_test")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--log_dir", type=str, default="tensorboard_logs/", help="Directory to save TensorBoard logs DONT CHANGE")
    parser.add_argument("--exp_name", type=str, default="experiment_1", help="Name of the experiment")
    parser.add_argument("--output_dir", type=str, default="./ckpt_")
    parser.add_argument('--weighted_loss', action='store_true', help='whether to use weighted loss based on class distribution')
    parser.add_argument('--ckpt', type=str, default='ckpt_', help='Path to save checkpoint')
    parser.add_argument('--freeze_backbone', action='store_true', help='whether to freeze the backbone')
    parser.add_argument('--freeze_encoder', type=int, default=0, help='Number of encoder layers to freeze (0 means no freezing)')
    parser.add_argument('--unfreeze_on_epoch', type=int, default=None, help='Epoch to unfreeze the backbone (if applicable)')
    parser.add_argument('--transform', action='store_true', help='whether to apply data augmentation transforms')
    parser.add_argument('--include_audio', action='store_true', help='whether to include audio features in the model')
    parser.add_argument('--dataset', type=str, default='MELD', help='Dataset to use (default: MELD)')
    parser.add_argument('--use_cross_attn', action='store_true', help='whether to use cross attention in the model')
    args = parser.parse_args()
    
    if args.dataset == 'IEMOCAP':
        print("Using IEMOCAP dataset")
        args.data_train = "/data/shared/iemocop/iemocap_split/iemocap/train_data/"
        args.data_val = "/data/shared/iemocop/iemocap_split/iemocap/valid_data/"
        args.data_test = "/data/shared/iemocop/iemocap_split/iemocap/test_data/"

    # Note: torchrun will provide env vars required by setup_ddp. We just run main.
    main(args)
