"""
 python evaluate_single_eeg_window.py --window 3
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report, accuracy_score
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import time
import gc
import argparse

from short_window_eeg_model import ShortWindowAttnSleep
from short_window_eeg_dataset import get_short_window_eeg_dataloaders
from collections import Counter


class SingleEEGWindowEvaluator:


    def __init__(self, config, window_minutes):
        self.config = config
        self.window_minutes = window_minutes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")


        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(config['output']['save_dir'],
                                       f'eeg_window_{window_minutes}min_{timestamp}')
        os.makedirs(self.output_dir, exist_ok=True)

    def calculate_class_weights(self, train_dataset, sample_size=200):

        print("\nCalculating class weights...")

        all_labels = []
        sample_size = min(len(train_dataset), sample_size)

        for idx in tqdm(range(sample_size), desc="Sampling labels"):
            _, labels = train_dataset[idx]
            all_labels.extend(labels.numpy().tolist())

        label_counts = Counter(all_labels)
        class_counts = [label_counts.get(i, 1) for i in range(4)]
        total_samples = sum(class_counts)

        print(f"\nLabel distribution (sampled {sample_size} windows):")
        stage_names = ['Wake', 'Light', 'Deep', 'REM']
        for i, count in enumerate(class_counts):
            percentage = count / total_samples * 100
            print(f"  {stage_names[i]}: {count} samples ({percentage:.2f}%)")

        class_weights = torch.tensor([total_samples / (4 * count) for count in class_counts],
                                     dtype=torch.float32)

        return class_weights.to(self.device)

    def train_epoch(self, model, dataloader, optimizer, criterion, scheduler=None):

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc="Training")
        for batch_idx, (eeg, labels) in enumerate(pbar):
            eeg = eeg.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()


            outputs = model(eeg)  # (batch, num_epochs, num_classes)


            batch_size, num_epochs, num_classes = outputs.shape
            outputs_flat = outputs.view(-1, num_classes)  # (batch * num_epochs, num_classes)
            labels_flat = labels.view(-1)  # (batch * num_epochs,)

            loss = criterion(outputs_flat, labels_flat)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()


            _, predicted = outputs_flat.max(1)
            correct += predicted.eq(labels_flat).sum().item()
            total += labels_flat.numel()
            running_loss += loss.item() * labels_flat.numel()

            if total > 0:
                pbar.set_postfix({
                    'loss': running_loss / total,
                    'acc': correct / total
                })

            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

        return running_loss / total if total > 0 else 0, correct / total if total > 0 else 0

    def validate(self, model, dataloader, criterion):

        model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for eeg, labels in tqdm(dataloader, desc="Validation"):
                eeg = eeg.to(self.device)
                labels = labels.to(self.device)

                outputs = model(eeg)

                batch_size, num_epochs, num_classes = outputs.shape
                outputs_flat = outputs.view(-1, num_classes)
                labels_flat = labels.view(-1)

                loss = criterion(outputs_flat, labels_flat)

                running_loss += loss.item() * labels_flat.numel()
                _, predicted = outputs_flat.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels_flat.cpu().numpy())

        epoch_loss = running_loss / len(all_labels) if all_labels else 0
        accuracy = accuracy_score(all_labels, all_preds)
        kappa = cohen_kappa_score(all_labels, all_preds)

        cm = confusion_matrix(all_labels, all_preds)
        per_class_precision = np.zeros(4)
        per_class_recall = np.zeros(4)

        for i in range(4):
            if cm[:, i].sum() > 0:
                per_class_precision[i] = cm[i, i] / cm[:, i].sum()
            if cm[i, :].sum() > 0:
                per_class_recall[i] = cm[i, i] / cm[i, :].sum()

        per_class_metrics = {
            'precision': per_class_precision,
            'recall': per_class_recall
        }

        return epoch_loss, accuracy, kappa, all_preds, all_labels, per_class_metrics

    def measure_inference_performance(self, model, dataloader, n_warmup=5, n_measure=20):

        model.eval()

        test_batch = next(iter(dataloader))
        eeg = test_batch[0].to(self.device)

        print("Warming up...")
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = model(eeg)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        print("Measuring inference time...")
        inference_times = []

        with torch.no_grad():
            for _ in range(n_measure):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.perf_counter()

                _ = model(eeg)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.perf_counter()

                inference_times.append(end_time - start_time)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = model(eeg)
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 3
        else:
            peak_memory = 0.0

        avg_inference_time = np.mean(inference_times)
        std_inference_time = np.std(inference_times)
        batch_size = eeg.size(0)
        throughput = batch_size / avg_inference_time
        epoch_count = eeg.size(1)
        time_per_epoch = avg_inference_time / epoch_count

        performance_metrics = {
            'batch_size': batch_size,
            'epoch_count': epoch_count,
            'avg_inference_time_ms': avg_inference_time * 1000,
            'std_inference_time_ms': std_inference_time * 1000,
            'throughput_samples_per_sec': throughput,
            'time_per_epoch_ms': time_per_epoch * 1000,
            'peak_gpu_memory_gb': peak_memory
        }

        return performance_metrics

    def train_and_evaluate(self):

        print(f"\n{'=' * 80}")
        print(f"Training and Evaluating {self.window_minutes}-minute window EEG model")
        print('=' * 80)


        print("\nLoading data...")
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = \
            get_short_window_eeg_dataloaders(
                self.config['data']['folders'],
                window_minutes=self.window_minutes,
                batch_size=int(self.config['training']['batch_size']),
                num_workers=int(self.config['data']['num_workers']),
                testset_path=self.config['data'].get('testset_path', 'testset.json')
            )


        print("\nCreating model...")
        model = ShortWindowAttnSleep(
            window_minutes=self.window_minutes,
            num_classes=4
        ).to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        class_weights = self.calculate_class_weights(train_dataset)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        learning_rate = float(self.config['training']['learning_rate'])
        weight_decay = float(self.config['training']['weight_decay'])

        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        num_epochs = int(self.config['training']['num_epochs'])
        total_steps = len(train_loader) * num_epochs
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )

        best_kappa = -1
        best_epoch = 0
        patience_counter = 0

        train_start_time = time.time()
        patience = int(self.config['training']['patience'])

        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()
            print(f"\nEpoch {epoch}/{num_epochs}")

            train_loss, train_acc = self.train_epoch(
                model, train_loader, optimizer, criterion, scheduler
            )

            val_loss, val_acc, val_kappa, _, _, _ = self.validate(model, val_loader, criterion)

            epoch_time = time.time() - epoch_start_time
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Kappa: {val_kappa:.4f}")
            print(f"Epoch time: {epoch_time:.1f}s")

            if val_kappa > best_kappa:
                best_kappa = val_kappa
                best_epoch = epoch
                patience_counter = 0

                checkpoint_path = os.path.join(self.output_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_kappa': best_kappa
                }, checkpoint_path)
                print(f"Saved best model with kappa: {best_kappa:.4f}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

            torch.cuda.empty_cache()
            gc.collect()

        train_end_time = time.time()
        total_training_time = train_end_time - train_start_time

        print(f"\nBest validation kappa: {best_kappa:.4f} at epoch {best_epoch}")
        print(f"Total training time: {total_training_time:.2f} seconds ({total_training_time / 60:.1f} minutes)")


        checkpoint = torch.load(os.path.join(self.output_dir, 'best_model.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])


        print("\nEvaluating on test set...")
        test_loss, test_acc, test_kappa, test_preds, test_labels, test_per_class = \
            self.validate(model, test_loader, criterion)

        print(f"\nTest Results:")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  Kappa: {test_kappa:.4f}")

        print("\nClassification Report:")
        print(classification_report(
            test_labels, test_preds,
            target_names=['Wake', 'Light', 'Deep', 'REM']
        ))


        print("\nMeasuring inference performance...")
        inference_metrics = self.measure_inference_performance(model, test_loader)

        print(f"\nInference Performance:")
        print(f"  Batch size: {inference_metrics['batch_size']}")
        print(f"  Epochs per window: {inference_metrics['epoch_count']}")
        print(f"  Average inference time: {inference_metrics['avg_inference_time_ms']:.2f} ± "
              f"{inference_metrics['std_inference_time_ms']:.2f} ms")
        print(f"  Throughput: {inference_metrics['throughput_samples_per_sec']:.2f} samples/sec")
        print(f"  Time per epoch: {inference_metrics['time_per_epoch_ms']:.4f} ms")
        print(f"  Peak GPU memory: {inference_metrics['peak_gpu_memory_gb']:.2f} GB")


        results = {
            'window_minutes': self.window_minutes,
            'model_params': {
                'total': total_params,
                'trainable': trainable_params
            },
            'training': {
                'best_epoch': best_epoch,
                'total_time_seconds': total_training_time,
                'time_per_epoch_seconds': total_training_time / best_epoch
            },
            'performance': {
                'test_accuracy': float(test_acc),
                'test_kappa': float(test_kappa),
                'test_loss': float(test_loss)
            },
            'per_class_metrics': {
                'precision': test_per_class['precision'].tolist(),
                'recall': test_per_class['recall'].tolist()
            },
            'inference_metrics': inference_metrics,
            'confusion_matrix': confusion_matrix(test_labels, test_preds).tolist()
        }


        self.plot_confusion_matrix(test_labels, test_preds)


        with open(os.path.join(self.output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {self.output_dir}")

        return results

    def plot_confusion_matrix(self, labels, preds):

        cm = confusion_matrix(labels, preds)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                    xticklabels=['Wake', 'Light', 'Deep', 'REM'],
                    yticklabels=['Wake', 'Light', 'Deep', 'REM'])
        plt.title(f'{self.window_minutes}-minute Window EEG Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'), dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate Single Window Length EEG Model')
    parser.add_argument('--window', type=float, required=True,
                        choices=[0.5, 1, 3, 5, 10, 30],
                        help='Window length in minutes (0.5=30s, 1, 3, 5, 10, or 30)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    args = parser.parse_args()


    config = {
        'data': {
            'folders': [
                '../../data/eeg-1',
                '../../data/eeg-2',
                '../../data/eeg-3'
            ],
            'testset_path': 'testset.json',
            'num_workers': 4
        },
        'training': {
            'batch_size': args.batch_size,
            'num_epochs': args.epochs,
            'learning_rate': args.lr,
            'weight_decay': 1e-5,
            'patience': 10
        },
        'output': {
            'save_dir': "./outputs"
        }
    }


    evaluator = SingleEEGWindowEvaluator(config, args.window)
    results = evaluator.train_and_evaluate()

    print(f"\n{'=' * 80}")
    print(f"COMPLETED: {args.window}-minute window EEG model")
    print(f"Results saved to: {evaluator.output_dir}")
    print('=' * 80)


if __name__ == "__main__":
    main()