
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import itertools  

from collections import defaultdict
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10

from sklearn.metrics import confusion_matrix
import seaborn as sns

import pynvml
import pandas as pd
import threading
import queue

from torch.cuda.amp import GradScaler, autocast

def calibrate_exit_times_vgg(model, device, loader, n_batches=10):
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, cannot perform precise exit time calibration. Returning zeros.")
        return [0.0] * 6
    model.training_mode = False
    model.eval()
    model.to(device)
    n_batches = min(n_batches, len(loader))
    if n_batches == 0:
        print("Warning: Loader is empty, cannot calibrate exit times.")
        return [0.0] * 6
    exit_times_ms = [0.0] * 6
    total_samples_processed = 0
    start_event = torch.cuda.Event(enable_timing=True)
    exit_events = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
    feature_blocks = [model.features1, model.features2, model.features3, model.features4, model.features5]
    exit_blocks = [model.exit1, model.exit2, model.exit3, model.exit4, model.exit5]
    with torch.no_grad():
        for images, _ in itertools.islice(loader, n_batches):
            images = images.to(device)
            batch_size = images.size(0)
            total_samples_processed += batch_size
            torch.cuda.synchronize()
            start_event.record()
            x_current = images
            for i in range(5):
                x_current = feature_blocks[i](x_current)
                _ = exit_blocks[i](x_current)
                exit_events[i].record()
            final_output = model.classifier(x_current)
            exit_events[5].record()
            torch.cuda.synchronize()
            for i in range(6):
                exit_times_ms[i] += start_event.elapsed_time(exit_events[i])
    if total_samples_processed == 0:
        print("Warning: No samples processed during calibration.")
        return [0.0] * 6
    avg_exit_times_s = [(t_ms / total_samples_processed) / 1000.0 for t_ms in exit_times_ms]
    print(f"Calibrated VGG exit times (s/sample): {avg_exit_times_s}")
    return avg_exit_times_s

class QLearningAgent:
    @staticmethod
    def _q_table_factory():
        return np.zeros(2)

    def __init__(self, n_exits, epsilon=0.1, alpha=0.1, gamma=0.9):
        self.n_exits = n_exits
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = defaultdict(QLearningAgent._q_table_factory)

    def export_q_table(self):
        return {k: v.copy() for k, v in self.q_table.items()}
        self.gamma = gamma
        self.q_table = defaultdict(QLearningAgent._q_table_factory)

    def get_state(self, layer_idx, confidence):
        conf_bin = int(confidence * 10)
        return (layer_idx, conf_bin)

    def select_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(2)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

class EarlyExitBlock(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(EarlyExitBlock, self).__init__()
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * 16, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = self.shared_conv(x)
        x = self.classifier(x)
        return x

class StaticVGG(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(StaticVGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class BranchyVGG(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(BranchyVGG, self).__init__()
        self.num_classes = num_classes
        self.training_mode = True
        self.exit_loss_weights = [0.10, 0.10, 0.10, 0.10, 0.10, 0.50]
        self.rl_agent = QLearningAgent(n_exits=5)
        self.features1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.exit1 = EarlyExitBlock(64, num_classes)
        self.features2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.exit2 = EarlyExitBlock(128, num_classes)
        self.features3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.exit3 = EarlyExitBlock(256, num_classes)
        self.features4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.exit4 = EarlyExitBlock(512, num_classes)
        self.features5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.exit5 = EarlyExitBlock(512, num_classes)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.training_mode:
            return self._forward_training(x)
        else:
            return self._forward_inference(x)

    def _forward_training(self, x):
        outputs = []
        x1 = self.features1(x)
        outputs.append(self.exit1(x1))
        x2 = self.features2(x1)
        outputs.append(self.exit2(x2))
        x3 = self.features3(x2)
        outputs.append(self.exit3(x3))
        x4 = self.features4(x3)
        outputs.append(self.exit4(x4))
        x5 = self.features5(x4)
        outputs.append(self.exit5(x5))
        outputs.append(self.classifier(x5))
        return outputs

    def _forward_inference(self, x):
        device = x.device
        batch_size = x.size(0)
        final_outputs = torch.zeros(batch_size, self.num_classes, device=device)
        exit_points = torch.zeros(batch_size, dtype=torch.int, device=device)
        remaining_indices = torch.arange(batch_size, device=device)
        x_current = x
        feature_blocks = [self.features1, self.features2, self.features3, self.features4, self.features5]
        exit_blocks = [self.exit1, self.exit2, self.exit3, self.exit4, self.exit5]
        for exit_idx, (features, exit_block) in enumerate(zip(feature_blocks, exit_blocks)):
            if len(remaining_indices) > 0:
                x_current = features(x_current)
                exit_output = exit_block(x_current)
                softmax_output = torch.softmax(exit_output, dim=1)
                confidence, _ = torch.max(softmax_output, dim=1)
                exit_decisions = []
                for i, conf in enumerate(confidence):
                    state = self.rl_agent.get_state(exit_idx, conf.item())
                    action = self.rl_agent.select_action(state, training=False)
                    exit_decisions.append(action == 0)
                exit_mask = torch.tensor(exit_decisions, dtype=torch.bool, device=device)
                exit_indices = remaining_indices[exit_mask]
                if len(exit_indices) > 0:
                    final_outputs[exit_indices] = exit_output[exit_mask]
                    exit_points[exit_indices] = exit_idx + 1
                remaining_indices = remaining_indices[~exit_mask]
                x_current = x_current[~exit_mask]
            else:
                break
        if len(remaining_indices) > 0:
            final_output = self.classifier(x_current)
            final_outputs[remaining_indices] = final_output
            exit_points[remaining_indices] = 6
        return final_outputs, exit_points

    def _calculate_reward(self, exit_idx, correct):
        base_reward = 1.0 if correct else -1.0
        early_exit_bonus = max(0, 5 - exit_idx) * 0.2
        return base_reward + early_exit_bonus

    def train_step(self, x, labels):
        device = x.device
        batch_size = x.size(0)
        outputs = self._forward_training(x)
        total_loss = 0
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        for output, weight in zip(outputs, self.exit_loss_weights):
            total_loss += weight * criterion(output, labels)
        x_current = x
        remaining_indices = torch.arange(batch_size, device=device)
        feature_blocks = [self.features1, self.features2, self.features3, self.features4, self.features5]
        exit_blocks = [self.exit1, self.exit2, self.exit3, self.exit4, self.exit5]
        for exit_idx, (features, exit_block) in enumerate(zip(feature_blocks, exit_blocks)):
            if len(remaining_indices) > 0:
                x_current = features(x_current)
                exit_output = exit_block(x_current)
                softmax_output = torch.softmax(exit_output, dim=1)
                confidence, predictions = torch.max(softmax_output, dim=1)
                for i, (conf, pred) in enumerate(zip(confidence, predictions)):
                    state = self.rl_agent.get_state(exit_idx, conf.item())
                    action = self.rl_agent.select_action(state, training=True)
                    correct = (pred == labels[remaining_indices[i]])
                    reward = self._calculate_reward(exit_idx, correct)
                    if exit_idx < len(feature_blocks) - 1:
                        next_conf = confidence[i].item()
                        next_state = self.rl_agent.get_state(exit_idx + 1, next_conf)
                        self.rl_agent.update(state, action, reward, next_state)
        return total_loss

# Data Loading
def load_datasets(batch_size=32):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader

# Training & Evaluation Functions
def train_static_vgg(model, train_loader, test_loader=None, num_epochs=100, learning_rate=0.001, weights_path=None):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    scaler = torch.amp.GradScaler() if device.type == 'cuda' else None
    best_accuracy = 0.0
    best_inference_time = float('inf')
    best_combined_score = 0.0
    best_state_dict = None
    best_epoch_metrics = {}
    history = {'train_loss': [], 'val_accuracy': [], 'inference_time': [], 'combined_score': []}
    print("Starting training with metric tracking...")
    print("Epoch | Train Loss | Val Acc | Inf Time | Combined Score | Best?")
    print("-" * 65)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            if device.type == 'cuda':
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        scheduler.step()
        if test_loader is not None:
            accuracy, inference_time = evaluate_static_vgg(model, test_loader)
            norm_accuracy = accuracy / 100.0
            max_acceptable_time = 50
            norm_time = 1 - min(inference_time / max_acceptable_time, 1)
            combined_score = (0.7 * norm_accuracy) + (0.3 * norm_time)
            history['train_loss'].append(avg_train_loss)
            history['val_accuracy'].append(accuracy)
            history['inference_time'].append(inference_time)
            history['combined_score'].append(combined_score)
            is_best = False
            if combined_score > best_combined_score:
                is_best = True
                best_combined_score = combined_score
                best_accuracy = accuracy
                best_inference_time = inference_time
                best_state_dict = model.state_dict()
                best_epoch_metrics = {
                    'epoch': epoch + 1,
                    'train_loss': avg_train_loss,
                    'val_accuracy': accuracy,
                    'inference_time': inference_time,
                    'combined_score': combined_score
                }
            print(f"{epoch+1:3d}   | {avg_train_loss:.4f}    | {accuracy:5.2f}% | {inference_time:6.2f}ms | {combined_score:.4f}      | {'*' if is_best else ''}")
    print("\nBest Epoch Metrics:")
    print(f"Epoch: {best_epoch_metrics['epoch']}")
    print(f"Training Loss: {best_epoch_metrics['train_loss']:.4f}")
    print(f"Validation Accuracy: {best_epoch_metrics['val_accuracy']:.2f}%")
    print(f"Inference Time: {best_epoch_metrics['inference_time']:.2f}ms")
    print(f"Combined Score: {best_epoch_metrics['combined_score']:.4f}")
    if weights_path and best_state_dict is not None:
        torch.save({
            'epoch': best_epoch_metrics['epoch'],
            'state_dict': best_state_dict,
            'accuracy': best_accuracy,
            'inference_time': best_inference_time,
            'combined_score': best_combined_score,
            'history': history,
        }, weights_path)
        print(f"\nBest model saved to {weights_path}")
    return model, best_epoch_metrics, history

def train_branchy_vgg(model, train_loader, test_loader, num_epochs=100, learning_rate=0.001, weights_path=None):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    scaler = torch.amp.GradScaler() if device.type == 'cuda' else None
    best_accuracy = 0.0
    best_inference_time = float('inf')
    best_combined_score = 0.0
    best_state_dict = None
    best_epoch_metrics = {}
    history = {
        'train_loss': [],
        'val_accuracy': [],
        'inference_time': [],
        'exit_distributions': [],
        'combined_score': []
    }
    print("\nStarting Branchy VGG training with metric tracking...")
    print("Epoch | Train Loss | Val Acc | Inf Time | Exit Distribution (1/2/3/4/5/F) | Combined Score | Best?")
    print("-" * 100)
    for epoch in range(num_epochs):
        model.train()
        model.training_mode = True
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            if device.type == 'cuda':
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    loss = model.train_step(images, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = model.train_step(images, labels)
                loss.backward()
                optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        model.eval()
        model.training_mode = False
        accuracy, inference_time, exit_percentages = evaluate_branchy_vgg(model, test_loader)
        norm_accuracy = accuracy / 100.0
        max_acceptable_time = 50
        norm_time = 1 - min(inference_time / max_acceptable_time, 1)
        early_exit_ratio = sum(exit_percentages[i] for i in range(1, 6)) / 100
        combined_score = (0.5 * norm_accuracy) + (0.3 * norm_time) + (0.2 * early_exit_ratio)
        history['train_loss'].append(avg_train_loss)
        history['val_accuracy'].append(accuracy)
        history['inference_time'].append(inference_time)
        history['exit_distributions'].append(exit_percentages)
        history['combined_score'].append(combined_score)
        is_best = False
        if combined_score > best_combined_score:
            is_best = True
            best_combined_score = combined_score
            best_accuracy = accuracy
            best_inference_time = inference_time
            best_state_dict = model.state_dict()
            best_epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_accuracy': accuracy,
                'inference_time': inference_time,
                'exit_distribution': exit_percentages,
                'combined_score': combined_score
            }
        exit_dist_str = f"{exit_percentages[1]:.1f}/{exit_percentages[2]:.1f}/{exit_percentages[3]:.1f}/{exit_percentages[4]:.1f}/{exit_percentages[5]:.1f}/{exit_percentages[6]:.1f}"
        print(f"{epoch+1:3d}   | {avg_train_loss:.4f}    | {accuracy:5.2f}% | {inference_time:6.2f}ms | {exit_dist_str:35} | {combined_score:.4f}      | {'*' if is_best else ''}")
        scheduler.step(avg_train_loss)
    print("\nBest Epoch Metrics:")
    print(f"Epoch: {best_epoch_metrics['epoch']}")
    print(f"Training Loss: {best_epoch_metrics['train_loss']:.4f}")
    print(f"Validation Accuracy: {best_epoch_metrics['val_accuracy']:.2f}%")
    print(f"Inference Time: {best_epoch_metrics['inference_time']:.2f}ms")
    print("Exit Distribution:")
    print(f"  Exit 1 (Very Early): {best_epoch_metrics['exit_distribution'][1]:.1f}%")
    print(f"  Exit 2 (Early): {best_epoch_metrics['exit_distribution'][2]:.1f}%")
    print(f"  Exit 3 (Middle): {best_epoch_metrics['exit_distribution'][3]:.1f}%")
    print(f"  Exit 4 (Late): {best_epoch_metrics['exit_distribution'][4]:.1f}%")
    print(f"  Exit 5 (Very Late): {best_epoch_metrics['exit_distribution'][5]:.1f}%")
    print(f"  Final Exit: {best_epoch_metrics['exit_distribution'][6]:.1f}%")
    print(f"Combined Score: {best_epoch_metrics['combined_score']:.4f}")
    if weights_path and best_state_dict is not None:
        torch.save({
            'epoch': best_epoch_metrics['epoch'],
            'state_dict': best_state_dict,
            'accuracy': best_accuracy,
            'inference_time': best_inference_time,
            'exit_distribution': best_epoch_metrics['exit_distribution'],
            'combined_score': best_combined_score,
            'history': history,
        }, weights_path)

        q_table_path = os.path.splitext(weights_path)[0] + "_q_table.npy"
        np.save(q_table_path, model.rl_agent.export_q_table())
        print(f"\nBest model saved to {weights_path}\nQ-table saved to {q_table_path}")
    return model, best_epoch_metrics, history

def evaluate_static_vgg(model, test_loader):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    correct = 0
    total_samples = 0
    total_time = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            batch_size = labels.size(0)
            start_event = None
            end_event = None
            if device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            start_cpu_time = time.time()
            outputs = model(images)
            end_cpu_time = time.time()
            batch_time = 0.0
            if device.type == 'cuda' and start_event and end_event:
                end_event.record()
                torch.cuda.synchronize()
                batch_time = start_event.elapsed_time(end_event) / 1000.0
            else:
                batch_time = end_cpu_time - start_cpu_time
            total_time += batch_time
            _, predicted = torch.max(outputs.data, 1)
            total_samples += batch_size
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total_samples if total_samples > 0 else 0
    avg_inference_time_per_sample = (total_time / total_samples) * 1000 if total_samples > 0 else 0
    return accuracy, avg_inference_time_per_sample

def get_exit_indices(model):
    """Helper to get all possible exit indices for a Branchy model based on its exit blocks."""
    indices = []
    for i in range(1, 10):  
        if hasattr(model, f"exit{i}"):
            indices.append(i)

    if hasattr(model, "classifier") and (len(indices) > 0):
        final_exit_idx = max(indices) + 1
        indices.append(final_exit_idx)
    return indices

def evaluate_branchy_vgg(model, test_loader):
    model.eval()
    model.training_mode = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    exit_indices = get_exit_indices(model)
    correct = 0
    total_samples = 0
    total_batch_processing_time = 0.0
    exit_counts = {idx: 0 for idx in exit_indices}
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            batch_size = labels.size(0)
            start_cpu_time = time.time()
            outputs, exit_points = model(images)
            end_cpu_time = time.time()
            total_batch_processing_time += (end_cpu_time - start_cpu_time)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += batch_size
            correct += (predicted == labels).sum().item()
            for exit_idx in exit_indices:
                count = (exit_points == exit_idx).sum().item()
                exit_counts[exit_idx] += count
    accuracy = 100 * correct / total_samples if total_samples > 0 else 0
    exit_percentages = {k: (v / total_samples) * 100 for k, v in exit_counts.items()} if total_samples > 0 else {k: 0 for k in exit_indices}
    print("Calibrating BranchyVGG exit times...")
    calibrated_times = calibrate_exit_times_vgg(model, device, test_loader, n_batches=20)

    if len(calibrated_times) < len(exit_indices):
        calibrated_times = list(calibrated_times) + [0.0] * (len(exit_indices) - len(calibrated_times))
    elif len(calibrated_times) > len(exit_indices):
        calibrated_times = calibrated_times[:len(exit_indices)]

    weighted_avg_time_s = 0.0
    for idx, exit_idx in enumerate(exit_indices):
        p = exit_percentages.get(exit_idx, 0) / 100.0
        t = calibrated_times[idx]
        weighted_avg_time_s += p * t
    final_inference_time_ms = weighted_avg_time_s * 1000
    print(f"Weighted Average Inference Time: {final_inference_time_ms:.2f} ms")
    return accuracy, final_inference_time_ms, exit_percentages


# Power Monitoring
class PowerMonitor:
    def __init__(self):
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.power_measurements = queue.Queue()
            self.is_monitoring = False
        except pynvml.NVMLError as error:
            print(f"Failed to initialize NVML: {error}")
            self.handle = None

    def start_monitoring(self):
        if self.handle is None:
            return
        self.is_monitoring = True
        while not self.power_measurements.empty():
            self.power_measurements.get()
        def monitor_power():
            while self.is_monitoring:
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                    self.power_measurements.put((time.time(), power))
                    time.sleep(0.005)
                except pynvml.NVMLError as error:
                    print(f"NVML Error: {error}")
                    break
        self.monitor_thread = threading.Thread(target=monitor_power)
        self.monitor_thread.start()

    def stop_monitoring(self):
        if self.handle is None:
            return
        self.is_monitoring = False
        self.monitor_thread.join()
        measurements = []
        while not self.power_measurements.empty():
            measurements.append(self.power_measurements.get())
        return pd.DataFrame(measurements, columns=['timestamp', 'power'])

def measure_power_consumption(model, test_loader, num_samples=100, device='cuda'):
    model.eval()
    model.to(device)
    power_monitor = PowerMonitor()
    results = {'avg_power': [], 'peak_power': [], 'energy': [], 'inference_time': []}
    total_samples = 0
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            batch_size = images.size(0)
            if total_samples >= num_samples:
                break
            if total_samples + batch_size > num_samples:
                images = images[:num_samples - total_samples]
                batch_size = images.size(0)
            total_samples += batch_size
            power_monitor.start_monitoring()
            start_time = time.time()
            if hasattr(model, 'training_mode'):
                model.training_mode = False
                _ = model(images)
            else:
                _ = model(images)
            end_time = time.time()
            power_data = power_monitor.stop_monitoring()
            if power_data.empty:
                print("No power data collected.")
                continue
            inference_time = end_time - start_time
            avg_power = power_data['power'].mean()
            peak_power = power_data['power'].max()
            energy = avg_power * inference_time
            results['avg_power'].append(avg_power)
            results['peak_power'].append(peak_power)
            results['energy'].append(energy)
            results['inference_time'].append(inference_time / batch_size)
    return {k: np.mean(v) if v else 0 for k, v in results.items()}

# =======================
# Visualization & Analysis
# =======================

def create_output_directory(dataset_name):
    output_dir = f'plots_{dataset_name.lower()}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def plot_comparative_analysis(static_results, branchy_results, dataset_name):
    output_dir = create_output_directory(dataset_name)
    methods = ['Static', 'Branchy']
    accuracies = [static_results['accuracy'], branchy_results['accuracy']]
    inference_times = [static_results['inference_time'], branchy_results['inference_time']]
    avg_powers = [static_results['power']['avg_power'], branchy_results['power']['avg_power']]
    peak_powers = [static_results['power']['peak_power'], branchy_results['power']['peak_power']]
    energies = [static_results['power']['energy'], branchy_results['power']['energy']]
    fig, axes = plt.subplots(1, 4, figsize=(30, 6))
    ax1 = axes[0]
    bars = ax1.bar(methods, accuracies, color=['#2ecc71', '#3498db'], width=0.6)
    ax1.set_title(f'{dataset_name.upper()} - Accuracy', fontsize=14)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_ylim(0,100)
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x()+bar.get_width()/2., h+0.5, f'{h:.2f}%', ha='center', va='bottom', fontsize=11)
    ax2 = axes[1]
    bars = ax2.bar(methods, inference_times, color=['#2ecc71', '#3498db'], width=0.6)
    ax2.set_title(f'{dataset_name.upper()} - Inference Time', fontsize=14)
    ax2.set_ylabel('Time (ms)', fontsize=12)
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x()+bar.get_width()/2., h+0.01, f'{h:.2f} ms', ha='center', va='bottom', fontsize=11)
    ax3 = axes[2]
    bars = ax3.bar(methods, avg_powers, color=['#2ecc71', '#3498db'], width=0.6)
    ax3.set_title(f'{dataset_name.upper()} - Avg Power', fontsize=14)
    ax3.set_ylabel('Power (W)', fontsize=12)
    for bar in bars:
        h = bar.get_height()
        ax3.text(bar.get_x()+bar.get_width()/2., h+0.1, f'{h:.2f}W', ha='center', va='bottom', fontsize=11)
    ax4 = axes[3]
    bars = ax4.bar(methods, peak_powers, color=['#2ecc71', '#3498db'], width=0.6)
    ax4.set_title(f'{dataset_name.upper()} - Peak Power', fontsize=14)
    ax4.set_ylabel('Power (W)', fontsize=12)
    for bar in bars:
        h = bar.get_height()
        ax4.text(bar.get_x()+bar.get_width()/2., h+0.1, f'{h:.2f}W', ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name.lower()}_comparative_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(8,6))
    bars = plt.bar(methods, energies, color=['#2ecc71', '#3498db'], width=0.6)
    plt.title(f'{dataset_name.upper()} - Energy Consumption', fontsize=14)
    plt.ylabel('Energy (Joules)', fontsize=12)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x()+bar.get_width()/2., h+0.01, f'{h:.2f}J', ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name.lower()}_energy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_exit_distribution(exit_percentages, dataset_name):
    output_dir = create_output_directory(dataset_name)
    exit_labels = ['Exit 1\n(Very Early)', 'Exit 2\n(Early)', 'Exit 3\n(Middle)', 'Exit 4\n(Late)', 'Exit 5\n(Very Late)', 'Final\nExit']
    percentages = [exit_percentages[i] for i in range(1, 7)]
    plt.figure(figsize=(12,6))
    bars = plt.bar(exit_labels, percentages, color=['#2980b9', '#3498db', '#85c1e9', '#aed6f1', '#d6eaf8', '#e74c3c'])
    plt.title(f'{dataset_name.upper()} - Exit Distribution', fontsize=14)
    plt.ylabel('Percentage of Samples (%)', fontsize=12)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.1f}%', ha='center', va='bottom')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name.lower()}_exit_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_exit_distribution(model, test_loader, dataset_name):
    model.eval()
    model.training_mode = False
    device = next(model.parameters()).device
    exit_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    class_distributions = {1: defaultdict(int), 2: defaultdict(int), 3: defaultdict(int), 4: defaultdict(int), 5: defaultdict(int), 6: defaultdict(int)}
    total_samples = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            batch_size = labels.size(0)
            total_samples += batch_size
            outputs, exit_points = model(images)
            _, predicted = torch.max(outputs.data, 1)
            for exit_idx in range(1, 7):
                mask = (exit_points == exit_idx)
                count = mask.sum().item()
                exit_counts[exit_idx] += count
                if count > 0:
                    exit_labels = labels[mask]
                    for label in exit_labels:
                        class_distributions[exit_idx][label.item()] += 1
    exit_distribution = {k: (v/total_samples)*100 for k,v in exit_counts.items()}
    return exit_distribution, class_distributions

def plot_class_distribution(class_distributions, dataset_name):
    output_dir = create_output_directory(dataset_name)
    class_names = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
    fig, axes = plt.subplots(2, 3, figsize=(20,15))
    fig.suptitle(f'{dataset_name.upper()} - Class Distributions Across Exits', fontsize=16)
    exit_names = ['Very Early Exit (1)', 'Early Exit (2)', 'Middle Exit (3)', 'Late Exit (4)', 'Very Late Exit (5)', 'Final Exit']
    total_samples = sum(sum(dist.values()) for dist in class_distributions.values())
    for idx, (exit_idx, distribution) in enumerate(class_distributions.items()):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        classes = sorted(distribution.keys())
        counts = [distribution[c] for c in classes]
        percentages = [(count/total_samples)*100 for count in counts]
        bars = ax.bar([class_names[i] for i in classes], percentages, color='#3498db')
        ax.set_title(f'{exit_names[idx-1]} Class Distribution', fontsize=14)
        ax.set_xlabel('Class Label', fontsize=12)
        ax.set_ylabel('Percentage of Samples (%)', fontsize=12)
        ax.set_xticklabels([class_names[i] for i in classes], rotation=45, ha='right')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name.lower()}_class_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_time_comparison(static_time, branchy_time, dataset_name):
    output_dir = create_output_directory(dataset_name)
    methods = ['Static ResNet', 'Branchy ResNet']
    times = [static_time, branchy_time]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, times, color=['#2ecc71', '#3498db'], width=0.6)
    plt.title(f'{dataset_name.upper()} - Training Time Comparison', fontsize=14)
    plt.ylabel('Training Time (seconds)', fontsize=12)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}s', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name.lower()}_training_time.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(model, test_loader, is_branchy=False, dataset_name='cifar10'):
    output_dir = create_output_directory(dataset_name)
    model.eval()
    if is_branchy:
        model.training_mode = False
    device = next(model.parameters()).device
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            if is_branchy:
                outputs, _ = model(images)
            else:
                outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{dataset_name.upper()} - {"Branchy" if is_branchy else "Static"} ResNet18\nConfusion Matrix', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    plt.tight_layout()
    model_type = 'branchy' if is_branchy else 'static'
    plt.savefig(os.path.join(output_dir, f'{dataset_name.lower()}_{model_type}_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Experiment Runner
def run_experiments():
    print("\nRunning experiments on CIFAR-10...")
    train_loader, test_loader = load_datasets(batch_size=128)
    dataset_name = 'cifar10'
    weights_dir = 'pretrained_weights'
    os.makedirs(weights_dir, exist_ok=True)
    static_weights_path = os.path.join(weights_dir, 'static_vgg_cifar10.pth')
    branchy_weights_path = os.path.join(weights_dir, 'branchy_vgg_cifar10.pth')
    print("\nInitializing Static VGG...")
    static_vgg = StaticVGG(num_classes=10, in_channels=3)
    static_vgg = static_vgg.to(device)
    static_start_time = time.time()
    if os.path.exists(static_weights_path):
        print("Loading pre-trained Static VGG weights...")
        checkpoint = torch.load(static_weights_path)
        static_vgg.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded weights from epoch {checkpoint['epoch']} with validation accuracy: {checkpoint['accuracy']:.2f}%")
        static_model = static_vgg
        static_best_metrics = {'accuracy': checkpoint['accuracy'], 'inference_time': checkpoint.get('inference_time', None)}
    else:
        print("Training Static VGG from scratch...")
        static_model, static_best_metrics, static_history = train_static_vgg(static_vgg, train_loader, test_loader, num_epochs=100, learning_rate=0.001, weights_path=static_weights_path)
    static_training_time = time.time() - static_start_time
    print("\nEvaluating Static VGG...")
    # Now returns per-sample time in ms
    static_accuracy, static_inference_time_per_sample = evaluate_static_vgg(static_model, test_loader)
    print(f"Static VGG Results:")
    print(f"Accuracy: {static_accuracy:.2f}%")
    print(f"Average Inference Time (per sample): {static_inference_time_per_sample:.2f} ms")
    print("\nMeasuring power consumption for Static VGG...")
    static_power = measure_power_consumption(static_model, test_loader, num_samples=100)
    static_best_metrics['inference_time'] = static_inference_time_per_sample

    print("\nInitializing Branchy VGG...")
    branchy_vgg = BranchyVGG(num_classes=10, in_channels=3)
    branchy_vgg = branchy_vgg.to(device)
    branchy_start_time = time.time()
    if os.path.exists(branchy_weights_path):
        print("Loading pre-trained Branchy VGG weights...")
        checkpoint = torch.load(branchy_weights_path)
        branchy_vgg.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded weights from epoch {checkpoint['epoch']} with validation accuracy: {checkpoint['accuracy']:.2f}%")
        branchy_model = branchy_vgg
        branchy_best_metrics = {'accuracy': checkpoint['accuracy'], 'inference_time': checkpoint.get('inference_time', None), 'exit_distribution': checkpoint.get('exit_distribution', None)}
    else:
        print("Training Branchy VGG from scratch...")
        branchy_model, branchy_best_metrics, branchy_history = train_branchy_vgg(branchy_vgg, train_loader, test_loader, num_epochs=100, learning_rate=0.001, weights_path=branchy_weights_path)
    branchy_training_time = time.time() - branchy_start_time
    print("\nEvaluating Branchy VGG...")
    final_accuracy, final_inference_time_weighted_ms, exit_percentages = evaluate_branchy_vgg(branchy_model, test_loader)
    print(f"Branchy VGG Results:")
    print(f"Accuracy: {final_accuracy:.2f}%")
    print(f"Average Inference Time (weighted, per sample): {final_inference_time_weighted_ms:.2f} ms")
    print(f"Exit Distribution: {exit_percentages}")
    print("\nMeasuring power consumption for Branchy VGG...")
    branchy_power = measure_power_consumption(branchy_model, test_loader, num_samples=100)
    branchy_best_metrics['inference_time'] = final_inference_time_weighted_ms
    branchy_best_metrics['exit_distribution'] = exit_percentages

    speed_improvement = ((static_inference_time_per_sample - final_inference_time_weighted_ms) / static_inference_time_per_sample) * 100 if static_inference_time_per_sample > 0 else 0
    accuracy_difference = final_accuracy - static_accuracy
    energy_savings = ((static_power['energy'] - branchy_power['energy']) / static_power['energy'] * 100) if static_power['energy'] > 0 else 0
    results = {
        'static': {'accuracy': static_accuracy, 'inference_time': static_inference_time_per_sample, 'power': static_power, 'training_time': static_training_time, 'best_metrics': static_best_metrics},
        'branchy': {'accuracy': final_accuracy, 'inference_time': final_inference_time_weighted_ms, 'exit_percentages': exit_percentages, 'power': branchy_power, 'training_time': branchy_training_time, 'best_metrics': branchy_best_metrics},
        'improvements': {'speed': speed_improvement, 'accuracy': accuracy_difference, 'energy_savings': energy_savings}
    }
    print("\nResults Summary:")
    print(f"Static VGG - Accuracy: {static_accuracy:.2f}%, Inference Time (per sample): {static_inference_time_per_sample:.2f}ms, Energy: {static_power.get('energy', 0):.2f}J")
    print(f"Branchy VGG - Accuracy: {final_accuracy:.2f}%, Inference Time (weighted, per sample): {final_inference_time_weighted_ms:.2f}ms, Energy: {branchy_power.get('energy', 0):.2f}J")
    print(f"Speed Improvement: {speed_improvement:.1f}%")
    print(f"Accuracy Difference: {accuracy_difference:+.2f}%")
    print(f"Energy Savings: {energy_savings:.1f}%")
    print(f"Exit Distribution: {exit_percentages}")
    print("\nGenerating plots...")
    plot_comparative_analysis(results['static'], results['branchy'], dataset_name)
    plot_exit_distribution(exit_percentages, dataset_name)
    plot_class_distribution(analyze_exit_distribution(branchy_model, test_loader, dataset_name)[1], dataset_name)
    plot_training_time_comparison(static_training_time, branchy_training_time, dataset_name)
    plot_confusion_matrix(static_model, test_loader, is_branchy=False, dataset_name=dataset_name)
    plot_confusion_matrix(branchy_model, test_loader, is_branchy=True, dataset_name=dataset_name)
    return results

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    results = run_experiments()
    print("\nAll experiments completed.")
