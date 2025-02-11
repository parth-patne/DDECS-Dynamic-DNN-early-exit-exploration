# -*- coding: utf-8 -*-
"""dynamic-resnet18-early-exits.py
This code implements a BranchyResNet18 model with only two early exits:
- Exit 1 after layer1
- Exit 3 after layer3
and then the final exit after layer4 (Exit 4).
"""

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

from collections import defaultdict
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from sklearn.metrics import confusion_matrix
import seaborn as sns

import pynvml
import pandas as pd
import threading
import queue

from torch.cuda.amp import GradScaler, autocast

# -------------------------------------------------------
# Q-Learning Agent
# -------------------------------------------------------
class QLearningAgent:
    def __init__(self, n_exits, epsilon=0.1, alpha=0.1, gamma=0.9):
        self.n_exits = n_exits
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = defaultdict(lambda: np.zeros(2))

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

# -------------------------------------------------------
# Early Exit Block
# -------------------------------------------------------
class EarlyExitBlock(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(EarlyExitBlock, self).__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((4,4)),
            nn.Flatten(),
            nn.Linear(in_channels * 16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.head(x)

# -------------------------------------------------------
# ResNet-18 Building Blocks
# -------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# -------------------------------------------------------
# Static ResNet-18
# -------------------------------------------------------
class StaticResNet18(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(StaticResNet18, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)
        self._initialize_weights()

    def _make_layer(self, planes, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            s = stride if i == 0 else 1
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x

# -------------------------------------------------------
# Branchy ResNet-18 with two early exits (Exit 1 after layer1 and Exit 3 after layer3)
# -------------------------------------------------------
class BranchyResNet18(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(BranchyResNet18, self).__init__()
        self.num_classes = num_classes
        self.training_mode = True
        self.exit_loss_weights = [0.10, 0.10, 0.80]  # For Exit 1, Exit 3, Final (Exit 4)
        self.rl_agent = QLearningAgent(n_exits=2)
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.exit1 = EarlyExitBlock(64, num_classes)  # Exit 1
        self.layer2 = self._make_layer(128, 2, stride=2)  # No exit here
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.exit3 = EarlyExitBlock(256, num_classes)  # Exit 3
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)
        self._initialize_weights()

    def _make_layer(self, planes, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            s = stride if i == 0 else 1
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
        x = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x)
        outputs.append(self.exit1(x1))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        outputs.append(self.exit3(x3))
        x4 = self.layer4(x3)
        x_final = self.avgpool(x4)
        x_final = torch.flatten(x_final, 1)
        outputs.append(self.fc(x_final))
        return outputs

    def _forward_inference(self, x):
        device = x.device
        batch_size = x.size(0)
        final_outputs = torch.zeros(batch_size, self.num_classes, device=device)
        exit_points = torch.zeros(batch_size, dtype=torch.int, device=device)
        remaining_indices = torch.arange(batch_size, device=device)
        x_current = self.relu(self.bn1(self.conv1(x)))
        # Exit 1 after layer1
        if len(remaining_indices) > 0:
            x_current = self.layer1(x_current)
            exit_output = self.exit1(x_current)
            softmax_output = torch.softmax(exit_output, dim=1)
            confidence = torch.max(softmax_output, dim=1)[0]
            exit_decisions = [self.rl_agent.select_action(self.rl_agent.get_state(0, conf.item()), training=False) == 0 for conf in confidence]
            exit_mask = torch.tensor(exit_decisions, dtype=torch.bool, device=device)
            exit_indices = remaining_indices[exit_mask]
            if len(exit_indices) > 0:
                final_outputs[exit_indices] = exit_output[exit_mask]
                exit_points[exit_indices] = 1
            remaining_indices = remaining_indices[~exit_mask]
            x_current = x_current[~exit_mask]
        # Pass through layer2 (no exit)
        if len(remaining_indices) > 0:
            x_current = self.layer2(x_current)
        # Exit 3 after layer3
        if len(remaining_indices) > 0:
            x_current = self.layer3(x_current)
            exit_output = self.exit3(x_current)
            softmax_output = torch.softmax(exit_output, dim=1)
            confidence = torch.max(softmax_output, dim=1)[0]
            exit_decisions = [self.rl_agent.select_action(self.rl_agent.get_state(1, conf.item()), training=False) == 0 for conf in confidence]
            exit_mask = torch.tensor(exit_decisions, dtype=torch.bool, device=device)
            exit_indices = remaining_indices[exit_mask]
            if len(exit_indices) > 0:
                final_outputs[exit_indices] = exit_output[exit_mask]
                exit_points[exit_indices] = 3
            remaining_indices = remaining_indices[~exit_mask]
            x_current = x_current[~exit_mask]
        # Final exit after layer4 (Exit 4)
        if len(remaining_indices) > 0:
            x_current = self.layer4(x_current)
            x_final = self.avgpool(x_current)
            x_final = torch.flatten(x_final, 1)
            final_output = self.fc(x_final)
            final_outputs[remaining_indices] = final_output
            exit_points[remaining_indices] = 4
        return final_outputs, exit_points

    def _calculate_reward(self, exit_idx, correct):
        base_reward = 1.0 if correct else -1.0
        early_exit_bonus = (1 - exit_idx) * 0.2
        return base_reward + early_exit_bonus

    def train_step(self, x, labels):
        device = x.device
        batch_size = x.size(0)
        outputs = self._forward_training(x)
        total_loss = 0
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        for output, weight in zip(outputs, self.exit_loss_weights):
            total_loss += weight * criterion(output, labels)
        x_current = self.relu(self.bn1(self.conv1(x)))
        remaining_indices = torch.arange(batch_size, device=device)
        # RL update at Exit 1
        if len(remaining_indices) > 0:
            x_current = self.layer1(x_current)
            exit_output = self.exit1(x_current)
            softmax_output = torch.softmax(exit_output, dim=1)
            confidence, predictions = torch.max(softmax_output, dim=1)
            for i, (conf, pred) in enumerate(zip(confidence, predictions)):
                state = self.rl_agent.get_state(0, conf.item())
                action = self.rl_agent.select_action(state, training=True)
                correct = (pred == labels[remaining_indices[i]])
                reward = self._calculate_reward(0, correct)
                next_state = self.rl_agent.get_state(1, conf.item())
                self.rl_agent.update(state, action, reward, next_state)
        # RL update at Exit 3
        if len(remaining_indices) > 0:
            x_current = self.layer2(x_current)
            x_current = self.layer3(x_current)
            exit_output = self.exit3(x_current)
            softmax_output = torch.softmax(exit_output, dim=1)
            confidence, predictions = torch.max(softmax_output, dim=1)
            for i, (conf, pred) in enumerate(zip(confidence, predictions)):
                state = self.rl_agent.get_state(1, conf.item())
                action = self.rl_agent.select_action(state, training=True)
                correct = (pred == labels[remaining_indices[i]])
                reward = self._calculate_reward(1, correct)
                next_state = state
                self.rl_agent.update(state, action, reward, next_state)
        return total_loss

# -------------------------------------------------------
# Data Loading
# -------------------------------------------------------
def load_datasets(batch_size=128):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader

# -------------------------------------------------------
# Training & Evaluation Functions
# -------------------------------------------------------
def train_static_resnet(model, train_loader, test_loader=None, num_epochs=100, learning_rate=0.001, weights_path=None):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    scaler = GradScaler() if device.type == 'cuda' else None
    best_accuracy = 0.0
    best_state_dict = None
    static_train_start = time.time()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            if scaler:
                with autocast():
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
        avg_loss = running_loss / len(train_loader)
        scheduler.step()
        if test_loader is not None:
            accuracy, _ = evaluate_static_resnet(model, test_loader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_state_dict = model.state_dict()
        else:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    static_train_end = time.time()
    static_training_time = static_train_end - static_train_start
    if weights_path and best_state_dict is not None:
        torch.save({'state_dict': best_state_dict, 'accuracy': best_accuracy}, weights_path)
    return model, static_training_time

def train_branchy_resnet(model, train_loader, test_loader, num_epochs=100, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    scaler = GradScaler() if device.type == 'cuda' else None
    best_accuracy = 0
    best_model_state = None
    all_exit_distributions = []
    all_accuracies = []
    all_inference_times = []
    branchy_train_start = time.time()
    for epoch in range(num_epochs):
        model.train()
        model.training_mode = True
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            if scaler:
                with autocast():
                    loss = model.train_step(images, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = model.train_step(images, labels)
                loss.backward()
                optimizer.step()
            running_loss += loss.item()
        scheduler.step(running_loss/len(train_loader))
        model.eval()
        model.training_mode = False
        accuracy, inference_time, exit_percentages = evaluate_branchy_resnet(model, test_loader)
        all_accuracies.append(accuracy)
        all_inference_times.append(inference_time)
        all_exit_distributions.append(exit_percentages)
        print(f'Epoch [{epoch+1}/{num_epochs}] - Accuracy: {accuracy:.2f}%, Inference Time: {inference_time:.2f} ms')
        print(f'Exit Distribution: {exit_percentages}')
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state = model.state_dict()
    branchy_train_end = time.time()
    branchy_training_time = branchy_train_end - branchy_train_start
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    avg_exit_distribution = {}
    if len(all_exit_distributions) > 0:
        keys = all_exit_distributions[0].keys()
        for k in keys:
            avg_exit_distribution[k] = np.mean([dist[k] for dist in all_exit_distributions])
    return model, branchy_training_time, all_exit_distributions, avg_exit_distribution

def evaluate_static_resnet(model, test_loader):
    model.eval()
    device = next(model.parameters()).device
    correct = 0
    total = 0
    inference_times = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()
            outputs = model(images)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            _, predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    avg_inference_time = (sum(inference_times)/len(inference_times))*1000
    return accuracy, avg_inference_time

def evaluate_branchy_resnet(model, test_loader):
    model.eval()
    model.training_mode = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    correct = 0
    total = 0
    inference_times = []
    exit_counts = {1:0, 3:0, 4:0}
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            batch_size = labels.size(0)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()
            outputs, exit_points = model(images)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            _, predicted = torch.max(outputs.data,1)
            total += batch_size
            correct += (predicted == labels).sum().item()
            for i in [1,3,4]:
                count = (exit_points == i).sum().item()
                exit_counts[i] += count
    accuracy = 100 * correct / total
    avg_inference_time = (sum(inference_times)/total)*1000
    exit_percentages = {k:(v/total)*100 for k,v in exit_counts.items()}
    return accuracy, avg_inference_time, exit_percentages

# -------------------------------------------------------
# Power Monitoring
# -------------------------------------------------------
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
                    power = pynvml.nvmlDeviceGetPowerUsage(self.handle)/1000.0
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
        return pd.DataFrame(measurements, columns=['timestamp','power'])

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
            if hasattr(model,'training_mode'):
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
    return {k: np.mean(v) if v else 0 for k,v in results.items()}

# -------------------------------------------------------
# Visualization & Analysis Functions
# -------------------------------------------------------
def create_output_directory(dataset_name):
    output_dir = f'plots_{dataset_name.lower()}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def plot_confusion_matrix(y_true, y_pred, class_names, title, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_comparative_analysis(static_results, branchy_results, dataset_name):
    output_dir = create_output_directory(dataset_name)
    methods = ['Static','Branchy']
    accuracies = [static_results['accuracy'], branchy_results['accuracy']]
    inference_times = [static_results['inference_time'], branchy_results['inference_time']]
    avg_powers = [static_results['power']['avg_power'], branchy_results['power']['avg_power']]
    peak_powers = [static_results['power']['peak_power'], branchy_results['power']['peak_power']]
    energies = [static_results['power']['energy'], branchy_results['power']['energy']]
    fig, axes = plt.subplots(1,4, figsize=(30,6))
    ax1 = axes[0]
    bars = ax1.bar(methods, accuracies, color=['#2ecc71','#3498db'], width=0.6)
    ax1.set_title(f'{dataset_name.upper()} - Accuracy', fontsize=14)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_ylim(0,100)
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x()+bar.get_width()/2., h+0.5, f'{h:.2f}%', ha='center', va='bottom', fontsize=11)
    ax2 = axes[1]
    bars = ax2.bar(methods, inference_times, color=['#2ecc71','#3498db'], width=0.6)
    ax2.set_title(f'{dataset_name.upper()} - Inference Time', fontsize=14)
    ax2.set_ylabel('Time (ms)', fontsize=12)
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x()+bar.get_width()/2., h+0.01, f'{h:.2f} ms', ha='center', va='bottom', fontsize=11)
    ax3 = axes[2]
    bars = ax3.bar(methods, avg_powers, color=['#2ecc71','#3498db'], width=0.6)
    ax3.set_title(f'{dataset_name.upper()} - Avg Power', fontsize=14)
    ax3.set_ylabel('Power (W)', fontsize=12)
    for bar in bars:
        h = bar.get_height()
        ax3.text(bar.get_x()+bar.get_width()/2., h+0.1, f'{h:.2f}W', ha='center', va='bottom', fontsize=11)
    ax4 = axes[3]
    bars = ax4.bar(methods, peak_powers, color=['#2ecc71','#3498db'], width=0.6)
    ax4.set_title(f'{dataset_name.upper()} - Peak Power', fontsize=14)
    ax4.set_ylabel('Power (W)', fontsize=12)
    for bar in bars:
        h = bar.get_height()
        ax4.text(bar.get_x()+bar.get_width()/2., h+0.1, f'{h:.2f}W', ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name.lower()}_comparative_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(8,6))
    bars = plt.bar(methods, energies, color=['#2ecc71','#3498db'], width=0.6)
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
    filtered_exits = {k: v for k, v in exit_percentages.items() if k in [1,3,4]}
    exits = list(filtered_exits.keys())
    percentages = list(filtered_exits.values())
    plt.figure(figsize=(10,6))
    plt.bar([f'Exit {i}' for i in exits], percentages, color='#e74c3c', width=0.6)
    plt.title(f'{dataset_name.upper()} - Exit Distribution', fontsize=14)
    plt.ylabel('Percentage of Samples (%)', fontsize=12)
    for i, perc in enumerate(percentages):
        plt.text(i, perc+0.5, f'{perc:.2f}%', ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name.lower()}_exit_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_exit_distribution(model, test_loader, dataset_name):
    model.eval()
    model.training_mode = False
    device = next(model.parameters()).device
    exit_counts = {1:0, 3:0, 4:0}
    class_distributions = {1: defaultdict(int), 3: defaultdict(int), 4: defaultdict(int)}
    total_samples = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            batch_size = labels.size(0)
            total_samples += batch_size
            outputs, exit_points = model(images)
            _, predicted = torch.max(outputs.data,1)
            for exit_idx in [1,3,4]:
                mask = (exit_points == exit_idx)
                count = mask.sum().item()
                exit_counts[exit_idx] += count
                if count > 0:
                    exit_labels = labels[mask]
                    for label in exit_labels:
                        class_distributions[exit_idx][label.item()] += 1
    exit_distribution = {k:(v/total_samples)*100 for k,v in exit_counts.items()}
    return exit_distribution, class_distributions

def plot_class_distribution(class_distributions, dataset_name):
    output_dir = create_output_directory(dataset_name)
    class_names = {0:'airplane',1:'automobile',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}
    fig, axes = plt.subplots(2,2, figsize=(20,10), sharey=True)
    fig.suptitle(f'{dataset_name.upper()} - Class Distributions Across Exits (Percentage)', fontsize=16)
    exit_indices = sorted(class_distributions.keys())
    exit_names = ['Exit 1 (Early)', 'Exit 3', 'Final (Exit 4)']
    for idx, exit_idx in enumerate(exit_indices):
        row, col = (0, idx) if idx < 2 else (1, idx-2)
        ax = axes[row, col]
        distribution = class_distributions[exit_idx]
        classes = list(distribution.keys())
        counts = list(distribution.values())
        total = sum(sum(d.values()) for d in class_distributions.values())
        percentages = [(count/total)*100 for count in counts]
        ax.bar([class_names[i] for i in classes], percentages, color='#3498db')
        ax.set_title(f'{exit_names[idx]} Class Distribution', fontsize=14)
        ax.set_xlabel('Class Label', fontsize=12)
        if col == 0:
            ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_xticklabels([class_names[i] for i in classes], rotation=45, ha='right')
        for i, p in enumerate(percentages):
            ax.text(i, p+0.5, f'{p:.2f}%', ha='center', va='bottom', fontsize=9)
    if len(exit_indices) < 4:
        for idx in range(len(exit_indices), 4):
            row = idx // 2
            col = idx % 2
            axes[row, col].axis('off')
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(os.path.join(output_dir, f'{dataset_name.lower()}_class_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_time_comparison(static_time, branchy_time, dataset_name):
    output_dir = create_output_directory(dataset_name)
    methods = ['Static ResNet','Branchy ResNet']
    times = [static_time, branchy_time]
    plt.figure(figsize=(10,6))
    bars = plt.bar(methods, times, color=['#2ecc71','#3498db'], width=0.6)
    plt.title(f'{dataset_name.upper()} - Training Time Comparison', fontsize=14)
    plt.ylabel('Training Time (s)', fontsize=12)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x()+bar.get_width()/2., h, f'{h:.2f}s', ha='center', va='bottom')
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
    plt.figure(figsize=(12,10))
    class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
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

# -------------------------------------------------------
# Experiment Runner
# -------------------------------------------------------
def run_experiments():
    dataset_name = 'cifar10'
    print(f"\nRunning experiments on {dataset_name.upper()}...")
    train_loader, test_loader = load_datasets(batch_size=128)
    weights_dir = 'pretrained_weights'
    os.makedirs(weights_dir, exist_ok=True)
    static_weights_path = os.path.join(weights_dir, 'static_resnet18_cifar10.pth')
    branchy_weights_path = os.path.join(weights_dir, 'branchy_resnet18_cifar10.pth')

    print("\nInitializing Static ResNet18...")
    static_resnet = StaticResNet18(num_classes=10, in_channels=3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    static_resnet = static_resnet.to(device)
    if os.path.exists(static_weights_path):
        print("Loading pre-trained Static ResNet18 weights...")
        checkpoint = torch.load(static_weights_path)
        static_resnet.load_state_dict(checkpoint['state_dict'])
        static_training_time = 0.0
    else:
        print("Training Static ResNet18 from scratch...")
        static_resnet, static_training_time = train_static_resnet(
            static_resnet, train_loader, test_loader,
            num_epochs=100,
            learning_rate=0.001,
            weights_path=static_weights_path
        )

    print("\nEvaluating Static ResNet18...")
    static_accuracy, static_inference_time = evaluate_static_resnet(static_resnet, test_loader)
    print(f"Static ResNet18 Results:")
    print(f"Accuracy: {static_accuracy:.2f}%")
    print(f"Average Inference Time: {static_inference_time:.2f} ms")

    print("\nMeasuring power consumption for Static ResNet18...")
    static_power = measure_power_consumption(static_resnet, test_loader, num_samples=100)

    print("\nInitializing Branchy ResNet18...")
    branchy_resnet = BranchyResNet18(num_classes=10, in_channels=3)
    branchy_resnet = branchy_resnet.to(device)
    if os.path.exists(branchy_weights_path):
        print("Loading pre-trained Branchy ResNet18 weights...")
        checkpoint = torch.load(branchy_weights_path)
        branchy_resnet.load_state_dict(checkpoint['state_dict'])
        branchy_training_time = 0.0
        all_exit_distributions = []
        avg_exit_distribution = {}
    else:
        print("Training Branchy ResNet18 from scratch...")
        branchy_resnet, branchy_training_time, all_exit_distributions, avg_exit_distribution = train_branchy_resnet(
            branchy_resnet, train_loader, test_loader,
            num_epochs=100,
            learning_rate=0.001
        )
        torch.save({
            'state_dict': branchy_resnet.state_dict(),
            'accuracy': evaluate_branchy_resnet(branchy_resnet, test_loader)[0]
        }, branchy_weights_path)

    print("\nEvaluating Branchy ResNet18...")
    final_accuracy, final_inference_time, exit_percentages = evaluate_branchy_resnet(branchy_resnet, test_loader)
    print(f"Branchy ResNet18 Results:")
    print(f"Accuracy: {final_accuracy:.2f}%")
    print(f"Average Inference Time: {final_inference_time:.2f} ms")
    print(f"Exit Distribution: {exit_percentages}")

    print("\nMeasuring power consumption for Branchy ResNet18...")
    branchy_power = measure_power_consumption(branchy_resnet, test_loader, num_samples=100)

    speed_improvement = ((static_inference_time - final_inference_time) / static_inference_time) * 100 if static_inference_time > 0 else 0
    accuracy_difference = final_accuracy - static_accuracy
    energy_savings = ((static_power['energy'] - branchy_power['energy']) / static_power['energy']) * 100 if static_power['energy'] > 0 else 0

    results = {
        'static': {
            'accuracy': static_accuracy,
            'inference_time': static_inference_time,
            'power': static_power,
            'training_time': static_training_time
        },
        'branchy': {
            'accuracy': final_accuracy,
            'inference_time': final_inference_time,
            'exit_percentages': exit_percentages,
            'power': branchy_power,
            'training_time': branchy_training_time,
            'all_exit_distributions': all_exit_distributions,
            'avg_exit_distribution': avg_exit_distribution
        },
        'improvements': {
            'speed': speed_improvement,
            'accuracy': accuracy_difference,
            'energy_savings': energy_savings
        }
    }

    print("\nResults Summary:")
    print(f"Static ResNet18 - Accuracy: {static_accuracy:.2f}%, Time: {static_inference_time:.2f}ms, Energy: {static_power['energy']:.2f}J, Training Time: {static_training_time:.2f}s")
    print(f"Branchy ResNet18 - Accuracy: {final_accuracy:.2f}%, Time: {final_inference_time:.2f}ms, Energy: {branchy_power['energy']:.2f}J, Training Time: {branchy_training_time:.2f}s")
    print(f"Speed Improvement: {speed_improvement:.1f}%")
    print(f"Accuracy Difference: {accuracy_difference:+.2f}%")
    print(f"Energy Savings: {energy_savings:.1f}%")
    print(f"Final Exit Distribution: {exit_percentages}")
    print(f"Average Exit Distribution Across Epochs: {avg_exit_distribution}")

    print("\nGenerating comparative plots...")
    static_results = {'accuracy': static_accuracy, 'inference_time': static_inference_time, 'power': static_power}
    branchy_results = {'accuracy': final_accuracy, 'inference_time': final_inference_time, 'power': branchy_power, 'exit_percentages': avg_exit_distribution}
    plot_comparative_analysis(static_results, branchy_results, dataset_name)
    plot_exit_distribution(avg_exit_distribution, dataset_name)
    _, class_distributions = analyze_exit_distribution(branchy_resnet, test_loader, dataset_name)
    plot_class_distribution(class_distributions, dataset_name)
    plot_training_time_comparison(static_training_time, branchy_training_time, dataset_name)
    plot_confusion_matrix(static_resnet, test_loader, is_branchy=False, dataset_name=dataset_name)
    plot_confusion_matrix(branchy_resnet, test_loader, is_branchy=True, dataset_name=dataset_name)

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
