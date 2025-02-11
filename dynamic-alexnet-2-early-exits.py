import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt
import numpy as np
import random
import os

from collections import defaultdict
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10

import pynvml
import pandas as pd
import threading
import queue

from torch.amp import GradScaler, autocast

# ---------------------------
# Model Definitions
# ---------------------------
class QLearningAgent:
    def __init__(self, n_exits, epsilon=0.1, alpha=0.1, gamma=0.9):
        """Initialize the Q-Learning agent."""
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

class EarlyExitBlock(nn.Module):
    def __init__(self, in_channels, num_classes):
        """Early exit block with a shared conv layer and classifier."""
        super(EarlyExitBlock, self).__init__()
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((in_channels // 2) * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.shared_conv(x)
        x = self.classifier(x)
        return x

class StaticAlexNet(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        """Standard AlexNet adapted for 32x32 inputs."""
        super(StaticAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.output_layer = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class BranchyAlexNet(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        """
        Branchy AlexNet with 2 early exits (after the first conv block and after the middle layers).
        The final output layer acts as Exit 3.
        """
        super(BranchyAlexNet, self).__init__()
        self.num_classes = num_classes
        self.training_mode = True
        self.exit_loss_weights = [0.15, 0.15, 0.70]
        self.rl_agent = QLearningAgent(n_exits=2)

        # Early Exit 1: after first conv block
        self.features1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.exit1 = EarlyExitBlock(64, num_classes)

        # Combined middle layers (acts as Exit 2)
        self.features2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.exit2 = EarlyExitBlock(256, num_classes)

        # Final layers (Exit 3)
        self.final_features = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.output_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
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
        x_final = self.final_features(x2)
        x_final = self.adaptive_pool(x_final)
        x_final = x_final.view(x_final.size(0), -1)
        outputs.append(self.output_layer(x_final))
        return outputs

    def _forward_inference(self, x):
        device = x.device
        batch_size = x.size(0)
        final_outputs = torch.zeros(batch_size, self.num_classes, device=device)
        exit_points = torch.zeros(batch_size, dtype=torch.int, device=device)
        remaining_indices = torch.arange(batch_size, device=device)
        x_current = x
        feature_blocks = [self.features1, self.features2]
        exit_blocks = [self.exit1, self.exit2]
        for exit_idx, (features, exit_block) in enumerate(zip(feature_blocks, exit_blocks)):
            if len(remaining_indices) > 0:
                x_current = features(x_current)
                exit_output = exit_block(x_current)
                softmax_output = torch.softmax(exit_output, dim=1)
                confidence, _ = torch.max(softmax_output, dim=1)
                exit_decisions = [self.rl_agent.select_action(self.rl_agent.get_state(exit_idx, conf.item()), training=False) == 0 for conf in confidence]
                exit_mask = torch.tensor(exit_decisions, dtype=torch.bool, device=device)
                exit_inds = remaining_indices[exit_mask]
                if len(exit_inds) > 0:
                    final_outputs[exit_inds] = exit_output[exit_mask]
                    exit_points[exit_inds] = exit_idx + 1
                remaining_indices = remaining_indices[~exit_mask]
                x_current = x_current[~exit_mask]
            else:
                break
        if len(remaining_indices) > 0:
            x_final = self.final_features(x_current)
            x_final = self.adaptive_pool(x_final)
            x_final = x_final.view(x_final.size(0), -1)
            final_output = self.output_layer(x_final)
            final_outputs[remaining_indices] = final_output
            exit_points[remaining_indices] = 3
        return final_outputs, exit_points

    def _calculate_reward(self, exit_idx, correct):
        base_reward = 1.0 if correct else -1.0
        early_exit_bonus = max(0, 2 - exit_idx) * 0.2
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
        feature_blocks = [self.features1, self.features2]
        exit_blocks = [self.exit1, self.exit2]
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
                        next_state = self.rl_agent.get_state(exit_idx + 1, conf.item())
                        self.rl_agent.update(state, action, reward, next_state)
        return total_loss

# ---------------------------
# Data Loading
# ---------------------------
def load_datasets(dataset_name='cifar10', batch_size=32):
    if dataset_name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
        train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
    else:
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

# ---------------------------
# Training Functions
# ---------------------------
def train_static_alexnet(model, train_loader, test_loader=None, num_epochs=100, learning_rate=0.001, weights_path=None):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    scaler = GradScaler() if device.type == 'cuda' else None
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            if device.type == 'cuda':
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        scheduler.step(avg_loss)
        if test_loader is not None:
            accuracy = evaluate_static_alexnet(model, test_loader)[0]
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        else:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    return model

def train_branchy_alexnet(model, train_loader, test_loader, num_epochs=100, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    scaler = GradScaler() if device.type == 'cuda' else None
    best_accuracy = 0
    best_model_state = None
    for epoch in range(num_epochs):
        model.train()
        model.training_mode = True
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            if device.type == 'cuda':
                with autocast(device_type='cuda', dtype=torch.float16):
                    loss = model.train_step(images, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = model.train_step(images, labels)
                loss.backward()
                optimizer.step()
            running_loss += loss.item()
            if (batch_idx + 1) % 500 == 0:
                print(f'\nEpoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}]')
                print(f'Current Loss: {loss.item():.4f}')
                model.eval()
                model.training_mode = False
                accuracy, inference_time, exit_percentages = evaluate_branchy_alexnet(model, test_loader)
                print(f'Current Accuracy: {accuracy:.2f}%')
                print(f'Inference Time: {inference_time:.2f} ms')
                print(f'Exit Distribution: {exit_percentages}')
                model.train()
                model.training_mode = True
        avg_loss = running_loss / len(train_loader)
        scheduler.step(avg_loss)
        model.eval()
        model.training_mode = False
        accuracy, inference_time, exit_percentages = evaluate_branchy_alexnet(model, test_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, Inference Time: {inference_time:.2f} ms')
        print(f'Exit Distribution: {exit_percentages}')
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state = model.state_dict()
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model

# ---------------------------
# Evaluation Functions
# ---------------------------
def evaluate_static_alexnet(model, test_loader):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
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
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    avg_inference_time = (sum(inference_times) / len(inference_times)) * 1000
    return accuracy, avg_inference_time

def evaluate_branchy_alexnet(model, test_loader):
    model.eval()
    model.training_mode = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    correct = 0
    total = 0
    inference_times = []
    exit_counts = {1: 0, 2: 0, 3: 0}  # Exit 1, Exit 2, and final output (Exit 3)
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
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for exit_idx in exit_counts.keys():
                count = (exit_points == exit_idx).sum().item()
                exit_counts[exit_idx] += count
    accuracy = 100 * correct / total
    avg_inference_time = (sum(inference_times) / total) * 1000
    exit_percentages = {k: (v / total) * 100 for k, v in exit_counts.items()}
    return accuracy, avg_inference_time, exit_percentages

# ---------------------------
# Power Monitoring
# ---------------------------
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
                    print(f"NVML Error during monitoring: {error}")
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
            if hasattr(model, 'training_mode'):
                model.training_mode = False
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

# ---------------------------
# Visualization & Analysis Functions
# ---------------------------
def create_output_directory(dataset_name):
    output_dir = f'plots_{dataset_name.lower()}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def plot_confusion_matrix(y_true, y_pred, class_names, title, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
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
    filtered_exits = {k: v for k, v in exit_percentages.items() if k in [1,2,3]}
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

def plot_class_distribution(class_distributions, dataset_name):
    if dataset_name.lower() == 'mnist':
        class_names = {i: str(i) for i in range(10)}
    else:
        class_names = {
            0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
            5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
        }
    output_dir = create_output_directory(dataset_name)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    fig.suptitle(f'{dataset_name.upper()} - Class Distributions Across Exits (Percentage)', fontsize=16)
    exit_indices = sorted(class_distributions.keys())
    for idx, exit_idx in enumerate(exit_indices):
        ax = axes[idx]
        distribution = class_distributions[exit_idx]
        classes = list(distribution.keys())
        total = sum(sum(d.values()) for d in class_distributions.values())
        percentages = [(distribution[i] / total) * 100 for i in classes]
        ax.bar([class_names[i] for i in classes], percentages, color='#3498db')
        ax.set_title(f'Exit {exit_idx} Class Distribution', fontsize=14)
        ax.set_xlabel('Class Label', fontsize=12)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_xticklabels([class_names[i] for i in classes], rotation=45, ha='right')
        for i, p in enumerate(percentages):
            ax.text(i, p+0.5, f'{p:.2f}%', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name.lower()}_class_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_exit_distribution(model, test_loader, dataset_name='mnist'):
    device = next(model.parameters()).device
    model.eval()
    model.training_mode = False
    exit_counts = {1: 0, 2: 0, 3: 0}  # For 2 early exits + final
    class_distributions = {1: defaultdict(int), 2: defaultdict(int), 3: defaultdict(int)}
    total_samples = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            batch_size = labels.size(0)
            total_samples += batch_size
            outputs, exit_points = model(images)
            _, predictions = torch.max(outputs, dim=1)
            for exit_idx in range(1, 4):
                mask = (exit_points == exit_idx)
                exit_counts[exit_idx] += mask.sum().item()
                for label in labels[mask]:
                    class_distributions[exit_idx][label.item()] += 1
    return exit_counts, class_distributions, total_samples

def run_experiments(dataset_name):
    print(f"\nRunning experiments on {dataset_name.upper()}...")
    train_loader, test_loader = load_datasets(dataset_name, batch_size=32)
    in_channels = 3
    weights_dir = 'pretrained_weights'
    os.makedirs(weights_dir, exist_ok=True)
    static_weights_path = os.path.join(weights_dir, f'static_alexnet_{dataset_name.lower()}.pth')
    branchy_weights_path = os.path.join(weights_dir, f'branchy_alexnet_{dataset_name.lower()}.pth')

    print(f"\nInitializing Static AlexNet for {dataset_name.upper()}...")
    static_alexnet = StaticAlexNet(num_classes=10, in_channels=in_channels)
    static_alexnet = static_alexnet.to(device)
    if os.path.exists(static_weights_path):
        print("Loading pre-trained Static AlexNet weights...")
        checkpoint = torch.load(static_weights_path)
        static_alexnet.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded weights with validation accuracy: {checkpoint['accuracy']:.2f}%")
    else:
        print("Training Static AlexNet from scratch...")
        static_alexnet = train_static_alexnet(static_alexnet, train_loader, test_loader, num_epochs=100, learning_rate=0.001)
        torch.save({
            'state_dict': static_alexnet.state_dict(),
            'accuracy': evaluate_static_alexnet(static_alexnet, test_loader)[0]
        }, static_weights_path)

    print(f"\nEvaluating Static AlexNet on {dataset_name.upper()}...")
    static_accuracy, static_inference_time = evaluate_static_alexnet(static_alexnet, test_loader)
    print(f"Static AlexNet Results:")
    print(f"Accuracy: {static_accuracy:.2f}%")
    print(f"Average Inference Time: {static_inference_time:.2f} ms")

    print(f"\nMeasuring power consumption for Static AlexNet...")
    static_power = measure_power_consumption(static_alexnet, test_loader, num_samples=100)
    print(f"Static AlexNet Power Consumption: {static_power}")

    print(f"\nInitializing Branchy AlexNet for {dataset_name.upper()}...")
    branchy_alexnet = BranchyAlexNet(num_classes=10, in_channels=in_channels)
    branchy_alexnet = branchy_alexnet.to(device)
    if os.path.exists(branchy_weights_path):
        print("Loading pre-trained Branchy AlexNet weights...")
        checkpoint = torch.load(branchy_weights_path)
        branchy_alexnet.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded weights with validation accuracy: {checkpoint['accuracy']:.2f}%")
    else:
        print("Training Branchy AlexNet from scratch...")
        branchy_alexnet = train_branchy_alexnet(branchy_alexnet, train_loader, test_loader, num_epochs=100, learning_rate=0.001)
        torch.save({
            'state_dict': branchy_alexnet.state_dict(),
            'accuracy': evaluate_branchy_alexnet(branchy_alexnet, test_loader)[0]
        }, branchy_weights_path)

    print("\nEvaluating Branchy AlexNet...")
    final_accuracy, final_inference_time, exit_percentages = evaluate_branchy_alexnet(branchy_alexnet, test_loader)
    print(f"Branchy AlexNet Results:")
    print(f"Accuracy: {final_accuracy:.2f}%")
    print(f"Average Inference Time: {final_inference_time:.2f} ms")
    print(f"Exit Distribution: {exit_percentages}")

    print(f"\nMeasuring power consumption for Branchy AlexNet...")
    branchy_power = measure_power_consumption(branchy_alexnet, test_loader, num_samples=100)

    speed_improvement = ((static_inference_time - final_inference_time) / static_inference_time) * 100
    accuracy_difference = final_accuracy - static_accuracy
    energy_savings = ((static_power['energy'] - branchy_power['energy']) / static_power['energy'] * 100)

    results = {
        'static': {
            'accuracy': static_accuracy,
            'inference_time': static_inference_time,
            'power': static_power
        },
        'branchy': {
            'accuracy': final_accuracy,
            'inference_time': final_inference_time,
            'exit_percentages': exit_percentages,
            'power': branchy_power
        },
        'improvements': {
            'speed': speed_improvement,
            'accuracy': accuracy_difference,
            'energy_savings': energy_savings
        }
    }

    print("\nResults Summary:")
    print(f"Static AlexNet - Accuracy: {static_accuracy:.2f}%, Inference Time: {static_inference_time:.2f}ms, Energy: {static_power['energy']:.2f}J")
    print(f"Branchy AlexNet - Accuracy: {final_accuracy:.2f}%, Inference Time: {final_inference_time:.2f}ms, Energy: {branchy_power['energy']:.2f}J")
    print(f"Speed Improvement: {speed_improvement:.1f}%")
    print(f"Accuracy Difference: {accuracy_difference:+.2f}%")
    print(f"Energy Savings: {energy_savings:.1f}%")
    print(f"Exit Distribution: {exit_percentages}")

    print(f"\nGenerating comparative plots for {dataset_name.upper()}...")
    static_results = {'accuracy': static_accuracy, 'inference_time': static_inference_time, 'power': static_power}
    branchy_results = {'accuracy': final_accuracy, 'inference_time': final_inference_time, 'power': branchy_power, 'exit_percentages': exit_percentages}
    plot_comparative_analysis(static_results, branchy_results, dataset_name)
    plot_exit_distribution(exit_percentages, dataset_name)
    _, class_distributions, _ = analyze_exit_distribution(branchy_alexnet, test_loader, dataset_name)
    plot_class_distribution(class_distributions, dataset_name)
    return results

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    mnist_results = run_experiments('mnist')
    cifar_results = run_experiments('cifar10')

    print("\nAll experiments completed.")
