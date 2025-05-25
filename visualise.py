import pandas as pd
from utils import limit_data
import os
import argparse
import matplotlib.pyplot as plt
import torch
from PIL import Image
import numpy as np
from model import BirdDroneResNet50

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


def show_sample_images(df_train, valid_categories):
    fig, ax = plt.subplots(len(valid_categories), 5, figsize=(25, 10))
    for i, idx in enumerate(valid_categories):
        path_cat = df_train[df_train['label'] == idx].path.values[:5]
        for j, path in enumerate(path_cat):
            img = Image.open(path)
            ax[i, j].imshow(img)
            ax[i, j].set(xlabel=idx)


def show_sample_images_from_df(df_train, valid_categories):
    fig, ax = plt.subplots(len(valid_categories), 5, figsize=(25, 10))
    for i, idx in enumerate(valid_categories):
        path_cat = df_train[df_train['label'] == idx].path.values[:5]
        for j, path in enumerate(path_cat):
            img = Image.open(path)
            ax[i, j].imshow(img)
            ax[i, j].set(xlabel=idx)

def get_image_resolutions(folder_path, max_images=1000):
    resolutions = []
    image_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ][:max_images]

    for img_file in image_files:
        try:
            img_path = os.path.join(folder_path, img_file)
            with Image.open(img_path) as img:
                resolutions.append(img.size)  # (width, height)
        except Exception as e:
            print(f"[!] Failed to read {img_file}: {e}")
    return resolutions

def plot_resolution_distribution(resolutions, title="Resolution Distribution"):
    if not resolutions:
        print("[!] No resolutions to plot.")
        return

    widths, heights = zip(*resolutions)
    plt.figure(figsize=(8, 6))
    plt.scatter(widths, heights, alpha=0.6, edgecolor='k')
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def analyze_rgb_components(image_paths):
    rgb_means = []
    rgb_stds = []
    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                arr = np.array(img) / 255.0
                rgb_means.append(arr.mean(axis=(0, 1)))
                rgb_stds.append(arr.std(axis=(0, 1)))
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
    if rgb_means:
        mean_rgb = np.mean(rgb_means, axis=0)
        std_rgb = np.mean(rgb_stds, axis=0)
        print(f"Mean RGB: {mean_rgb}")
        print(f"Std RGB: {std_rgb}")
        return mean_rgb, std_rgb
    else:
        print("No images processed.")
        return None, None

def plot_rgb_components(mean_rgb, std_rgb, label, color_palette=None):
    if color_palette is None:
        color_palette = ['#FF595E', '#1982C4', '#8AC926']  # R, G, B

    components = ['Red', 'Green', 'Blue']
    x = np.arange(len(components))

    plt.figure(figsize=(6, 5))
    bars = plt.bar(x, mean_rgb, yerr=std_rgb, capsize=10, color=color_palette, alpha=0.85)
    plt.xticks(x, components, fontsize=12)
    plt.ylabel('Normalized Mean Value', fontsize=13)
    plt.title(f'RGB Components for {label}', fontsize=15, weight='bold')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{mean_rgb[i]:.2f}", 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

from torchsummary import summary

def print_resnet_tree(model, indent=0):
    for name, module in model.named_children():
        print(' ' * indent + f"{name}: {module.__class__.__name__}")
        # Only recurse if the module has children
        if any(module.named_children()):
            print_resnet_tree(module, indent + 2)



def main():
    # Example usage
    parser = argparse.ArgumentParser(description="Visualise Bird/Drone Dataset")
    parser.add_argument('--train_path', type=str, default='Train', help='Path to the training data')
    parser.add_argument('--val_path', type=str, default='Val', help='Path to the validation data')
    parser.add_argument('--test_path', type=str, default='Test', help='Path to the test data')
    args = parser.parse_args()

    train_path = args.train_path
    val_path = args.val_path
    test_path = args.test_path

    IMAGE_SAMPLES=None      # Pass image samples as None to take all samples of images
    categories=os.listdir(train_path)          # list of all categories

    #df_train=pd.DataFrame(columns=['path','label'])
    #df_val=pd.DataFrame(columns=['path','label'])
    #df_test=pd.DataFrame(columns=['path','label'])

    df_train=limit_data(train_path,categories,IMAGE_SAMPLES)
    df_val=limit_data(val_path,categories,IMAGE_SAMPLES)
    df_test=limit_data(test_path,categories,IMAGE_SAMPLES)
    df_train=df_train.sample(frac=1, ignore_index=True)
    df_val=df_val.sample(frac=1, ignore_index=True)
    df_test=df_test.sample(frac=1, ignore_index=True)

    valid_categories = [cat for cat in categories if cat in df_train['label'].unique()]
    if not valid_categories:
        raise ValueError("No valid categories found in the training dataset. Please check your data.")
    show_sample_images_from_df(df_train, valid_categories)

    # Assuming you have a DataLoader named test_loader
    # evaluate(model, test_loader)
    print("Resolution Distribution for Validation Data:")
    bird_res = get_image_resolutions("val/Bird")
    plot_resolution_distribution(bird_res, title="Bird Image Resolutions - Validation")

    drone_res = get_image_resolutions("val/Drone")
    plot_resolution_distribution(drone_res, title="Drone Image Resolutions - Validation")

    print("Analyzing RGB Components for Validation Data:")
    analyze_rgb_components(df_val['path'].values)
    bird_folder = "test/Bird"
    bird_image_files = [
        os.path.join(bird_folder, f)
        for f in os.listdir(bird_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    bird_mean_rgb, bird_std_rgb = analyze_rgb_components(bird_image_files)

    # Plot for birds
    drone_folder = "test/Drone"
    drone_image_files = [
        os.path.join(drone_folder, f)
        for f in os.listdir(drone_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    drone_mean_rgb, drone_std_rgb = analyze_rgb_components(drone_image_files)

    plot_rgb_components(bird_mean_rgb, bird_std_rgb, label='Bird Images')

    # Plot for drones
    plot_rgb_components(drone_mean_rgb, drone_std_rgb, label='Drone Images')

    resnet_model = BirdDroneResNet50().to(device)
    print("BirdDroneResNet50 architecture tree:")
    resnet_model = BirdDroneResNet50()
    print_resnet_tree(resnet_model)

if __name__ == "__main__":
    main()