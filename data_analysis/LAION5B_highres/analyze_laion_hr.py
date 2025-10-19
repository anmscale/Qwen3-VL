#!/usr/bin/env python3
"""
Analyze LAION-High-Resolution dataset metadata to understand image resolution distribution.
"""

import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

def analyze_resolution_distribution(sample_size=10000):
    """Download and analyze a sample of LAION-High-Resolution metadata."""

    print(f"Loading {sample_size} samples from LAION-High-Resolution dataset...")
    print("Note: This requires accepting terms on Hugging Face first.")
    print("If you get an authentication error, run: huggingface-cli login")
    print()

    try:
        # Load dataset with streaming to avoid downloading everything
        dataset = load_dataset(
            "laion/laion-high-resolution",
            split="train",
            streaming=True
        )

        # Collect sample data
        samples = []
        widths = []
        heights = []
        resolutions = []
        aspect_ratios = []

        print("Collecting samples...")
        for i, sample in enumerate(dataset):
            if i >= sample_size:
                break

            if i % 1000 == 0:
                print(f"  Processed {i}/{sample_size} samples...")

            width = sample.get('WIDTH', 0)
            height = sample.get('HEIGHT', 0)

            if width > 0 and height > 0:
                widths.append(width)
                heights.append(height)
                resolutions.append(width * height)
                aspect_ratios.append(width / height)

                samples.append({
                    'width': width,
                    'height': height,
                    'resolution': width * height,
                    'aspect_ratio': width / height,
                    'url': sample.get('URL', ''),
                    'text': sample.get('TEXT', '')[:100] if sample.get('TEXT') else ''  # First 100 chars
                })

        print(f"\nCollected {len(samples)} valid samples")

        # Create DataFrame
        df = pd.DataFrame(samples)

        # Analysis
        print("\n" + "="*80)
        print("LAION-HIGH-RESOLUTION DATASET ANALYSIS")
        print("="*80)

        print(f"\nSample size: {len(df):,}")

        print("\n--- Resolution Statistics ---")
        print(f"Width  - Min: {df['width'].min():,}px, Max: {df['width'].max():,}px, Mean: {df['width'].mean():.0f}px, Median: {df['width'].median():.0f}px")
        print(f"Height - Min: {df['height'].min():,}px, Max: {df['height'].max():.0f}px, Mean: {df['height'].mean():.0f}px, Median: {df['height'].median():.0f}px")
        print(f"Total Pixels - Mean: {df['resolution'].mean()/1e6:.2f}MP, Median: {df['resolution'].median()/1e6:.2f}MP")

        print("\n--- Resolution Percentiles ---")
        percentiles = [25, 50, 75, 90, 95, 99]
        for p in percentiles:
            w = np.percentile(df['width'], p)
            h = np.percentile(df['height'], p)
            print(f"{p}th percentile: {w:.0f} x {h:.0f}")

        print("\n--- Common Resolution Buckets ---")
        def bucket_resolution(row):
            w, h = row['width'], row['height']
            if w >= 1024 and h >= 1024 and w < 2048 and h < 2048:
                return "1024-2048"
            elif w >= 2048 and h >= 2048 and w < 4096 and h < 4096:
                return "2048-4096"
            elif w >= 4096 or h >= 4096:
                return "4096+"
            else:
                return "< 1024 (shouldn't happen)"

        df['bucket'] = df.apply(bucket_resolution, axis=1)
        bucket_counts = df['bucket'].value_counts()
        print(bucket_counts.to_string())
        print(f"\nPercentage >= 2048: {(bucket_counts.get('2048-4096', 0) + bucket_counts.get('4096+', 0)) / len(df) * 100:.1f}%")

        print("\n--- Aspect Ratio Statistics ---")
        print(f"Mean: {df['aspect_ratio'].mean():.2f}, Median: {df['aspect_ratio'].median():.2f}")
        print(f"Landscape (>1.1): {(df['aspect_ratio'] > 1.1).sum() / len(df) * 100:.1f}%")
        print(f"Portrait (<0.9): {(df['aspect_ratio'] < 0.9).sum() / len(df) * 100:.1f}%")
        print(f"Square (0.9-1.1): {((df['aspect_ratio'] >= 0.9) & (df['aspect_ratio'] <= 1.1)).sum() / len(df) * 100:.1f}%")

        print("\n--- Sample Text Captions ---")
        for i, row in df.head(5).iterrows():
            print(f"\n{i+1}. [{row['width']}x{row['height']}] {row['text']}")

        # Save data
        output_file = "laion_hr_sample_analysis.csv"
        df.to_csv(output_file, index=False)
        print(f"\n--- Output ---")
        print(f"Saved detailed data to: {output_file}")

        # Create visualizations
        create_visualizations(df)

        return df

    except Exception as e:
        print(f"Error: {e}")
        print("\nIf you get an authentication error, you need to:")
        print("1. Create a Hugging Face account")
        print("2. Accept the dataset terms at: https://huggingface.co/datasets/laion/laion-high-resolution")
        print("3. Run: huggingface-cli login")
        return None


def create_visualizations(df):
    """Create visualization plots."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Width vs Height scatter
    axes[0, 0].scatter(df['width'], df['height'], alpha=0.1, s=1)
    axes[0, 0].set_xlabel('Width (px)')
    axes[0, 0].set_ylabel('Height (px)')
    axes[0, 0].set_title('Image Dimensions Distribution')
    axes[0, 0].axhline(y=1024, color='r', linestyle='--', alpha=0.5, label='1024px')
    axes[0, 0].axvline(x=1024, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlim(0, df['width'].quantile(0.99))
    axes[0, 0].set_ylim(0, df['height'].quantile(0.99))
    axes[0, 0].legend()

    # 2. Resolution histogram
    axes[0, 1].hist((df['resolution'] / 1e6).values, bins=50, edgecolor='black')
    axes[0, 1].set_xlabel('Resolution (Megapixels)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Resolution Distribution')
    axes[0, 1].axvline(x=1, color='r', linestyle='--', alpha=0.5, label='1MP (1024x1024)')
    axes[0, 1].legend()

    # 3. Aspect ratio distribution
    # Clip extreme aspect ratios for better visualization
    ar_clipped = df['aspect_ratio'].clip(0.2, 5).values
    axes[1, 0].hist(ar_clipped, bins=50, edgecolor='black')
    axes[1, 0].set_xlabel('Aspect Ratio (width/height)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Aspect Ratio Distribution')
    axes[1, 0].axvline(x=1, color='r', linestyle='--', alpha=0.5, label='Square')
    axes[1, 0].legend()

    # 4. Resolution buckets
    bucket_counts = df['bucket'].value_counts().sort_index()
    axes[1, 1].bar(range(len(bucket_counts)), bucket_counts.values)
    axes[1, 1].set_xticks(range(len(bucket_counts)))
    axes[1, 1].set_xticklabels(bucket_counts.index, rotation=45)
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Resolution Buckets')

    plt.tight_layout()
    output_plot = "laion_hr_analysis.png"
    plt.savefig(output_plot, dpi=150)
    print(f"Saved visualization to: {output_plot}")


if __name__ == "__main__":
    import sys

    sample_size = 10000
    if len(sys.argv) > 1:
        sample_size = int(sys.argv[1])

    print("LAION-High-Resolution Dataset Analyzer")
    print(f"Sample size: {sample_size}")
    print()

    df = analyze_resolution_distribution(sample_size=sample_size)
