import numpy as np
import pandas as pd
import sklearn
from PIL import Image
import torch
import os
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler

#import numpy as np
import pandas as pd
import sklearn
from PIL import Image
import torch
import os

# Convert laptops dataset to DataFrame
def create_laptops_dataframe(csv_path, images_path):
    """Convert laptops dataset to pandas DataFrame with correct image paths"""
    
    # Read the CSV file
    df = pd.read_csv(csv_path, index_col=0)
    
    # Create the correct image paths by combining class and image name
    df['image_path'] = df['class'] + '_' + df['img']
    
    # Create full path to images
    df['full_image_path'] = df['image_path'].apply(lambda x: os.path.join(images_path, x))
    
    # Add type column
    df['type'] = 'laptop'
    
    # Keep only the columns we need
    df = df[['img', 'class', 'image_path', 'full_image_path', 'type']]
    
    # Verify that image files actually exist
    df['exists'] = df['full_image_path'].apply(os.path.exists)
    
    print(f"Total laptop records: {len(df)}")
    print(f"Images found: {df['exists'].sum()}")
    print(f"Images missing: {(~df['exists']).sum()}")
    
    return df

# Create laptops DataFrame
laptops_csv_path = r'C:\lost-and-found\laptops\laptops.csv'
laptops_images_path = r'C:\lost-and-found\laptops\all_images'
laptops = create_laptops_dataframe(laptops_csv_path, laptops_images_path)

# Convert shoes dataset to DataFrame
def create_shoes_dataframe(base_path):
    """Convert shoes image dataset to pandas DataFrame with uniform 'shoes' label"""
    data = []
    
    # Process both train and test folders
    for split in ['train', 'test']:
        split_path = os.path.join(base_path, split)
        if os.path.exists(split_path):
            # Process each brand folder (but don't use brand as label)
            for brand_folder in os.listdir(split_path):
                brand_path = os.path.join(split_path, brand_folder)
                if os.path.isdir(brand_path):
                    # Process each image file
                    for image_file in os.listdir(brand_path):
                        if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            image_path = os.path.join(brand_path, image_file)
                            data.append({
                                'image_path': image_path,
                                'type': 'shoes'
                            })
    
    return pd.DataFrame(data)

# Create shoes DataFrame
shoes_base_path = r'C:\lost-and-found\shoes'
shoes = create_shoes_dataframe(shoes_base_path)

# Convert wallets dataset to DataFrame
def create_wallets_dataframe(base_path):
    """Convert wallets image dataset to pandas DataFrame"""
    data = []
    
    # Navigate to the data folder
    data_path = os.path.join(base_path, 'images.cv_fxnemk1hbz4rjms5la9wn', 'data')
    
    if os.path.exists(data_path):
        # Process train, test, and val folders
        for split in ['train', 'test', 'val']:
            split_path = os.path.join(data_path, split)
            if os.path.exists(split_path):
                # Process wallet folder
                wallet_path = os.path.join(split_path, 'wallet')
                if os.path.isdir(wallet_path):
                    # Process each image file
                    for image_file in os.listdir(wallet_path):
                        if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            image_path = os.path.join(wallet_path, image_file)
                            data.append({
                                'image_path': image_path,
                                'type': 'wallet'
                            })
    
    return pd.DataFrame(data)

# Create wallets DataFrame
wallets_base_path = r'C:\lost-and-found\wallets'
wallets = create_wallets_dataframe(wallets_base_path)

# Convert earbud dataset to DataFrame
def create_earbud_dataframe(base_path):
    """Convert earbud image dataset to pandas DataFrame"""
    data = []
    
    if not os.path.exists(base_path):
        print(f"Earbud dataset path not found: {base_path}")
        return pd.DataFrame(data)
    
    # Process train, test, and valid folders - combine all into one dataset
    total_images = 0
    for split in ['train', 'test', 'valid']:
        split_path = os.path.join(base_path, split)
        if os.path.exists(split_path):
            # Get all image files directly in the split folder
            image_files = [f for f in os.listdir(split_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')) 
                          and not f.startswith('_')]  # Skip annotation files
            
            for image_file in image_files:
                image_path = os.path.join(split_path, image_file)
                if os.path.exists(image_path):
                    data.append({
                        'image_path': image_path,
                        'type': 'earbud'
                    })
            
            total_images += len(image_files)
    
    df = pd.DataFrame(data)
    print(f"Total earbud records: {len(df)} (combined from train/test/valid)")
    return df

# Create earbud DataFrame
earbud_base_path = r'C:\lost-and-found\earbud'
earbud = create_earbud_dataframe(earbud_base_path)

# Convert all key datasets to DataFrame
def create_key_dataframe():
    """Convert all key-related datasets to a single pandas DataFrame"""
    data = []
    
    # 1. Car Key Detector.v2i.tensorflow
    car_key_v2i_path = r'C:\lost-and-found\Car Key Detector.v2i.tensorflow'
    if os.path.exists(car_key_v2i_path):
        for split in ['train', 'valid']:
            split_path = os.path.join(car_key_v2i_path, split)
            if os.path.exists(split_path):
                image_files = [f for f in os.listdir(split_path) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')) 
                              and not f.startswith('_')]
                for image_file in image_files:
                    image_path = os.path.join(split_path, image_file)
                    if os.path.exists(image_path):
                        data.append({
                            'image_path': image_path,
                            'type': 'key'
                        })
    
    # 2. Car-key.v2-yolov8.tensorflow
    car_key_yolo_path = r'C:\lost-and-found\Car-key.v2-yolov8.tensorflow'
    if os.path.exists(car_key_yolo_path):
        for split in ['train', 'test', 'valid']:
            split_path = os.path.join(car_key_yolo_path, split)
            if os.path.exists(split_path):
                image_files = [f for f in os.listdir(split_path) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')) 
                              and not f.startswith('_')]
                for image_file in image_files:
                    image_path = os.path.join(split_path, image_file)
                    if os.path.exists(image_path):
                        data.append({
                            'image_path': image_path,
                            'type': 'key'
                        })
    
    # 3. Normal key folder
    normal_key_path = r'C:\lost-and-found\normal key'
    if os.path.exists(normal_key_path):
        # Process images directly in the folder
        image_files = [f for f in os.listdir(normal_key_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        for image_file in image_files:
            image_path = os.path.join(normal_key_path, image_file)
            if os.path.exists(image_path):
                data.append({
                    'image_path': image_path,
                    'type': 'key'
                })
        
        # Process subfolders
        for subfolder in ['dark-contrast', 'non-white-background', 'shadowed']:
            subfolder_path = os.path.join(normal_key_path, subfolder)
            if os.path.exists(subfolder_path):
                image_files = [f for f in os.listdir(subfolder_path) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
                for image_file in image_files:
                    image_path = os.path.join(subfolder_path, image_file)
                    if os.path.exists(image_path):
                        data.append({
                            'image_path': image_path,
                            'type': 'key'
                        })
    
    df = pd.DataFrame(data)
    
    if len(df) > 0:
        print(f"Total key records: {len(df)}")
        print("Key type distribution:")
        print(df['type'].value_counts())
    else:
        print("No key images found")
    
    return df

# Create key DataFrame
keys = create_key_dataframe()

# Convert umbrellas dataset to DataFrame
def create_umbrellas_dataframe(base_path):
    """Convert umbrellas image dataset to pandas DataFrame"""
    data = []
    
    # Navigate to the data folder
    data_path = os.path.join(base_path, 'images.cv_x7krft41s98erwqyuwyhk5', 'data')
    
    if os.path.exists(data_path):
        # Process train, test, and val folders
        for split in ['train', 'test', 'val']:
            split_path = os.path.join(data_path, split)
            if os.path.exists(split_path):
                # Process umbrella folder
                umbrella_path = os.path.join(split_path, 'umbrella')
                if os.path.isdir(umbrella_path):
                    # Process each image file
                    for image_file in os.listdir(umbrella_path):
                        if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                            image_path = os.path.join(umbrella_path, image_file)
                            data.append({
                                'image_path': image_path,
                                'type': 'umbrella'
                            })
    
    df = pd.DataFrame(data)
    print(f"Total umbrella records: {len(df)}")
    return df

# Create umbrellas DataFrame
umbrellas_base_path = r'C:\lost-and-found\umbrellas'
umbrellas = create_umbrellas_dataframe(umbrellas_base_path)

# Calculator dataset removed

# Convert notebooks dataset to DataFrame
def create_notebooks_dataframe(base_path):
    """Convert notebooks image dataset to pandas DataFrame"""
    data = []
    
    # Navigate to the data folder
    data_path = os.path.join(base_path, 'images.cv_lko4id3m3rj9xlfmhyt7o6', 'data')
    
    if os.path.exists(data_path):
        # Process train, test, and val folders
        for split in ['train', 'test', 'val']:
            split_path = os.path.join(data_path, split)
            if os.path.exists(split_path):
                # Process notebook subfolders (like "office_equipment paper_notebook")
                for subfolder in os.listdir(split_path):
                    subfolder_path = os.path.join(split_path, subfolder)
                    if os.path.isdir(subfolder_path):
                        # Process each image file
                        for image_file in os.listdir(subfolder_path):
                            if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                                image_path = os.path.join(subfolder_path, image_file)
                                data.append({
                                    'image_path': image_path,
                                    'type': 'notebook'
                                })
    
    df = pd.DataFrame(data)
    print(f"Total notebook records: {len(df)}")
    return df

# Convert chargers dataset to DataFrame
def create_chargers_dataframe(base_path):
    """Convert chargers image dataset to pandas DataFrame"""
    data = []
    
    # Navigate to the data folder
    data_path = os.path.join(base_path, 'images.cv_l0ka40y3ha5v0biy83zz', 'data')
    
    if os.path.exists(data_path):
        # Process train, test, and val folders
        for split in ['train', 'test', 'val']:
            split_path = os.path.join(data_path, split)
            if os.path.exists(split_path):
                # Process charger subfolders (like "calculator", "office_equipment calculator")
                for subfolder in os.listdir(split_path):
                    subfolder_path = os.path.join(split_path, subfolder)
                    if os.path.isdir(subfolder_path):
                        # Process each image file
                        for image_file in os.listdir(subfolder_path):
                            if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                                image_path = os.path.join(subfolder_path, image_file)
                                data.append({
                                    'image_path': image_path,
                                    'type': 'charger'
                                })
    
    df = pd.DataFrame(data)
    print(f"Total charger records: {len(df)}")
    return df

# Create notebooks DataFrame
notebooks_base_path = r'C:\lost-and-found\notebooks'
notebooks = create_notebooks_dataframe(notebooks_base_path)

# Create chargers DataFrame
chargers_base_path = r'C:\lost-and-found\chargers'
chargers = create_chargers_dataframe(chargers_base_path)

# Convert purses and backpacks datasets to combined bags DataFrame
def create_bags_dataframe():
    """Convert purses and backpacks image datasets to a single bags pandas DataFrame"""
    data = []
    
    # 1. Process purses dataset
    purses_data_path = os.path.join(r'C:\lost-and-found\purses', 'images.cv_dil70dux3uw8w3quj2foi', 'data')
    if os.path.exists(purses_data_path):
        for split in ['train', 'test', 'val']:
            split_path = os.path.join(purses_data_path, split)
            if os.path.exists(split_path):
                # Process purse folder
                purse_path = os.path.join(split_path, 'purse')
                if os.path.isdir(purse_path):
                    for image_file in os.listdir(purse_path):
                        if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                            image_path = os.path.join(purse_path, image_file)
                            data.append({
                                'image_path': image_path,
                                'type': 'bag'
                            })
    
    # 2. Process backpacks dataset
    backpacks_data_path = os.path.join(r'C:\lost-and-found\backpacks', 'images.cv_snurq733hgldfujaowqr4e', 'data')
    if os.path.exists(backpacks_data_path):
        for split in ['train', 'test', 'val']:
            split_path = os.path.join(backpacks_data_path, split)
            if os.path.exists(split_path):
                # Process backpack folder
                backpack_path = os.path.join(split_path, 'backpack')
                if os.path.isdir(backpack_path):
                    for image_file in os.listdir(backpack_path):
                        if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                            image_path = os.path.join(backpack_path, image_file)
                            data.append({
                                'image_path': image_path,
                                'type': 'bag'
                            })
    
    df = pd.DataFrame(data)
    print(f"Total bag records: {len(df)} (combined purses and backpacks)")
    return df

# Create bags DataFrame (combining purses and backpacks)
bags = create_bags_dataframe()

# Convert IDs dataset to DataFrame
def create_ids_dataframe(base_path):
    """Convert IDs image dataset to pandas DataFrame"""
    data = []
    
    if os.path.exists(base_path):
        # Process all image files directly in the IDs folder
        image_files = [f for f in os.listdir(base_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        
        for image_file in image_files:
            image_path = os.path.join(base_path, image_file)
            if os.path.exists(image_path):
                data.append({
                    'image_path': image_path,
                    'type': 'id'
                })
    
    df = pd.DataFrame(data)
    print(f"Total ID records: {len(df)}")
    return df

# Create IDs DataFrame
ids_base_path = r'C:\lost-and-found\ids'
ids = create_ids_dataframe(ids_base_path)

# Convert headphones dataset to DataFrame
def create_headphones_dataframe(base_path):
    """Convert headphones image dataset to pandas DataFrame"""
    data = []
    
    # Navigate to the data folder
    data_path = os.path.join(base_path, 'images.cv_0zr6t67y5j6avxcwnmsqj8', 'data')
    
    if os.path.exists(data_path):
        # Process train, test, and val folders
        for split in ['train', 'test', 'val']:
            split_path = os.path.join(data_path, split)
            if os.path.exists(split_path):
                # Process headphones folder
                headphones_path = os.path.join(split_path, 'headphones')
                if os.path.isdir(headphones_path):
                    # Process each image file
                    for image_file in os.listdir(headphones_path):
                        if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                            image_path = os.path.join(headphones_path, image_file)
                            data.append({
                                'image_path': image_path,
                                'type': 'headphones'
                            })
    
    df = pd.DataFrame(data)
    print(f"Total headphones records: {len(df)}")
    return df

# Create headphones DataFrame
headphones_base_path = r'C:\lost-and-found\headphones'
headphones = create_headphones_dataframe(headphones_base_path)

# Convert phones dataset to DataFrame
def create_phones_dataframe(images_path):
    """Convert phones dataset to pandas DataFrame"""
    data = []
    
    # Loop through all files in the images directory
    for filename in os.listdir(images_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(images_path, filename)
            data.append({
                'image_path': image_path,
                'type': 'phone'
            })
    
    df = pd.DataFrame(data)
    print(f"Phones dataset: {len(df)} images")
    return df

# Create phones DataFrame
phones_base_path = r"c:\lost-and-found\phones\images"
phones = create_phones_dataframe(phones_base_path)

# Check sizes of all DataFrames
print("\n" + "="*50)
print("DATASET SIZES SUMMARY:")
print("="*50)

datasets = {
    'laptops': laptops,
    'shoes': shoes, 
    'wallets': wallets,
    'earbud': earbud,
    'keys': keys,
    'umbrellas': umbrellas,
    'notebooks': notebooks,
    'chargers': chargers,
    'bags': bags,
    'ids': ids,
    'headphones': headphones,
    'phones': phones
}

sizes = {}
for name, df in datasets.items():
    size = len(df)
    sizes[name] = size
    print(f"{name:12}: {size:,} images")

# Find smallest dataset
min_size = min(sizes.values())
min_dataset = min(sizes, key=sizes.get)

print("="*50)
print(f"SMALLEST DATASET: {min_dataset} with {min_size:,} images")
print(f"TOTAL IMAGES: {sum(sizes.values()):,}")
print("="*50)








# MEMORY-ONLY AUGMENTATION STRATEGY
# Update DataFrames directly without saving images to disk

import random
from PIL import Image, ImageEnhance, ImageFilter
import io
import base64

def augment_dataframe_memory_only(source_df, target_size=400):
    """
    Augment DataFrame to target size using virtual image paths (memory only)
    This creates new DataFrame rows with synthetic paths but doesn't save actual images
    """
    current_size = len(source_df)
    needed = target_size - current_size
    
    if needed <= 0:
        print(f"Dataset already has {current_size} images (>= target {target_size})")
        return source_df.copy()
    
    print(f"Augmenting {source_df.iloc[0]['type']} from {current_size} to {target_size} images ({needed} needed)")
    
    # Augmentation techniques (names only for virtual paths)
    augmentations = [
        'rotate_15', 'rotate_-15', 'rotate_30', 'rotate_-30',
        'brightness_1.2', 'brightness_0.8', 'contrast_1.3', 'contrast_0.7',
        'flip_horizontal', 'blur_slight', 'sharpen', 'saturation_1.3', 'saturation_0.7'
    ]
    
    augmented_data = []
    original_images = source_df['image_path'].tolist()
    dataset_type = source_df.iloc[0]['type']
    
    # Generate virtual augmented entries
    for i in range(needed):
        # Select random original image and augmentation
        orig_path = random.choice(original_images)
        aug_name = random.choice(augmentations)
        
        # Create virtual augmented path (won't actually exist on disk)
        orig_name = os.path.splitext(os.path.basename(orig_path))[0]
        virtual_path = f"virtual_aug_{dataset_type}_{orig_name}_{aug_name}_{i:03d}.jpg"
        
        augmented_data.append({
            'image_path': virtual_path,
            'type': dataset_type,
            'is_augmented': True,
            'source_image': orig_path,
            'augmentation': aug_name
        })
    
    # Combine original and augmented data
    original_data = source_df.copy()
    original_data['is_augmented'] = False
    original_data['source_image'] = original_data['image_path']  
    original_data['augmentation'] = 'none'
    
    augmented_df = pd.DataFrame(augmented_data)
    combined_df = pd.concat([original_data, augmented_df], ignore_index=True)
    
    print(f"✅ Created {len(augmented_data)} virtual augmented entries")
    print(f"📊 Final dataset size: {len(combined_df)} images")
    
    return combined_df

# Apply memory-only augmentation to the three smallest datasets
print("🎯 MEMORY-ONLY AUGMENTATION TO ~400 IMAGES")
print("="*60)

# Update keys DataFrame 
keys_augmented = augment_dataframe_memory_only(keys, target_size=400)

# Update notebooks DataFrame  
notebooks_augmented = augment_dataframe_memory_only(notebooks, target_size=400)

# Update phones DataFrame
phones_augmented = augment_dataframe_memory_only(phones, target_size=400)

print("\n📈 FINAL AUGMENTED SIZES:")
print(f"Keys:      {len(keys):3d} → {len(keys_augmented):3d} images")
print(f"Notebooks: {len(notebooks):3d} → {len(notebooks_augmented):3d} images") 
print(f"Phones:    {len(phones):3d} → {len(phones_augmented):3d} images")

# Helper function to standardize DataFrame columns
def standardize_dataframe(df):
    """Add missing columns to make all DataFrames compatible"""
    df = df.copy()
    if 'is_augmented' not in df.columns:
        df['is_augmented'] = False
    if 'source_image' not in df.columns:
        df['source_image'] = df.get('full_image_path', df['image_path'])
    if 'augmentation' not in df.columns:
        df['augmentation'] = 'none'
    return df

# Create datasets_augmented dictionary with standardized columns
datasets_augmented = {
    'laptops': standardize_dataframe(laptops),
    'shoes': standardize_dataframe(shoes), 
    'wallets': standardize_dataframe(wallets),
    'earbud': standardize_dataframe(earbud),
    'keys': keys_augmented,  # Already has augmentation columns
    'umbrellas': standardize_dataframe(umbrellas),
    'notebooks': notebooks_augmented,  # Already has augmentation columns
    'chargers': standardize_dataframe(chargers),
    'bags': standardize_dataframe(bags),
    'ids': standardize_dataframe(ids),
    'headphones': standardize_dataframe(headphones),
    'phones': phones_augmented  # Already has augmentation columns
}

# Create balanced dictionary for ML training
balanced = {}
target_size = 400
for name, df in datasets_augmented.items():
    if len(df) <= target_size:
        balanced[name] = df
    else:
        # Sample down to target size for large datasets
        balanced[name] = df.sample(n=target_size, random_state=42)

# CHECK AVAILABLE DATAFRAMES AND CREATE FINAL COMBINED DATASET
print("🔍 CURRENT AVAILABLE DATAFRAMES:")
print("="*50)

# Show individual augmented DataFrames
individual_augmented = {
    'keys_augmented': keys_augmented,
    'notebooks_augmented': notebooks_augmented, 
    'phones_augmented': phones_augmented
}

for name, df in individual_augmented.items():
    print(f"{name:20}: {len(df):4d} images")

print(f"\n📚 DATASETS_AUGMENTED DICTIONARY:")
print("-" * 40)
for name, df in datasets_augmented.items():
    is_augmented = "_augmented" in name or name in ['keys', 'notebooks', 'phones']
    marker = "🔄" if is_augmented else "📊"
    print(f"{marker} {name:12}: {len(df):4d} images")

print(f"\n📚 BALANCED DICTIONARY:")
print("-" * 40)
for name, df in balanced.items():
    print(f"⚖️  {name:12}: {len(df):4d} images")

# CREATE FINAL COMBINED DATAFRAME
print(f"\n🎯 CREATING FINAL COMBINED DATAFRAME:")
print("="*50)

final_df = pd.concat(datasets_augmented.values(), ignore_index=True)

print(f"✅ final_df created with {len(final_df):,} total images")
print(f"📊 Type distribution:")
print(final_df['type'].value_counts())





# Encode labels
le = LabelEncoder()
final_df['label'] = le.fit_transform(final_df['type'])

# Custom Dataset
class ImageTagDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = row['label']
        
        # Check if this is an augmented image (virtual path)
        if row.get('is_augmented', False):
            # Load from source image path
            img_path = row['source_image']
            
            # Debug: Check if img_path is valid
            if pd.isna(img_path) or not isinstance(img_path, str):
                print(f"Error at idx {idx}: source_image is {img_path} (type: {type(img_path)})")
                print(f"Row data: {row}")
                raise ValueError(f"Invalid source_image path: {img_path}")
            
            image = Image.open(img_path).convert('RGB')
            
            # Apply the specific augmentation
            image = self._apply_augmentation(image, row['augmentation'])
        else:
            # Regular image - try different path options
            img_path = None
            
            # Try full_image_path first
            if 'full_image_path' in row and pd.notna(row['full_image_path']):
                img_path = row['full_image_path']
            # Then try source_image
            elif 'source_image' in row and pd.notna(row['source_image']):
                img_path = row['source_image']
            # Finally try image_path
            elif 'image_path' in row and pd.notna(row['image_path']):
                img_path = row['image_path']
            
            # Debug: Check if img_path is valid
            if img_path is None or pd.isna(img_path) or not isinstance(img_path, str):
                print(f"Error at idx {idx}: img_path is {img_path} (type: {type(img_path)})")
                print(f"Row data: {row}")
                raise ValueError(f"Invalid image path: {img_path}")
            
            image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def _apply_augmentation(self, image, aug_type):
        """Apply specific augmentation to image"""
        from PIL import ImageEnhance, ImageFilter
        
        if aug_type == 'none':
            return image
        elif aug_type == 'rotate_15':
            return image.rotate(15, expand=True, fillcolor='white')
        elif aug_type == 'rotate_-15':
            return image.rotate(-15, expand=True, fillcolor='white')
        elif aug_type == 'rotate_30':
            return image.rotate(30, expand=True, fillcolor='white')
        elif aug_type == 'rotate_-30':
            return image.rotate(-30, expand=True, fillcolor='white')
        elif aug_type == 'brightness_1.2':
            return ImageEnhance.Brightness(image).enhance(1.2)
        elif aug_type == 'brightness_0.8':
            return ImageEnhance.Brightness(image).enhance(0.8)
        elif aug_type == 'contrast_1.3':
            return ImageEnhance.Contrast(image).enhance(1.3)
        elif aug_type == 'contrast_0.7':
            return ImageEnhance.Contrast(image).enhance(0.7)
        elif aug_type == 'flip_horizontal':
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        elif aug_type == 'blur_slight':
            return image.filter(ImageFilter.GaussianBlur(radius=0.5))
        elif aug_type == 'sharpen':
            return image.filter(ImageFilter.SHARPEN)
        elif aug_type == 'saturation_1.3':
            return ImageEnhance.Color(image).enhance(1.3)
        elif aug_type == 'saturation_0.7':
            return ImageEnhance.Color(image).enhance(0.7)
        else:
            return image  # Unknown augmentation, return original

# Enhanced Image transforms with data augmentation
train_transform = transforms.Compose([
    transforms.Resize((144, 144)),  # Slightly larger for random crop
    transforms.RandomCrop((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
])

# Validation/test transform (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = ImageTagDataset(final_df, transform=val_transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

# Improved CNN Model with Batch Normalization and Dropout
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Second convolutional block
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        # Third convolutional block
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # 128x128 -> 64x64
        
        # Second block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)  # 64x64 -> 32x32
        
        # Third block
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)  # 32x32 -> 16x16
        
        # Global average pooling
        x = self.global_pool(x)  # 16x16 -> 1x1
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x

# Transfer Learning Model with ResNet18
class TransferLearningModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(TransferLearningModel, self).__init__()
        
        # Load pre-trained ResNet18 with compatibility for different torchvision versions
        try:
            if pretrained:
                self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.resnet18(weights=None)
        except AttributeError:
            # Fallback for older torchvision versions
            self.backbone = models.resnet18(pretrained=pretrained)
        
        # Freeze early layers (optional - can be unfrozen for fine-tuning)
        if pretrained:
            # Freeze all layers except the last few
            for name, param in self.backbone.named_parameters():
                if 'layer4' not in name and 'fc' not in name:
                    param.requires_grad = False
        
        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

num_classes = len(le.classes_)
model = ImprovedCNN(num_classes)
print(model)

# Option to use subset for faster training (set to False for full dataset)
USE_SUBSET = True
SUBSET_SIZE = 1  # Use 30% of data for faster training

if USE_SUBSET:
    # Use a smaller subset for faster experimentation
    subset_df = final_df.sample(frac=SUBSET_SIZE, random_state=46)
    print(f"🚀 Using subset: {len(subset_df):,} images (vs {len(final_df):,} full dataset)")
    train_df, test_df = train_test_split(subset_df, test_size=0.2, stratify=subset_df['label'], random_state=42)
else:
    # Split full dataset for training
    train_df, test_df = train_test_split(final_df, test_size=0.2, stratify=final_df['label'], random_state=42)

# Create datasets and dataloaders
train_dataset = ImageTagDataset(train_df, transform=train_transform)
test_dataset = ImageTagDataset(test_df, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=False)

# Advanced Training with Scheduling and Early Stopping
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Choose between ImprovedCNN and Transfer Learning
USE_TRANSFER_LEARNING = True  # Set to False to use ImprovedCNN

if USE_TRANSFER_LEARNING:
    model = TransferLearningModel(num_classes, pretrained=True).to(device)
    print("Using Transfer Learning with ResNet18")
else:
    model = ImprovedCNN(num_classes).to(device)
    print("Using Improved CNN")

# Compile model for faster execution (PyTorch 2.0+) - DISABLED due to compatibility issues
# try:
#     model = torch.compile(model)
#     print("✅ Model compiled for faster execution")
# except:
#     print("⚠️ Model compilation not available (PyTorch 2.0+ required)")
print("⚠️ Model compilation disabled for compatibility")
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# Calculate class weights to handle imbalance
class_counts = final_df['type'].value_counts().sort_index()
total_samples = len(final_df)
class_weights = []

print("Class distribution and weights:")
for class_name in le.classes_:
    count = class_counts[class_name]
    weight = total_samples / (num_classes * count)
    class_weights.append(weight)
    print(f"  {class_name}: {count} samples, weight: {weight:.3f}")

class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Early stopping parameters
best_val_acc = 0.0
patience = 7
patience_counter = 0
best_model_state = None

num_epochs = 15
train_losses = []
val_accuracies = []

print("Starting enhanced training...")
for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader.dataset)
    train_acc = 100 * train_correct / train_total
    
    # Validation phase
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_loss = val_loss / len(test_loader.dataset)
    val_acc = 100 * val_correct / val_total
    
    train_losses.append(epoch_loss)
    val_accuracies.append(val_acc)
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Early stopping and model checkpointing
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        best_model_state = model.state_dict().copy()
        print(f"  ✅ New best validation accuracy: {val_acc:.2f}%")
    else:
        patience_counter += 1
        print(f"  ⏳ No improvement. Patience: {patience_counter}/{patience}")
    
    if patience_counter >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break
    
    print("-" * 50)

# Load best model
if best_model_state:
    model.load_state_dict(best_model_state)
    print(f"Loaded best model with validation accuracy: {best_val_acc:.2f}%")

print("Enhanced training complete!")

# Save the best model
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_save_path = f'lost_found_model_acc_{best_val_acc:.1f}_{timestamp}.pth'

torch.save({
    'model_state_dict': model.state_dict(),
    'best_val_acc': best_val_acc,
    'num_classes': num_classes,
    'class_names': list(le.classes_),
    'model_type': 'TransferLearningModel_ResNet18',
    'training_completed': True,
    'timestamp': timestamp
}, model_save_path)

print(f"✅ Model saved to: {model_save_path}")
print(f"📊 Model info: {num_classes} classes, {best_val_acc:.2f}% validation accuracy")

# Final comprehensive evaluation
print("\n" + "="*60)
print("FINAL MODEL EVALUATION")
print("="*60)

model.eval()
correct = 0
total = 0
class_correct = list(0. for i in range(num_classes))
class_total = list(0. for i in range(num_classes))

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Per-class accuracy
        c = (predicted == labels).squeeze()
        for i in range(labels.size(0)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

overall_accuracy = 100 * correct / total
print(f"Overall Test Accuracy: {overall_accuracy:.2f}%")
print(f"Improvement over baseline: +{overall_accuracy - 59.92:.2f}%")

print("\nPer-class accuracy:")
for i in range(num_classes):
    if class_total[i] > 0:
        class_acc = 100 * class_correct[i] / class_total[i]
        class_name = le.classes_[i]
        print(f"  {class_name:12}: {class_acc:.2f}% ({int(class_correct[i])}/{int(class_total[i])})")

print("\n" + "="*60)
print("ACCURACY IMPROVEMENT SUMMARY:")
print("="*60)
print("✅ Enhanced data augmentation (rotation, flip, color jitter, normalization)")
print("✅ Improved CNN architecture (deeper, batch norm, dropout)")
print("✅ Transfer learning with pre-trained ResNet18")
print("✅ Advanced training (LR scheduling, early stopping, gradient clipping)")
print("✅ Weighted loss function for class imbalance")
print("✅ AdamW optimizer with weight decay")
print("="*60)





