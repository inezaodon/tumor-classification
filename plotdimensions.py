import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import kagglehub


#downloading data
path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")

# Create empty list containers to temporarily store data
widths = []
heights = []
classes = []



# Walk through the hidden cache folder to find all of the images and add their features to the arrays above
for root, dirs, files in os.walk(path):
    for file in files:
        # Check if the file is actually an image
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Glue folder path and file name together
            img_path = os.path.join(root, file)
            
            # Open the image in grayscale mode (fast)
            img = cv2.imread(img_path, 0)
            
            # If the image opened successfully, grab its features
            if img is not None:
                h, w = img.shape
                heights.append(h)
                widths.append(w)
                
                # Snatch the name of the folder the image lives in
                class_name = os.path.basename(root)
                classes.append(class_name)


# Package our lists into a structured spreadsheet of data (rows are each image, columns are width, height, and class)
df = pd.DataFrame({
    'Width (X-axis)': widths,
    'Height (Y-axis)': heights,
    'Class': classes
})

# ==========================================
# 4. PLOTTING WITH MATPLOTLIB
# ==========================================
# Initialize a blank canvas size 10x7 inches
plt.figure(figsize=(10, 7))

# Get a unique list of tumor types found in our dataframe
unique_classes = df['Class'].unique()

# Define 4 distinct hex-code colors for our 4 classes
colors = ['#FF5733', '#33FF57', '#3357FF', '#F3FF33']

# Loop through each of the 4 tumor classes and plot them layer by layer
for i, class_name in enumerate(unique_classes):
    # Filter the spreadsheet: Keep ONLY rows belonging to the current class loop
    class_data = df[df['Class'] == class_name]
    
    # Draw the scattered dots for this specific class
    plt.scatter(
        x=class_data['Width (X-axis)'], 
        y=class_data['Height (Y-axis)'], 
        label=class_name,
        alpha=0.6,
        edgecolors='none',
        s=30,
        color=colors[i]
    )

# Draw a 1:1 ratio diagonal line to spot perfectly square images
max_dim = max(max(widths), max(heights))
plt.plot([0, max_dim], [0, max_dim], color='gray', linestyle='--', alpha=0.7, label='Perfect Square (1:1)')

#display now
plt.title('Brain Tumor MRI Image Dimensions (Height vs. Width)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Width (Pixels)', fontsize=12)
plt.ylabel('Height (Pixels)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)

# Position the legend on the right side outside of the main chart area
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1)) 

# Adjust layout automatically so nothing gets clipped or cut off
plt.tight_layout()

# Render the window to the screen
plt.show()