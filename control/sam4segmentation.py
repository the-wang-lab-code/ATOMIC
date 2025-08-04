import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from sklearn.cluster import KMeans

# Initialize SAM model
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mask_generator = SamAutomaticMaskGenerator(model=sam)

def process_image_with_sam(image_path):
    """
    Process an image using the SAM model, filter overlapping masks, perform K-means clustering, 
    and progressively update the same matplotlib window with the original image, filtered masks, 
    and clustered mask visualization.
    
    Args:
        image_path: Path to the image file.

    Returns:
        result: A dictionary containing clustering results and mask information.
    """
    
    # Initialize the plot window
    plt.ion()  # Turn on interactive mode for matplotlib
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    fig.suptitle("Image Processing Pipeline", fontsize=16)
    
    # Load and process the image
    print(f"Loading image from: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    
    print(f"Image loaded successfully. Image shape: {image.shape}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_area = image.shape[0] * image.shape[1]
    print(f"Converted image to RGB. Total image area: {image_area}")
    
    # Display the original image
    ax.imshow(image_rgb)
    ax.set_title("Original Image")
    plt.pause(1)  # Pause for a second to visualize the original image
    
    # Generate masks using SAM
    print("Generating masks using SAM...")
    masks = mask_generator.generate(image)
    print(f"Number of masks generated: {len(masks)}")

    # Filter overlapping or fully contained masks
    print("Filtering overlapping or fully contained masks...")
    filtered_masks = filter_overlapping_masks([ann['segmentation'] for ann in masks])
    print(f"Number of masks after filtering: {len(filtered_masks)}")
    
    # Update the figure with filtered masks
    img_with_masks = plot_filtered_masks(image_rgb, filtered_masks, ax)
    ax.set_title("Filtered Masks Overlay")
    plt.pause(1)  # Pause to visualize filtered masks

    # Compute average RGB values for each mask
    print("Computing average RGB values for each mask...")
    rgb_values, mask_areas = [], []
    overall_mask = np.zeros(image_rgb.shape[:2], dtype=bool)

    for i, mask in enumerate(filtered_masks):
        mask_pixels = np.where(mask)
        print(f"Processing mask {i+1}/{len(filtered_masks)} with {len(mask_pixels[0])} pixels.")
        overall_mask = np.logical_or(overall_mask, mask)

        # Extract RGB values from the mask
        avg_r = np.mean(image_rgb[mask_pixels[0], mask_pixels[1], 0])
        avg_g = np.mean(image_rgb[mask_pixels[0], mask_pixels[1], 1])
        avg_b = np.mean(image_rgb[mask_pixels[0], mask_pixels[1], 2])

        rgb_values.append([avg_r, avg_g, avg_b])
        mask_areas.append(len(mask_pixels[0]))

    print("Finished computing RGB values for masks.")

    # Calculate average RGB values for unmasked (substrate) regions
    print("Calculating average RGB values for unmasked regions...")
    unmasked_pixels = np.where(~overall_mask)
    substrate_r = np.mean(image_rgb[unmasked_pixels[0], unmasked_pixels[1], 0])
    substrate_g = np.mean(image_rgb[unmasked_pixels[0], unmasked_pixels[1], 1])
    substrate_b = np.mean(image_rgb[unmasked_pixels[0], unmasked_pixels[1], 2])
    substrate_rgb = np.array([substrate_r, substrate_g, substrate_b])
    print(f"Calculated substrate RGB: {substrate_rgb}")

    # Perform K-means clustering on the RGB values
    print("Performing K-means clustering on the RGB values...")
    rgb_array = np.array(rgb_values)
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(rgb_array)
    labels = kmeans.labels_
    print(f"K-means clustering completed. Cluster labels: {labels}")

    # Calculate total area for each cluster
    cluster_areas = np.zeros(3)
    for idx, label in enumerate(labels):
        cluster_areas[label] += mask_areas[idx]
    print(f"Cluster areas: {cluster_areas}")

    # Find the cluster closest to the substrate
    print("Calculating the closest cluster to the substrate...")
    distances_to_substrate = np.linalg.norm(rgb_array - substrate_rgb, axis=1)
    average_distances = [np.mean(distances_to_substrate[labels == i]) for i in range(3)]
    closest_cluster = np.argmin(average_distances)
    print(f"Closest cluster to substrate: {closest_cluster}")

    # Update the figure with clustered mask visualization
    plot_clustered_masks(image_rgb, filtered_masks, labels, ax)
    ax.set_title("Clustered Masks")
    plt.pause(1)

    # Prepare the result
    result = {
        "closest_cluster": closest_cluster,
        "cluster_areas": cluster_areas.tolist(),
        "non_substrate_area_proportion": np.sum(cluster_areas) / image_area
    }

    print(f"Final result: {result}")
    
    # Keep the plot window open at the end
    plt.ioff()
    plt.show()

    return result

def filter_overlapping_masks(masks):
    """
    Filter out masks that are either fully contained within another mask or
    are completely overlapped by another mask.
    
    Args:
        masks: List of binary masks (2D arrays).

    Returns:
        filtered_masks: List of filtered masks with overlaps removed.
    """
    filtered_masks = []

    for i, mask_i in enumerate(masks):
        keep = True  # Flag to check if we should keep this mask

        for j, mask_j in enumerate(masks):
            if i == j:
                continue  # Skip comparison with itself

            # If mask_i is fully contained within mask_j, discard mask_i
            if np.all(mask_i & mask_j == mask_i):
                keep = False
                break

            # If mask_i and mask_j overlap but neither is fully contained
            intersection = mask_i & mask_j
            if np.any(intersection) and np.sum(intersection) == np.sum(mask_i):
                keep = False
                break

        if keep:
            filtered_masks.append(mask_i)

    return filtered_masks

def plot_filtered_masks(image_rgb, filtered_masks, ax):
    """
    Plot filtered masks over the original image using Matplotlib without opening a new window.

    Args:
        image_rgb: The original image in RGB format.
        filtered_masks: A list of binary masks that have been filtered.
        ax: The Matplotlib axis to update the plot.

    Returns:
        img_with_masks: The image with masks overlaid.
    """
    img_with_masks = image_rgb.copy()

    # Assign random colors to the masks
    for mask in filtered_masks:
        color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
        img_with_masks[mask] = color  # Apply color to mask region

    # Update the existing plot without opening a new window
    ax.imshow(img_with_masks)
    ax.set_title("Filtered Masks Overlay")
    ax.axis('off')

    return img_with_masks

def plot_clustered_masks(image_rgb, filtered_masks, labels, ax):
    """
    Plot clustered masks over the original image in the same Matplotlib window.

    Args:
        image_rgb: The original image in RGB format.
        filtered_masks: A list of binary masks that have been filtered.
        labels: The cluster labels for each mask.
        ax: The Matplotlib axis to update the plot.
    """
    img_with_clusters = image_rgb.copy()

    # Define cluster colors
    cluster_colors = [
        [255, 0, 0],  # Red
        [0, 255, 0],  # Green
        [0, 0, 255]   # Blue
    ]

    # Apply cluster colors to masks
    for idx, mask in enumerate(filtered_masks):
        color = cluster_colors[labels[idx]]
        img_with_clusters[mask] = color  # Apply color based on cluster

    # Update the existing plot without opening a new window
    ax.imshow(img_with_clusters)
    ax.set_title("Clustered Masks")
    ax.axis('off')
