import cv2
import numpy as np
from sklearn.cluster import KMeans
from openai import OpenAI
import matplotlib.pyplot as plt

# Initialize the OpenAI client
api_key = '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'
client = OpenAI(api_key=api_key)

# Function to load and convert the image to RGB format
def load_and_convert_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

# Function to determine the number of clusters using GPT
def get_number_of_clusters(image_rgb):
    prompt = ("Based on the input image shape " + str(image_rgb.shape) + 
        ", determine the appropriate number of clusters (K) for KMeans clustering. "
        "Only provide the number (an integer)."
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=10,
        top_p=1
    )
    
    # Print the GPT response to understand its output
    print(f"GPT response for cluster determination: {response.choices[0].message.content}")
    
    # Extract and parse the number of clusters from the response
    try:
        num_clusters = int(response.choices[0].message.content.strip())
    except ValueError:
        num_clusters = 3  # Default value if parsing fails
    return num_clusters

# Function to perform KMeans clustering
def cluster_image(image_rgb, num_clusters):
    pixels = image_rgb.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(pixels)
    clustered = kmeans.labels_.reshape(image_rgb.shape[:2])
    return clustered, kmeans.cluster_centers_

# Function to save the clustered image without displaying it
def save_clustered_image(clustered_image, cluster_centers, image_rgb):
    # Map the cluster labels to their respective colors
    clustered_colors = np.zeros_like(image_rgb)
    
    for i in range(len(cluster_centers)):
        clustered_colors[clustered_image == i] = cluster_centers[i]
    
    # Save the image
    plt.imshow(clustered_colors.astype(np.uint8))
    plt.axis('off')  # Remove axes for a cleaner image
    
    output_path = 'clustered_image_output.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Clustered image saved to {output_path}")
    
    # Clear the figure after saving to prevent overlap in future plots
    plt.clf()

# Function to classify each cluster using GPT based on cluster center RGB color
def classify_clusters(cluster_centers, num_clusters):
    classifications = {}
    typical_info = (
        "[106, 147, 222] is the typical RGB for monolayer 2D material, "
        "[143, 143, 210] is the typical RGB for substrate, "
        "[35, 193, 244] is the typical RGB for impurities."
    )
    
    for cluster_id in range(num_clusters):
        # Extract the RGB color of the cluster center
        avg_color = cluster_centers[cluster_id]

        # Create a prompt to classify based on the RGB color, including the typical values information
        prompt = (
            f"{typical_info}\n"
            f"Now, based on the RGB color value {avg_color}, classify it as a substrate, 2D material, or impurity. "
            "Only provide the category name (substrate, 2D material, or impurity)."
        )
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=50,
            top_p=1
        )
        
        # Print the GPT response along with the RGB color value
        print(f"Cluster {cluster_id} - RGB: {avg_color}")
        print(f"GPT response for cluster {cluster_id} classification: {response.choices[0].message.content}\n")
        
        label = response.choices[0].message.content.strip()
        classifications[cluster_id] = label
    return classifications


# # Function to separate disconnected regions in the 2D material cluster and trace their edges
# def separate_disconnected_regions(clustered_image, material_id, original_image):
#     material_mask = (clustered_image == material_id).astype(np.uint8)
    
#     # Find connected components in the material mask
#     num_labels, labels_im = cv2.connectedComponents(material_mask)
    
#     # Loop over each connected component to trace the edges
#     for region_id in range(1, num_labels):  # Starting from 1 to skip the background
#         # Create a mask for the current region
#         region_mask = (labels_im == region_id).astype(np.uint8)
        
#         # Find the contours of the region
#         contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         # Draw the contours (outline) on the original image
#         cv2.drawContours(original_image, contours, -1, (0, 255, 0), 2)  # Green contour lines

#     return labels_im, num_labels

# Function to separate disconnected regions in the 2D material cluster
# and return valid regions that meet the size requirement
def separate_disconnected_regions(clustered_image, material_id):
    material_mask = (clustered_image == material_id).astype(np.uint8)
    
    # Find connected components in the material mask
    num_labels, labels_im = cv2.connectedComponents(material_mask)
    
    return labels_im, num_labels

# Function to get the pixel-to-length conversion factor based on magnification
def get_pixel_to_length_conversion(magnification):
    if magnification == 10:
        return 100 / 91  # 10x objective, 91 pixels represent 100 µm
    elif magnification == 50:
        return 20 / 91  # 50x objective, 91 pixels represent 20 µm
    elif magnification == 100:
        return 10 / 91  # 100x objective, 91 pixels represent 10 µm
    else:
        raise ValueError("Unsupported magnification level")

# Function to calculate the dimensions of each region and check size requirements
def calculate_line_dimensions(labels_im, num_labels, pixel_to_length_conversion, min_size):
    valid_regions = []
    for region_id in range(1, num_labels):
        # Get the bounding rectangle of the region
        region_mask = (labels_im == region_id).astype(np.uint8)
        x, y, w, h = cv2.boundingRect(region_mask)
        # Calculate the maximum dimension
        max_dimension = max(w, h) * pixel_to_length_conversion
        if max_dimension >= min_size:
            valid_regions.append((region_id, max_dimension))
    return valid_regions

# Main function to process the image and trace the edges of valid regions
def process_image(image_path, min_size, magnification):
    # Step 1: Load and convert the image
    image_rgb = load_and_convert_image(image_path)

    # Step 2: Determine the number of clusters using GPT
    num_clusters = get_number_of_clusters(image_rgb)

    # Step 3: Perform KMeans clustering
    clustered_image, cluster_centers = cluster_image(image_rgb, num_clusters)

    # Step 4: Save the clustered image (this part is retained from the previous implementation)
    save_clustered_image(clustered_image, cluster_centers, image_rgb)

    # Step 5: Classify each cluster based on RGB color
    classifications = classify_clusters(cluster_centers, num_clusters)

    # Step 6: Process the 2D material cluster
    pixel_to_length_conversion = get_pixel_to_length_conversion(magnification)
    original_image = cv2.imread(image_path)  # Load the original image to trace edges
    valid_regions_exist = False  # To track if there are valid regions to show

    for cluster_id, label in classifications.items():
        if label.lower() == "2d material":
            # Separate disconnected regions
            labels_im, num_labels = separate_disconnected_regions(clustered_image, cluster_id)
            
            # Calculate dimensions and check size requirements
            valid_regions = calculate_line_dimensions(
                labels_im, num_labels, pixel_to_length_conversion, min_size
            )
            
            # Draw contours for valid regions
            for region_id, dimension in valid_regions:
                print(f"Region {region_id} has a dimension of {dimension:.2f} µm, which meets the requirement.")
                
                # Create a mask for the current valid region
                region_mask = (labels_im == region_id).astype(np.uint8)
                
                # Find the contours of the region
                contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw the contours (outline) only for the valid regions
                cv2.drawContours(original_image, contours, -1, (0, 255, 0), 2)  # Green contour lines
                valid_regions_exist = True  # Set flag to true if valid regions exist

    # If there are valid regions, save and display the final image
    if valid_regions_exist:
        output_image_path = 'final_image_with_traced_edges.png'
        cv2.imwrite(output_image_path, original_image)
        print(f"Final image with traced edges saved to {output_image_path}")
        
        # Show the final image with traced edges using matplotlib
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying
        plt.title("Final Image with Valid Traced Edges")
        plt.axis('off')  # Hide axes for better display
        plt.show()
    else:
        print("No regions met the size requirement.")
    
    return classifications, valid_regions

# Example usage
if __name__ == "__main__":
    image_path = '/Users/yangjy/Repositories/OM_with_LLM_agent/PaliGemma/om_snap/Snap-1659.jpg'
    min_size = 200  # Minimum required size in micrometers
    magnification = 10  # Magnification level (10, 50, or 100)
    
    process_image(image_path, min_size, magnification)