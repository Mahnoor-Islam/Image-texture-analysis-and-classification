import cv2
from skimage import feature
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity

# Function to compute LBP texture features with grayscale conversion and image resizing
def extract_lbp_texture(image, P=8, R=1):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize the image for processing
    resized_image = cv2.resize(gray_image, (150, 150))  # Keep original image at 150x150 pixels

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)  # Moderate blur
    
    # Apply LBP texture extraction
    lbp = feature.local_binary_pattern(blurred_image, P, R, method="uniform")
    
    # Normalize LBP for better visualization
    lbp_normalized = (lbp - lbp.min()) / (lbp.max() - lbp.min()) * 255
    
    # Convert LBP to uint8 for visualization
    lbp_image = lbp_normalized.astype(np.uint8)

    # Adjust texture intensity for better visualization
    lbp_image = cv2.convertScaleAbs(lbp_image, alpha=0.5, beta=0)  # Scale the texture intensity down

    # Reduce the size of the texture to 50x50 pixels
    small_lbp_image = cv2.resize(lbp_image, (50, 50))  # Resize texture to 50x50 pixels

    return gray_image, small_lbp_image  # Return both grayscale and reduced-size LBP images

# Simulating class creation with image loading for 'moon', 'sun', and 'mountain'
def load_images():
    # Each class has 5 images
    moon = [f'moon_img_{i}.jpg' for i in range(1, 6)]
    sun = [f'sun_img_{i}.jpg' for i in range(1, 6)]
    mountain = [f'mountain_img_{i}.jpg' for i in range(1, 6)]

    # Dictionary mapping class names to image lists
    classes = {
        'moon': moon,
        'sun': sun,
        'mountain': mountain
    }
    
    # Loading images (simulated with random matrices in this case)
    images = []
    for class_name, img_list in classes.items():
        for img in img_list:
            # Replace with actual image loading using cv2.imread(img)
            img_matrix = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)  # Simulated random images
            images.append((img_matrix, class_name))
    
    return images

# Function to randomly select an image and compute its texture
def random_image_selection(images):
    selected_image, class_name = random.choice(images)
    gray_image, lbp = extract_lbp_texture(selected_image)
    return selected_image, gray_image, lbp, class_name

# Function to calculate the standard deviation of LBP texture
def calculate_std(texture):
    return np.std(texture)

# Function to compare textures using cosine similarity
def compare_textures(texture_a, texture_b):
    return cosine_similarity([texture_a.flatten()], [texture_b.flatten()])[0][0]

# Main program
def main():
    # Load the images from the classes 'moon', 'sun', and 'mountain'
    images = load_images()

    # Randomly select two images and extract their textures
    random_image_a, gray_a, lbp_a, class_a = random_image_selection(images)
    print(f"Randomly selected image A belongs to: {class_a}")
    
    random_image_b, gray_b, lbp_b, class_b = random_image_selection(images)
    print(f"Randomly selected image B belongs to: {class_b}")

    # Calculate standard deviations
    std_a = calculate_std(lbp_a)
    std_b = calculate_std(lbp_b)

    # Calculate cosine similarity between the textures
    similarity = compare_textures(lbp_a, lbp_b)

    # Display both original images (converted to grayscale) and their resized textures (grayscale)
    plt.figure(figsize=(10, 5))
    
    # Display grayscale version of selected image (image A)
    plt.subplot(1, 2, 1)
    plt.title("Selected Image")
    plt.imshow(gray_a, cmap='gray')  # Displaying the grayscale selected image
    plt.axis('off')

    # Display reduced-size LBP texture of random image (image B)
    plt.subplot(1, 2, 2)
    plt.title("Random Image")
    plt.imshow(gray_b, cmap='gray')  # Displaying the grayscale random image
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Print the standard deviations and cosine similarity
    print(f"Standard Deviation of Texture A: {std_a:.4f}")
    print(f"Standard Deviation of Texture B: {std_b:.4f}")
    print(f"Cosine similarity between textures: {similarity:.4f}")

if __name__ == "__main__":
    main()
