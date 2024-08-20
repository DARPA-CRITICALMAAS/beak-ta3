import os
import re
from PIL import Image, ImageDraw, ImageFont
from fpdf import FPDF

# Function for natural sorting (i.e., handling numbers properly)
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def categorize_images(files):
    categories = {
        "Cluster results": [],
        "Parameter Maps SOM space": [],
        "Parameter Maps Geo space": [],
        "Boxplots": []
    }

    # Separate files by pattern and naturally sort them
    somplot_images = sorted([f for f in files if f.startswith('somplot_')], key=natural_sort_key)
    geoplot_images = sorted([f for f in files if f.startswith('geoplot_')], key=natural_sort_key)
    cluster_images = [f for f in files if f.startswith('cluster_')]
    boxplot_images = [f for f in files if f.startswith('boxplot_')]

    # Assign the last 4 somplot images to "Cluster results"
    if len(somplot_images) >= 4:
        categories["Cluster results"].extend(somplot_images[-4:-1])
        categories["Parameter Maps SOM space"].extend(somplot_images[:-4])
    else:
        categories["Cluster results"].extend(somplot_images)

    # Assign the first and last geoplot images to "Cluster results"
    if len(geoplot_images) > 0:
        categories["Cluster results"].append(geoplot_images[0])  # First geoplot image
        if len(geoplot_images) > 1:
            categories["Cluster results"].append(geoplot_images[-1])  # Last geoplot image
        categories["Parameter Maps Geo space"].extend(geoplot_images[1:-1])  # Middle images

    # Add remaining cluster images to "Cluster results"
    categories["Cluster results"].extend(cluster_images)

    # Add boxplot images to "Boxplots"
    categories["Boxplots"].extend(boxplot_images)
    
    return categories

def pngs_to_pdf(folder_path, output_pdf_path, number_of_images_per_row):
    pdf = FPDF('P', 'mm', 'A4')  # Create a PDF object, A4 size in portrait mode
     
    # Page size in mm
    page_width = 210  # A4 width in mm
    page_height = 297  # A4 height in mm
    margin = 10  # Margin around the images
    space = 5   # horizontal space between images 
    
    # Calculate available width for images (page width minus margins)
    available_width = page_width - 2 * margin - (number_of_images_per_row - 1) * space
    
    # Set width for each image to ensure x images per row
    img_width_mm = available_width / number_of_images_per_row

    # Get all PNG files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

    # Categorize images by file name
    categorized_images = categorize_images(files)

    

    # Iterate through the categories and add their images
    for category, image_files in categorized_images.items():
        
        print(category)

        if not image_files:
            continue  # Skip empty categories

        # Add a new page for the category and write the category title
        pdf.add_page()
        pdf.set_font("Arial", size=16)
        pdf.cell(200, 10, txt=category, ln=True, align='C')
        
        # Add space after the title
        pdf.ln(10)  

        # Initialize X and Y position for the first image
        x_position = margin
        y_position = pdf.get_y()
        current_row_images = 0  # Track images in the current row
        
        #img_height_max = 0

        for image_file in image_files:
            img_path = os.path.join(folder_path, image_file)

            # Open the image and calculate the aspect ratio
            with Image.open(img_path) as img:
                img_width, img_height = img.size
                aspect_ratio = img_height / img_width
                img_height_mm = img_width_mm * aspect_ratio

            # Check if the image fits in the current row, if not, start a new row
            if current_row_images >= number_of_images_per_row:
                # Reset X position and move Y down to start a new row
                x_position = margin
                y_position += img_height_mm + space
                current_row_images = 0  # Reset image count for the new row

            # Check if the image fits on the current page, if not, add a new page
            if y_position + img_height_mm > page_height - margin:
                pdf.add_page()
                y_position = margin  # Reset Y position after page break
                x_position = margin  # Reset X position after page break
                pdf.set_y(y_position)

            # Add the image at the current X, Y position
            pdf.image(img_path, x=x_position, y=y_position, w=img_width_mm)

            # Update X position for the next image in the same row
            x_position += img_width_mm + space
            current_row_images += 1  # Track images in the row

            # If the current row is full, move to the next line
            if current_row_images > number_of_images_per_row:
                y_position += space  # Add some vertical space between rows
                pdf.ln(img_height_mm + space)  # Move to the next line
                current_row_images = 0

        # Reset for next category
        y_position = margin

    # Save the PDF file
    pdf.output(output_pdf_path)
    print(f"PDF successfully created: {output_pdf_path}")

# Example usage
folder_path = 'E:\\20230082_CriticalMAAS\\GitHub\\beak-ta3\\experiments\\04_hackathon_12m_related\\03_cma\\regional_laculi_southwest_102008_500\\som\\models\\LOSS_LLAMA\\F18_X40_Y40_E10_CMAX50_20240816-101004\\exports\\plots'
output_pdf_path = 'E:\\20230082_CriticalMAAS\\GitHub\\beak-ta3\\experiments\\04_hackathon_12m_related\\03_cma\\regional_laculi_southwest_102008_500\\som\\models\\LOSS_LLAMA\\F18_X40_Y40_E10_CMAX50_20240816-101004\\exports\\result.pdf'
#folder_path = 'E:\\20230082_CriticalMAAS\\GitHub\\beak-ta3\\experiments\\03_hackathon_9m_related\\03_cma\\tungsten_skarn_ytu\\som\\models\\SOM_FEATURE_FOX_F21_X50_Y50_CMAX50_20240503-175501\\exports\\plots'
#output_pdf_path = 'E:\\20230082_CriticalMAAS\\GitHub\\beak-ta3\\experiments\\03_hackathon_9m_related\\03_cma\\tungsten_skarn_ytu\\som\\models\\SOM_FEATURE_FOX_F21_X50_Y50_CMAX50_20240503-175501\\exports\\result.pdf'
# Call the function
pngs_to_pdf(folder_path, output_pdf_path, 3)



