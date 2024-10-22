import os
import re
from PIL import Image
from fpdf import FPDF

# Function for natural sorting (i.e., handling numbers properly)
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def get_cluster_results(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    
    # Separate files by pattern and naturally sort them
    somplot_images = sorted([f for f in files if f.startswith('somplot_')], key=natural_sort_key)
    geoplot_images = sorted([f for f in files if f.startswith('geoplot_')], key=natural_sort_key)
    cluster_images = [f for f in files if f.startswith('cluster_') or f.startswith('bmu')]
    
    cluster_results = []

    # Add specific somplot images to "Cluster results"
    if len(somplot_images) >= 4:
        cluster_results.extend(somplot_images[-3:-2])
        #cluster_results.extend(somplot_images[-1:])
        cluster_results.extend(somplot_images[-4:-3])
        cluster_results.extend(somplot_images[-2:-1])
        ## Add all cluster and BMU images to "Cluster results"
        #cluster_results.extend(cluster_images)
    
    # Add all cluster and BMU images to "Cluster results"
    cluster_results.extend(cluster_images)

    # Add first and last geoplot images to "Cluster results"
    if len(geoplot_images) > 0:
        cluster_results.append(geoplot_images[0])  # First geoplot image
        if len(geoplot_images) > 1:
            cluster_results.append(geoplot_images[-1])  # Last geoplot image
    
    return [os.path.join(folder_path, img) for img in cluster_results]

def pngs_to_pdf(folder_paths, run_names, title_name, output_pdf_path, number_of_columns):
    pdf = FPDF('P', 'mm', 'A4')  # Create a PDF object, A4 size in portrait mode
     
    # Page size in mm
    page_width = 210  # A4 width in mm
    page_height = 297  # A4 height in mm
    margin = 10  # Margin around the images
    space = 5   # Space between images 
    
    # Calculate available width for images (page width minus margins)
    available_width = page_width - 2 * margin - (number_of_columns - 1) * space
    img_width_mm = available_width / number_of_columns  # Image width to fit in one row

    # Retrieve "Cluster results" images for each folder
    cluster_results_by_model = [get_cluster_results(folder) for folder in folder_paths]

    # Assuming that all model folders have the same number of cluster images
    num_images_per_model = min(len(images) for images in cluster_results_by_model)

    # Add a new page for "Cluster results"
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt= title_name, ln=True, align='C')
    pdf.ln(10)

    y_position = pdf.get_y()

   # Add custom RUN names at the top of each column
    x_position = margin
    pdf.set_font("Arial", size=12)

    #for run_name in run_names:
    #    pdf.cell(img_width_mm, 10, txt=run_name, ln=False, align='C')
    #    x_position += img_width_mm + space

    for run_name in run_names:
        text_width = pdf.get_string_width(run_name)
        centered_x_position = x_position + (img_width_mm - text_width) / 2
        pdf.set_xy(centered_x_position, y_position)
        pdf.cell(text_width, 10, txt=run_name, ln=False, align='C')
        x_position += img_width_mm + space
    
    pdf.ln(15)  # Move down after the titles
    y_position = pdf.get_y()

    for img_index in range(num_images_per_model):
        x_position = margin

        for model_images in cluster_results_by_model:
            img_path = model_images[img_index]
            
            # Open the image and calculate the aspect ratio
            with Image.open(img_path) as img:
                img_width, img_height = img.size
                aspect_ratio = img_height / img_width
                img_height_mm = img_width_mm * aspect_ratio
            
            # Check if the image fits on the current page, if not, add a new page
            if y_position + img_height_mm > page_height - margin:
                pdf.add_page()
                y_position = margin
                x_position = margin
                pdf.set_y(y_position)
            
            # Add the image to the PDF
            pdf.image(img_path, x=x_position, y=y_position, w=img_width_mm)
            
            # Move to the next column
            x_position += img_width_mm + space
        
        y_position += img_height_mm + space  # Move down for the next row of images

    # Save the PDF file
    pdf.output(output_pdf_path)
    print(f"PDF successfully created: {output_pdf_path}")
