import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
from skimage import measure, morphology
import os

def get_structure_data(image, color_definitions, lobule_image_path=None):
    """Extract structure data including centroids and measurements"""
    
    centroids_by_structure = {}
    structure_data = {}
    regions_by_structure = {}
    
    for structure_name, color_info in color_definitions.items():
        # Special handling for lobules - use separate image
        if structure_name == 'Lobule' and lobule_image_path:
            print(f"Processing lobules from {lobule_image_path}")
            lobule_image = cv2.imread(lobule_image_path)
            if lobule_image is not None:
                lobule_image_rgb = cv2.cvtColor(lobule_image, cv2.COLOR_BGR2RGB)
                current_image = lobule_image_rgb
            else:
                print(f"Could not load lobule image, using blended image")
                current_image = image
        else:
            current_image = image
            
        target_color = np.array(color_info['color'])
        tolerance = color_info['tolerance']
        
        # Create mask
        color_diff = np.linalg.norm(current_image - target_color, axis=2)
        mask = color_diff <= tolerance
        
        # Clean up mask
        mask = morphology.remove_small_objects(mask, min_size=100)
        mask = morphology.binary_closing(mask, morphology.disk(5))
        
        # Get regions
        labeled_mask = measure.label(mask)
        regions = measure.regionprops(labeled_mask)
        
        centroids = []
        areas = []
        equivalent_diameters = []
        perimeters = []
        major_axis_lengths = []
        minor_axis_lengths = []
        filtered_regions = []
        
        for region in regions:
            # Use Feret diameter for all structures
            diameter_measure = region.feret_diameter_max
            
            # Apply area-based filters
            include_structure = False
            area_measure = region.area  # Area in pixels²
            
            # Get area limits from color_definitions
            min_area = color_info.get('min_area', 0)
            max_area = color_info.get('max_area', float('inf'))
            
            # Check if area is within range
            include_structure = min_area <= area_measure <= max_area
            
            if include_structure:
                centroids.append([region.centroid[1], region.centroid[0]])  # (x, y)
                areas.append(region.area)
                # Use Feret diameter for all structures
                equivalent_diameters.append(region.feret_diameter_max)
                perimeters.append(region.perimeter)
                major_axis_lengths.append(region.major_axis_length)
                minor_axis_lengths.append(region.minor_axis_length)
                filtered_regions.append(region)
        
        centroids_by_structure[structure_name] = np.array(centroids) if centroids else np.array([]).reshape(0, 2)
        regions_by_structure[structure_name] = filtered_regions
        
        if len(centroids) > 0:
            structure_data[structure_name] = {
                'areas': areas,
                'equivalent_diameters': equivalent_diameters,
                'perimeters': perimeters,
                'major_axis_lengths': major_axis_lengths,
                'minor_axis_lengths': minor_axis_lengths,
                'centroids': centroids,  # Store centroids for CSV
                'count': len(centroids),
                'total_detected': len(centroids),
                'filtered_out': 0,
                'filter_type': 'area_range',
                'min_threshold': color_info.get('min_area', 0),
                'max_threshold': color_info.get('max_area', float('inf'))
            }
    
    return centroids_by_structure, structure_data, regions_by_structure

def create_morphometric_analysis_plot(structure_data, centroids_by_structure, regions_by_structure, lobule_image_path=None):
    """
    Create standard morphometric analysis plot with 6 subplots
    """
    # Conversion factor: 227 nm/pixel = 0.227 μm/pixel
    pixel_to_um = 0.227
    
    # Set larger font sizes
    FONT_SIZE = 18
    plt.rc('font', size=FONT_SIZE-2)
    plt.rc('axes', titlesize=FONT_SIZE)
    plt.rc('axes', labelsize=FONT_SIZE-2)
    plt.rc('xtick', labelsize=FONT_SIZE-4)
    plt.rc('ytick', labelsize=FONT_SIZE-4)
    plt.rc('legend', fontsize=FONT_SIZE-4)
    plt.rc('figure', titlesize=FONT_SIZE+2)
    
    # Create figure with 3x2 layout
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Morphometric Analysis', fontsize=FONT_SIZE+2, fontweight='bold')
    
    # Define structure order and colors
    structure_order = ['Lobule', 'PV (Portal Vein)', 'Artery', 'Bile Duct', 'Central Vein']
    structure_colors = {
        'PV (Portal Vein)': 'blue',
        'Artery': 'red',
        'Central Vein': 'green',
        'Bile Duct': 'purple',
        'Lobule': 'yellow'
    }
    
    # Top row, left (0,0): Structure Count
    ax1 = axes[0, 0]
    structure_names = []
    counts = []
    colors = []
    
    for name in structure_order:
        if name in structure_data:
            structure_names.append(name.replace(' ', '\n'))
            counts.append(structure_data[name]['count'])
            colors.append(structure_colors.get(name, 'gray'))
    
    bars = ax1.bar(structure_names, counts, color=colors)
    ax1.set_title('Total Structure Count', fontweight='bold')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=30, labelsize=12)
    
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.02,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    ax1.set_ylim(0, max(counts) * 1.15)
    
    # Top row, right (0,1): Shape Analysis (Aspect Ratio)
    ax2 = axes[0, 1]
    
    # Calculate aspect ratios
    aspect_dict = {}
    color_dict = {}
    
    for name, data in structure_data.items():
        if data['count'] > 0:
            # Calculate aspect ratio as minor/major (inverse of traditional)
            aspect_ratios = [minor/major if major > 0 else 1.0 
                           for major, minor in zip(data['major_axis_lengths'], data['minor_axis_lengths'])]
            aspect_dict[name] = aspect_ratios
            color_dict[name] = structure_colors.get(name, 'gray')
    
    if aspect_dict:
        sorted_data = []
        sorted_labels = []
        sorted_colors = []
        
        for name in structure_order:
            if name in aspect_dict:
                sorted_data.append(aspect_dict[name])
                sorted_labels.append(name.replace(' ', '\n'))
                sorted_colors.append(color_dict[name])
        
        bp2 = ax2.boxplot(sorted_data, labels=sorted_labels, patch_artist=True)
        for patch, color in zip(bp2['boxes'], sorted_colors):
            patch.set_facecolor(color)
        
        ax2.set_title('Shape Analysis (Aspect Ratio)', fontweight='bold')
        ax2.set_ylabel('Aspect Ratio (Minor/Major Axis)')
        ax2.tick_params(axis='x', rotation=30, labelsize=12)
        ax2.axhline(y=1.0, color='blue', linestyle='--', alpha=0.7, label='Perfect Circle')
        ax2.legend()
        # Linear scale instead of log scale
        ax2.set_ylim(0, 1.1)  # 0 to 1.1 range
        ax2.grid(True, alpha=0.3)
        
        # Set y-ticks at 0.2 intervals
        ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    
    # Middle left (1,0): Feret Diameter Major Axis
    ax3 = axes[1, 0]
    major_axis_dict = {}
    color_dict = {}
    
    for name, data in structure_data.items():
        if data['count'] > 0:
            # Correct for half resolution then convert to μm
            major_axes = [d * 2 * pixel_to_um for d in data['major_axis_lengths']]
            major_axis_dict[name] = major_axes
            color_dict[name] = structure_colors.get(name, 'gray')
    
    if major_axis_dict:
        sorted_data = []
        sorted_labels = []
        sorted_colors = []
        
        for name in structure_order:
            if name in major_axis_dict:
                sorted_data.append(major_axis_dict[name])
                sorted_labels.append(name.replace(' ', '\n'))
                sorted_colors.append(color_dict[name])
        
        bp3 = ax3.boxplot(sorted_data, labels=sorted_labels, patch_artist=True)
        for patch, color in zip(bp3['boxes'], sorted_colors):
            patch.set_facecolor(color)
        
        ax3.set_title('Size Distribution (Feret Diameter Major Axis)', fontweight='bold')
        ax3.set_ylabel('Feret Diameter Major Axis (μm)')
        ax3.tick_params(axis='x', rotation=30, labelsize=12)
        ax3.set_yscale('log')
        ax3.set_ylim(10, 10000)
        ax3.grid(True, alpha=0.3, which='both')
        
        from matplotlib.ticker import FixedLocator, FuncFormatter
        ax3.yaxis.set_major_locator(FixedLocator([10, 100, 1000, 10000]))
        ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}'))
        ax3.minorticks_on()
    
    # Middle right (1,1): Feret Diameter Minor Axis
    ax4 = axes[1, 1]
    minor_axis_dict = {}
    color_dict = {}
    
    for name, data in structure_data.items():
        if data['count'] > 0:
            # Correct for half resolution then convert to μm
            minor_axes = [d * 2 * pixel_to_um for d in data['minor_axis_lengths']]
            minor_axis_dict[name] = minor_axes
            color_dict[name] = structure_colors.get(name, 'gray')
    
    if minor_axis_dict:
        sorted_data = []
        sorted_labels = []
        sorted_colors = []
        
        for name in structure_order:
            if name in minor_axis_dict:
                sorted_data.append(minor_axis_dict[name])
                sorted_labels.append(name.replace(' ', '\n'))
                sorted_colors.append(color_dict[name])
        
        bp4 = ax4.boxplot(sorted_data, labels=sorted_labels, patch_artist=True)
        for patch, color in zip(bp4['boxes'], sorted_colors):
            patch.set_facecolor(color)
        
        ax4.set_title('Size Distribution (Feret Diameter Minor Axis)', fontweight='bold')
        ax4.set_ylabel('Feret Diameter Minor Axis (μm)')
        ax4.tick_params(axis='x', rotation=30, labelsize=12)
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3, which='both')
        
        from matplotlib.ticker import LogLocator, FuncFormatter
        ax4.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
        ax4.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}' if x < 1e6 else f'{x:.0e}'))
        ax4.minorticks_on()
    
    # Bottom left (2,0): Perimeter Distribution
    ax5 = axes[2, 0]
    perimeter_dict = {}
    color_dict = {}
    
    for name, data in structure_data.items():
        if data['count'] > 0:
            # Correct for half resolution then convert to μm
            perimeters = np.array(data['perimeters']) * 2 * pixel_to_um
            perimeter_dict[name] = perimeters
            color_dict[name] = structure_colors.get(name, 'gray')
    
    if perimeter_dict:
        sorted_data = []
        sorted_labels = []
        sorted_colors = []
        
        for name in structure_order:
            if name in perimeter_dict:
                sorted_data.append(perimeter_dict[name])
                sorted_labels.append(name.replace(' ', '\n'))
                sorted_colors.append(color_dict[name])
        
        bp = ax5.boxplot(sorted_data, labels=sorted_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], sorted_colors):
            patch.set_facecolor(color)
        
        ax5.set_title('Perimeter Distribution', fontweight='bold')
        ax5.set_ylabel('Perimeter (μm)')
        ax5.tick_params(axis='x', rotation=30, labelsize=12)
        ax5.set_yscale('log')
        ax5.grid(True, alpha=0.3, which='both')
        
        from matplotlib.ticker import LogLocator, FuncFormatter
        ax5.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
        ax5.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}' if x < 1e6 else f'{x:.0e}'))
        ax5.minorticks_on()
    
    # Bottom right (2,1): Area Distribution
    ax6 = axes[2, 1]
    area_dict = {}
    color_dict = {}
    
    for name, data in structure_data.items():
        if data['count'] > 0:
            # Correct for half resolution then convert to μm²
            areas = np.array(data['areas']) * 4 * (pixel_to_um**2)
            area_dict[name] = areas
            color_dict[name] = structure_colors.get(name, 'gray')
    
    if area_dict:
        sorted_data = []
        sorted_labels = []
        sorted_colors = []
        
        for name in structure_order:
            if name in area_dict:
                sorted_data.append(area_dict[name])
                sorted_labels.append(name.replace(' ', '\n'))
                sorted_colors.append(color_dict[name])
        
        bp6 = ax6.boxplot(sorted_data, labels=sorted_labels, patch_artist=True)
        for patch, color in zip(bp6['boxes'], sorted_colors):
            patch.set_facecolor(color)
        
        ax6.set_title('Area Distribution', fontweight='bold')
        ax6.set_ylabel('Area (μm²)')
        ax6.tick_params(axis='x', rotation=30, labelsize=12)
        ax6.set_yscale('log')
        ax6.grid(True, alpha=0.3, which='both')
        
        from matplotlib.ticker import LogLocator, FuncFormatter
        ax6.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
        ax6.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}' if x < 1e6 else f'{x:.0e}'))
        ax6.minorticks_on()
    
    plt.tight_layout()
    return fig

def save_structure_csvs(structure_data, regions_by_structure, output_dir='medical_structure_plots'):
    """Save CSV files for each structure type"""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Conversion factor: 227 nm/pixel = 0.227 μm/pixel
    pixel_to_um = 0.227
    
    for structure_name, data in structure_data.items():
        if data['count'] > 0:
            # Get regions for this structure to extract Feret diameters
            regions = regions_by_structure.get(structure_name, [])
            
            csv_data = []
            for i in range(data['count']):
                if i < len(regions):
                    region = regions[i]
                    # Get actual Feret diameters from region properties
                    feret_major = region.feret_diameter_max
                    # Calculate minor Feret diameter from major axis length (approximation)
                    feret_minor = region.minor_axis_length
                else:
                    feret_major = data['equivalent_diameters'][i]
                    feret_minor = 0
                
                # Get coordinates
                if 'centroids' in data and i < len(data['centroids']):
                    x, y = data['centroids'][i]
                else:
                    x, y = 0, 0
                
                # Calculate measurements in μm and μm² (same scaling for all structures)
                feret_major_um = feret_major * 2 * pixel_to_um
                feret_minor_um = feret_minor * 2 * pixel_to_um
                equiv_diam_um = data['equivalent_diameters'][i] * 2 * pixel_to_um
                perimeter_um = data['perimeters'][i] * 2 * pixel_to_um
                area_um2 = data['areas'][i] * 4 * (pixel_to_um**2)
                
                row = {
                    'ID': i + 1,
                    'Feret_Diameter_Major_px': feret_major,
                    'Feret_Diameter_Minor_px': feret_minor,
                    'Equivalent_Diameter_px': data['equivalent_diameters'][i],
                    'Perimeter_px': data['perimeters'][i],
                    'Area_px': data['areas'][i],
                    'Feret_Diameter_Major_um': feret_major_um,
                    'Feret_Diameter_Minor_um': feret_minor_um,
                    'Equivalent_Diameter_um': equiv_diam_um,
                    'Perimeter_um': perimeter_um,
                    'Area_um2': area_um2,
                    'Center_X': x,  # Original pixel coordinates
                    'Center_Y': y    # Original pixel coordinates
                }
                csv_data.append(row)
            
            import pandas as pd
            df = pd.DataFrame(csv_data)
            csv_filename = f"{output_dir}/{structure_name.replace(' ', '_').replace('(', '').replace(')', '')}.csv"
            df.to_csv(csv_filename, index=False)
            print(f"CSV saved: {csv_filename}")

def save_morphometric_analysis(structure_data, centroids_by_structure, regions_by_structure, lobule_image_path=None, output_dir='medical_structure_plots'):
    """Save morphometric analysis plots and CSV files"""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create and save ONLY the standard morphometric plot
    fig1 = create_morphometric_analysis_plot(
        structure_data, 
        centroids_by_structure, 
        regions_by_structure,
        lobule_image_path
    )
    
    if fig1:
        fig1.savefig(f'{output_dir}/morphometric_analysis_standard.png', 
                    dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print(f"Standard morphometric plot saved to {output_dir}/morphometric_analysis_standard.png")
    
    # Save CSV files for all structures
    save_structure_csvs(structure_data, regions_by_structure, output_dir)
    
    return True

def main():
    """Run the final plotting with combined approach, large fonts, and measurements in μm"""
    
    # Path to the images
    lobule_image_path = "/work/xi47luy/smooth_traversal_no_holes.jpg"
    blended_image_path = "/work/xi47luy/blend4.jpg"
    
    # Define color definitions for structure identification
    # Using area-based filtering in pixels²
    color_definitions = {
        'PV (Portal Vein)': {'color': [0, 0, 255], 'tolerance': 30, 'min_area': 5000},
        'Artery': {'color': [255, 0, 0], 'tolerance': 30, 'min_area': 600},
        'Lobule': {'color': [255, 255, 0], 'tolerance': 30, 'min_area': 60000, 'max_area': 6000000},
        'Bile Duct': {'color': [128, 0, 128], 'tolerance': 30, 'min_area': 600},
        'Central Vein': {'color': [0, 255, 0], 'tolerance': 30, 'min_area': 600}
    }
    
    try:
        # Process the blended image to get structure centroids
        print(f"Processing blended image from {blended_image_path}")
        blended_image = cv2.imread(blended_image_path)
        if blended_image is None:
            print(f"Error: Could not load image from {blended_image_path}")
            return
        
        blended_image = cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB)
        
        # Get structure data including measurements
        print("Extracting structure data...")
        centroids_by_structure, structure_data, regions_by_structure = get_structure_data(blended_image, color_definitions, lobule_image_path)
        
        # Create and save morphometric analysis plots
        print("Creating and saving morphometric analysis plots...")
        save_morphometric_analysis(structure_data, centroids_by_structure, regions_by_structure, lobule_image_path)
        
        print("Done! Check the 'medical_structure_plots' directory for all plots.")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()
