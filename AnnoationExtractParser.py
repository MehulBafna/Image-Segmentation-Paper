import xml.etree.ElementTree as ET
import pandas as pd
import os

OPENSLIDE_PATH = r'C:\Program Files\Openslide\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin'

import os
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


class AnnotationParser:
    def __init__(self, xml_file, slide_path):
        self.xml_file = xml_file
        self.slide_path = slide_path
        self.slide = openslide.OpenSlide(slide_path)
        self.id_map = {}
        self.color_map = {}
        self.xoffset = self.slide.properties['hamamatsu.XOffsetFromSlideCentre']
        self.yoffset = self.slide.properties['hamamatsu.YOffsetFromSlideCentre']

    def parse_annotations(self):
        tree = ET.parse(self.xml_file)
        root = tree.getroot()
        
        for ndpviewstate in root.findall('.//ndpviewstate'):
            id_value = ndpviewstate.get('id')
            annotation_node = ndpviewstate.find('annotation')
            color = annotation_node.get('color')
            pointlist_node = annotation_node.find('pointlist')
            point_nodes = pointlist_node.findall('point')
            current_coordinates = []

            for point in point_nodes:
                x_value = (float(point.find('x').text) - float(self.xoffset)) / 227 + self.slide.dimensions[0] / 2
                y_value = -(float(point.find('y').text) - float(self.yoffset)) / 227 - self.slide.dimensions[1] / 2
                current_coordinates.append((x_value, y_value))

            if id_value in self.id_map:
                self.id_map[id_value].extend(current_coordinates)
            else:
                self.id_map[id_value] = current_coordinates
                self.color_map[id_value] = color

    def create_dataframe(self):
        id_list = list(self.id_map.keys())
        color_list = [self.color_map[id] for id in id_list]
        coordinate_matrices = [self.id_map[id] for id in id_list]
        df = pd.DataFrame({'ID': id_list, 'Color': color_list, 'Coordinates': coordinate_matrices})
        return df

    def map_colors(self, df):
        color_mapping = {
            '#ffff00': 'lobule',
            '#000000': 'portal vein',
            '#00ff00': 'bile duct',
            '#ff0000': 'artery',
            '#0000ff': 'central vein',
        }
        df['Color'] = df['Color'].replace(color_mapping)

    def sort_and_save(self, df, output_csv):
        df['ID'] = df['ID'].astype(float)
        df = df.sort_values(by='ID')
        df.to_csv(output_csv
