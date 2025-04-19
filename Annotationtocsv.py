import os
import xml.etree.ElementTree as ET
import pandas as pd
import openslide

class AnnotationParser:
    def __init__(self, xml_path, slide_path, output_csv_path):
        self.xml_path = xml_path
        self.slide_path = slide_path
        self.output_csv_path = output_csv_path
        self.id_map = {}
        self.color_map = {}

    def parse(self):
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        slide = openslide.OpenSlide(self.slide_path)
        xoffset = float(slide.properties['hamamatsu.XOffsetFromSlideCentre'])
        yoffset = float(slide.properties['hamamatsu.YOffsetFromSlideCentre'])

        for ndpviewstate in root.findall('.//ndpviewstate'):
            id_value = ndpviewstate.get('id')
            
            annotation_node = ndpviewstate.find('annotation')
            if annotation_node is None:
                continue

            color = annotation_node.get('color')
            annotation_type = annotation_node.get('type')
            current_coordinates = []

            if annotation_type == "circle":
                x = float(annotation_node.find('x').text)
                y = float(annotation_node.find('y').text)
                radius = float(annotation_node.find('radius').text)

                center_x = (x - xoffset) / 227 + slide.dimensions[0] / 2
                center_y = -(y - yoffset) / 227 + slide.dimensions[1] / 2
                adjusted_radius = radius / 227
                current_coordinates.append((center_x, center_y, adjusted_radius))

            elif annotation_node.find('pointlist') is not None:
                pointlist_node = annotation_node.find('pointlist')
                point_nodes = pointlist_node.findall('point')

                for point in point_nodes:
                    x_value = (float(point.find('x').text) - xoffset) / 227 + slide.dimensions[0] / 2
                    y_value = -(float(point.find('y').text) - yoffset) / 227 + slide.dimensions[1] / 2
                    current_coordinates.append((x_value, y_value))

            if id_value in self.id_map:
                self.id_map[id_value].extend(current_coordinates)
            else:
                self.id_map[id_value] = current_coordinates
                self.color_map[id_value] = color

        id_list = list(self.id_map.keys())
        color_list = [self.color_map[id] for id in id_list]
        coordinate_matrices = [self.id_map[id] for id in id_list]

        df = pd.DataFrame({
            'ID': id_list,
            'Color': color_list,
            'Coordinates': coordinate_matrices
        })

        color_mapping = {
            '#000000': 'portal vein',
            '#00ff00': 'bile duct',
            '#ff0000': 'artery',
            '#0000ff': 'central vein',
        }

        df['Color'] = df['Color'].replace(color_mapping)

        df['ID'] = df['ID'].astype(float)
        df = df.sort_values(by='ID')
        os.makedirs(os.path.dirname(self.output_csv_path), exist_ok=True)
        df.to_csv(self.output_csv_path, index=False)

#xml_path and slide_path to be replaced with actual path where the .ndpi and .xml file are stored
if __name__ == "__main__":
    xml_path = r"Annotations.xml"
    slide_path = r"SSES2021 14 LIVER LL 14 7 21_J-21-157_4_Pig_GS 1-10000 + PSR_RUN08_Part II_Liver LL 0h_VAB.ndpi"
    output_csv_path = r"Annotations.csv"

    parser = AnnotationParser(xml_path, slide_path, output_csv_path)
    parser.parse()
