import os
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom

class LabelFormatter:
    
    @staticmethod
    def save(format_type, boxes, labels, filename, output_dir, img_w, img_h, class_map=None):
        save_path = os.path.join(output_dir, "labels")
        os.makedirs(save_path, exist_ok=True)

        mapping = class_map or {}

        if format_type == "yolo":
            LabelFormatter._save_yolo(boxes, labels, filename, save_path, img_w, img_h, mapping)
        elif format_type == "voc":
            LabelFormatter._save_voc(boxes, labels, filename, save_path, img_w, img_h, mapping)
        elif format_type == "json":
            LabelFormatter._save_json(boxes, labels, filename, save_path, img_w, img_h, mapping)
        else:
            raise ValueError(f"unsupported format: {format_type}")

    @staticmethod
    def _resolve_class_id(label, class_map):
        clean_label = str(label).strip()
        
        if clean_label in class_map:
            return class_map[clean_label]
        
        if len(class_map) == 1:
            return list(class_map.values())[0]
            
        return -1

    @staticmethod
    def _save_yolo(boxes, labels, filename, save_path, img_w, img_h, class_map):
        lines = []
        
        for box, label in zip(boxes, labels):
            class_id = LabelFormatter._resolve_class_id(label, class_map)
            
            if class_id == -1:
                continue

            x1, y1, x2, y2 = box
            
            # normalize
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            xc = (x1 + (x2 - x1) / 2) / img_w
            yc = (y1 + (y2 - y1) / 2) / img_h
            
            # clamping 0-1
            xc = max(0, min(1, xc))
            yc = max(0, min(1, yc))
            w = max(0, min(1, w))
            h = max(0, min(1, h))
            
            lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            
        with open(os.path.join(save_path, f"{filename}.txt"), "w") as f:
            f.write("\n".join(lines))

    @staticmethod
    def _save_voc(boxes, labels, filename, save_path, img_w, img_h, class_map):
        root = ET.Element("annotation")
        ET.SubElement(root, "filename").text = f"{filename}.jpg"
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(img_w)
        ET.SubElement(size, "height").text = str(img_h)

        for box, label in zip(boxes, labels):
            class_id = LabelFormatter._resolve_class_id(label, class_map)
            
            if class_id == -1: continue

            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = str(class_id)
            
            bbox = ET.SubElement(obj, "bbox")
            ET.SubElement(bbox, "xmin").text = str(int(box[0]))
            ET.SubElement(bbox, "ymin").text = str(int(box[1]))
            ET.SubElement(bbox, "xmax").text = str(int(box[2]))
            ET.SubElement(bbox, "ymax").text = str(int(box[3]))

        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
        with open(os.path.join(save_path, f"{filename}.xml"), "w") as f:
            f.write(xml_str)

    @staticmethod
    def _save_json(boxes, labels, filename, save_path, img_w, img_h, class_map):
        data = {
            "image": f"{filename}.jpg",
            "width": img_w,
            "height": img_h,
            "annotations": []
        }
        
        for box, label in zip(boxes, labels):
            class_id = LabelFormatter._resolve_class_id(label, class_map)
            
            if class_id == -1: continue
            
            data["annotations"].append({
                "label": class_id,
                "bbox": [int(x) for x in box]
            })
            
        with open(os.path.join(save_path, f"{filename}.json"), "w") as f:
            json.dump(data, f, indent=4)