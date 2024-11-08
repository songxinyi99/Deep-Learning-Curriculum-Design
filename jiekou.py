import argparse
import os
import xml.etree.ElementTree as ET
from ultralytics import YOLOv10


def process_yolo_model(input_dir, model_file):
    model = YOLOv10(model_file)
    results = {}

    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.png')):
            file_path = os.path.join(input_dir, filename)
            detections = model.predict(file_path)
            results[filename] = detections


    return results


def generate_xml(results, output_dir):
    for key, value in results.items():
        # 为每个结果生成一个 XML 文件
        output_file = os.path.join(output_dir, f"{key}.xml")
        for item in value:
            box = item.boxes  # 获取 boxes 属性
        root = ET.Element("annotation")
        # 添加边界框和其他信息
        for row in range(len(box.xyxy)):
            detection = ET.SubElement(root, "object")
            bndbox = ET.SubElement(detection, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(box.xyxy[row][0])
            ET.SubElement(bndbox, "ymin").text = str(box.xyxy[row][1])
            ET.SubElement(bndbox, "xmax").text = str(box.xyxy[row][2])
            ET.SubElement(bndbox, "ymax").text = str(box.xyxy[row][3])
            ET.SubElement(bndbox, "confidence").text = str(box.conf[row])
            ET.SubElement(bndbox, "class").text = str(box.cls[row])

        # 创建 XML 树并写入文件
        tree = ET.ElementTree(root)
        tree.write(output_file)

def main():
    parser = argparse.ArgumentParser(description='Process YOLOv10 model.')
    parser.add_argument('-dir', required=True, help='Input directory containing images')
    parser.add_argument('-model', required=True, help='YOLOv10 model file path')
    parser.add_argument('-out', required=True, help='Output XML file path')

    args = parser.parse_args()

    # 处理模型并获取结果
    results = process_yolo_model(args.dir, args.model)
    # 生成 XML 文件
    generate_xml(results, args.out)

    print(f"Results saved to: {args.out}")


if __name__ == "__main__":
    main()
