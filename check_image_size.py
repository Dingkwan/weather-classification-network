import os
import json
from pathlib import Path
from PIL import Image
import numpy as np

def collect_image_sizes(data_dir, output_json="image_sizes.json", stats_json="image_stats.json"):
    data_dir = Path(data_dir)
    results = []
    widths, heights = [], []

    # 遍历所有子文件夹和图片
    for img_path in data_dir.rglob("*.*"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue
        try:
            with Image.open(img_path) as img:
                width, height = img.size
            results.append({
                "filename": str(img_path.relative_to(data_dir)),  # 相对路径
                "size": {"width": width, "height": height}
            })
            widths.append(width)
            heights.append(height)
        except Exception as e:
            print(f"Error reading {img_path}: {e}")

    # 保存每张图片的尺寸
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # 计算统计信息
    if widths and heights:
        stats = {
            "total_images": len(widths),
            "width": {
                "min": int(np.min(widths)),
                "max": int(np.max(widths)),
                "mean": float(np.mean(widths))
            },
            "height": {
                "min": int(np.min(heights)),
                "max": int(np.max(heights)),
                "mean": float(np.mean(heights))
            }
        }
        with open(stats_json, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)

        print(f"统计完成 ✅ 共 {stats['total_images']} 张图片")
        print(f"宽度: min={stats['width']['min']}, max={stats['width']['max']}, mean={stats['width']['mean']:.2f}")
        print(f"高度: min={stats['height']['min']}, max={stats['height']['max']}, mean={stats['height']['mean']:.2f}")
        print(f"结果已保存到 {output_json} 和 {stats_json}")
    else:
        print("没有找到符合条件的图片。")

if __name__ == "__main__":
    # 修改成你的数据集路径
    collect_image_sizes(data_dir="/Users/dingkwanmok/Desktop/test/split_dataset", 
                        output_json="image_sizes.json",
                        stats_json="image_stats.json")
