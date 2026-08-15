import os
import glob
import random
import shutil
import argparse
from pathlib import Path

def copy_split_dataset(src_images, src_labels, dst_root, ratio=0.8, seed=42, copy=True):
    """
    按比例随机划分并复制图像和标签到目标目录。
    
    Args:
        src_images: 源图像目录
        src_labels: 源标签目录（.txt）
        dst_root: 目标根目录（将创建 images/train, images/val, labels/train, labels/val）
        ratio: 训练集比例，默认0.8
        seed: 随机种子
        copy: True表示复制，False表示移动（剪切）
    """
    # 获取所有标签文件
    label_files = glob.glob(os.path.join(src_labels, "*.txt"))
    if not label_files:
        print(f"错误：在 {src_labels} 中未找到任何 .txt 文件")
        return

    # 提取基名（不含扩展名）
    base_names = [Path(f).stem for f in label_files]

    # 检查图像是否存在，支持多种扩展名
    image_extensions = ['.jpg', '.jpeg', '.png']
    valid_pairs = []  # 存储 (基名, 图像完整路径)
    for name in base_names:
        found_img = None
        for ext in image_extensions:
            img_path = os.path.join(src_images, name + ext)
            if os.path.exists(img_path):
                found_img = img_path
                break
        if found_img:
            valid_pairs.append((name, found_img))
        else:
            print(f"警告：标签 {name}.txt 未找到对应的图像文件，已跳过")

    total = len(valid_pairs)
    if total == 0:
        print("错误：没有有效的样本")
        return

    print(f"总有效样本数：{total}")

    # 随机打乱
    random.seed(seed)
    shuffled = valid_pairs.copy()
    random.shuffle(shuffled)

    # 分割
    split_idx = int(total * ratio)
    train_pairs = shuffled[:split_idx]
    val_pairs = shuffled[split_idx:]

    # 创建目标目录结构
    for sub in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        os.makedirs(os.path.join(dst_root, sub), exist_ok=True)

    # 复制函数
    def copy_or_move(src, dst, is_copy):
        if is_copy:
            shutil.copy2(src, dst)  # 复制并保留元数据
        else:
            shutil.move(src, dst)   # 移动（剪切）

    # 处理训练集
    for name, img_path in train_pairs:
        # 复制图像
        dst_img = os.path.join(dst_root, 'images', 'train', os.path.basename(img_path))
        copy_or_move(img_path, dst_img, copy)
        # 复制标签
        src_label = os.path.join(src_labels, name + '.txt')
        dst_label = os.path.join(dst_root, 'labels', 'train', name + '.txt')
        copy_or_move(src_label, dst_label, copy)

    # 处理验证集
    for name, img_path in val_pairs:
        dst_img = os.path.join(dst_root, 'images', 'val', os.path.basename(img_path))
        copy_or_move(img_path, dst_img, copy)
        src_label = os.path.join(src_labels, name + '.txt')
        dst_label = os.path.join(dst_root, 'labels', 'val', name + '.txt')
        copy_or_move(src_label, dst_label, copy)

    print(f"训练集样本数：{len(train_pairs)}，已复制到 {dst_root}/images/train 和 {dst_root}/labels/train")
    print(f"验证集样本数：{len(val_pairs)}，已复制到 {dst_root}/images/val 和 {dst_root}/labels/val")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="按比例随机划分并复制/移动数据集到新的目录结构")
    parser.add_argument("--src_images", required=True, help="源图像目录")
    parser.add_argument("--src_labels", required=True, help="源标签目录（YOLO格式的.txt文件）")
    parser.add_argument("--dst_root", required=True, help="目标根目录，将在其下创建 images/ 和 labels/ 子目录")
    parser.add_argument("--ratio", type=float, default=0.8, help="训练集比例，默认0.8")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认42")
    parser.add_argument("--move", action="store_true", help="使用移动（剪切）而不是复制，注意会删除源文件")

    args = parser.parse_args()

    copy_split_dataset(
        src_images=args.src_images,
        src_labels=args.src_labels,
        dst_root=args.dst_root,
        ratio=args.ratio,
        seed=args.seed,
        copy=not args.move   # 默认复制，如果指定 --move 则为移动
    )