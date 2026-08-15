"""本地标签 ID 映射。

类别顺序来自 models/chicken-label-12-23.pt 内嵌的 names。模型中三个 `%`
类别的名称末尾意外带有逗号；这里使用数据集和线上展示所采用的规范名称。
"""

COLORS = ("red", "yellow", "blue", "black", "pink")
MARKS = ("@", "#", "$", "%", "&", "1", "2", "4", "B", "C", "M", "X")

label_map = {
    class_id: f"{color}-{mark}"
    for class_id, (color, mark) in enumerate(
        (color, mark) for color in COLORS for mark in MARKS
    )
}
