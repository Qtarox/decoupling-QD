import os
from pathlib import Path
# 1. 设置你要处理的文件夹路径 (注意替换为你的实际路径)
folder_path = Path('./inequalities/')

# 2. 获取该文件夹下所有的文件和文件夹名称
files = os.listdir(folder_path)

# 3. 遍历并重命名
for file_path in folder_path.iterdir():
    if file_path.is_file():
        old_name = file_path.name
        
        # 判断条件：名字里有 ',' 或者有 '_superball'
        if ',' in old_name or '_superball' in old_name:
            
            # 1. 把 ',' 替换成 '.'
            new_name = old_name.replace(',', '.')
            
            # 2. 把 '_superball' 删掉（替换为空）
            new_name = new_name.replace('_superball', '')
            
            new_file_path = file_path.with_name(new_name)
            
            file_path.rename(new_file_path)
            print(f"✅ 成功将: '{old_name}' \n   重命名为 -> '{new_name}'\n")

print("🎉 字符替换与删除完成！")