import os
# import shutil

# # 设置当前工作目录为包含文本文件的目录
# current_directory = '/path/to/your/text/files'  # 请将这里的路径替换为你的文本文件所在的目录
# files = os.listdir(current_directory)

# for file_name in files:
#     if file_name.endswith('.txt'):
#         # 创建与文件同名的文件夹
#         folder_name = os.path.splitext(file_name)[0]
#         folder_path = os.path.join(current_directory, folder_name)
        
#         # 如果文件夹不存在，则创建它
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)
        
#         # 移动文件到新创建的文件夹
#         shutil.move(os.path.join(current_directory, file_name), folder_path)

# print("所有文本文件已成功移动到对应的文件夹中。")
# 除去文件夹名字中有.mp4
def rename_subdirectories(target_dir):
    for dirpath, dirnames, filenames in os.walk(target_dir):
        for dirname in dirnames:
            if '.mp4' in dirname:
                old_path = os.path.join(dirpath, dirname)
                new_dirname = dirname.replace('.mp4', '')
                new_path = os.path.join(dirpath, new_dirname)
                os.rename(old_path, new_path)
                print(f'Renamed: {old_path} to {new_path}')

# 使用方法
# 将'your/directory/path'替换为你要处理的目录路径
rename_subdirectories('/data/lyh/Affwild2/6th_ABAW_Annotations/VA')
