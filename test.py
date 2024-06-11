import os

def list_directory_contents(directory_path):
    try:
        contents = os.listdir(directory_path)
        print(f"Contents of '{directory_path}':")
        for item in contents:
            print(item)
    except FileNotFoundError:
        print(f"The directory '{directory_path}' does not exist.")
    except PermissionError:
        print(f"Permission denied to access '{directory_path}'.")

# 列出 ASV 数据集目录的内容
asv_path = 'datasets/ASV'
list_directory_contents(asv_path)

# # 列出 release_in_the_wild 数据集目录的内容
# release_in_the_wild_path = 'datasets/release_in_the_wild'
# list_directory_contents(release_in_the_wild_path)
