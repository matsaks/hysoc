import os
import shutil

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, "..")
    processed_dir = os.path.join(project_root, "data", "processed")

    if not os.path.exists(processed_dir):
        print(f"Directory {processed_dir} does not exist.")
        return

    print(f"Clearing contents of {processed_dir} ...")
    
    count_files = 0
    count_dirs = 0

    for item in os.listdir(processed_dir):
        # Optional: Skip .gitkeep if you're tracking the empty dir
        if item == ".gitkeep":
            continue

        item_path = os.path.join(processed_dir, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
                count_files += 1
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                count_dirs += 1
        except Exception as e:
            print(f"Failed to delete {item_path}. Reason: {e}")

    print(f"Successfully deleted {count_files} files and {count_dirs} directories.")
    print("Done!")

if __name__ == "__main__":
    main()
