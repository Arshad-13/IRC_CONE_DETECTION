import os

def process_folder(folder_path, mapping):
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    print(f"Processing {folder_path} with mapping {mapping}...")
    count = 0
    for filename in os.listdir(folder_path):
        if not filename.endswith(".txt"):
            continue
        
        filepath = os.path.join(folder_path, filename)
        with open(filepath, "r") as f:
            lines = f.readlines()
        
        new_lines = []
        changed = False
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            
            class_id = int(parts[0])
            
            # Check if already converted (heuristic)
            # For cone1: 0 -> 3. If we see 3, assume done.
            # For cone2: 0 -> 2. If we see 2, assume done.
            
            if class_id in mapping:
                new_class_id = mapping[class_id]
                if new_class_id != class_id:
                    parts[0] = str(new_class_id)
                    new_lines.append(" ".join(parts) + "\n")
                    changed = True
                else:
                    new_lines.append(line)
            else:
                # If class_id is already a target value (e.g. 3 in cone1, or 2 in cone2), assume it's processed
                # But wait, what if cone2 has class 2 originally? It didn't.
                new_lines.append(line)
        
        if changed:
            with open(filepath, "w") as f:
                f.writelines(new_lines)
            count += 1
    
    print(f"Updated {count} files in {folder_path}")

def main():
    base_path = "s:/cone"
    
    # Dataset 1: cone1 (Generic Cone)
    # Map 0 -> 3
    cone1_mapping = {0: 3}
    cone1_dirs = [
        "cone1.v1i.yolov11/train/labels",
        "cone1.v1i.yolov11/valid/labels",
        "cone1.v1i.yolov11/test/labels"
    ]
    
    for d in cone1_dirs:
        process_folder(os.path.join(base_path, d), cone1_mapping)

    # Dataset 2: cone2 (Yellow/Red)
    # Map 0 (Yellow) -> 2 (Yellow)
    # Map 1 (Red) -> 1 (Red) - no change needed, but good to be explicit if I was rewriting logic
    cone2_mapping = {0: 2} 
    cone2_dirs = [
        "cone2.v1i.yolov11/train/labels",
        "cone2.v1i.yolov11/valid/labels",
        "cone2.v1i.yolov11/test/labels"
    ]
    
    for d in cone2_dirs:
        process_folder(os.path.join(base_path, d), cone2_mapping)

if __name__ == "__main__":
    main()
