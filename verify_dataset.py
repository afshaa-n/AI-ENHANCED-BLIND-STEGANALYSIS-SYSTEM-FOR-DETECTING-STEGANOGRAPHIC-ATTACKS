import os

def count_images(base_path):
    total = 0
    print(f"\nChecking: {base_path}\n")

    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)

        if os.path.isdir(folder_path):
            count = len([
                f for f in os.listdir(folder_path)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ])

            print(f"{folder}: {count}")
            total += count

    print(f"\nTotal in {base_path}: {total}")
    return total


cover_total = count_images("dataset/cover")
stego_total = count_images("dataset/stego")

print("\n============================")
print("FINAL SUMMARY")
print("============================")
print("Cover Images:", cover_total)
print("Stego Images:", stego_total)

if cover_total == stego_total:
    print("Dataset is balanced and correct.")
else:
    print("Mismatch detected! Check folders.")