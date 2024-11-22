import os

def find_keywords_in_file(file_path, keywords):
    with open(file_path, 'r') as file:
        content = file.read()
        return all(keyword in content for keyword in keywords)

def find_documents_with_keywords(keywords, folders):
    matching_documents = []
    for folder in folders:
        for root, _, files in os.walk(folder):
            for file in files:
                file_path = os.path.join(root, file)
                if find_keywords_in_file(file_path, keywords):
                    matching_documents.append(file_path)
    return matching_documents

if __name__ == "__main__":
    keywords = ["liquidity", "capital", "small bank"]  # Replace with your keywords
    folders = ["./documents/post_SVG", "./documents/pre_SVB"]
    matching_documents = find_documents_with_keywords(keywords, folders)
    print("Documents containing all keywords:", matching_documents)