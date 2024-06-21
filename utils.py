import os

def save_uploaded_file(file):
    # Create the uploads directory if it doesn't exist
    uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)

    filename = file.filename
    filepath = os.path.join(uploads_dir, filename)
    file.save(filepath)
    return filepath
