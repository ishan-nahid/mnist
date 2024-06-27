import os

def save_uploaded_file(file):
    # Get the directory of the current script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create an 'uploads' directory if it doesn't exist
    upload_dir = os.path.join(base_dir, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    
    # Create a file path
    filename = file.filename
    file_path = os.path.join(upload_dir, filename)
    
    # Save the file
    file.save(file_path)
    
    return file_path