import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
UPLOADS_FOLDER = os.path.join(BASE_DIR, 'uploads')
PRESERVE_FOLDERS = ['css', 'js']  # Add folders to preserve here

def clear_folders():
    # Clear static folder
    for filename in os.listdir(STATIC_FOLDER):
        file_path = os.path.join(STATIC_FOLDER, filename)
        try:
            if filename not in PRESERVE_FOLDERS:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    # Clear uploads folder
    for filename in os.listdir(UPLOADS_FOLDER):
        file_path = os.path.join(UPLOADS_FOLDER, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


