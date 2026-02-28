wget -O assets.zip "https://www.dropbox.com/scl/fi/oyippb2cw7w8wlzggqjsy/assets.zip?rlkey=1sjgzmmgkedycpimfzy3rpv7s&st=0cfkg1ca&dl=1"
unzip assets.zip
rm assets.zip
python tools/simplify_meshes.py
