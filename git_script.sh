# Multi Human Pose to Body Translation
#!/bin/sh

if [ $# -eq 0 ]
  then
    echo "Please give the commit message. Aborting!"
    exit
fi

#ignore large files. note it appends to gitignore. It might be huge.
find ./* -size +100M | sed 's|^./||'| cat >> .gitignore


echo "__pycache__/" >> .gitignore
echo ".DS_Store" >> .gitignore
echo "._*" >> .gitignore
echo "*.pyc" >> .gitignore
echo "*.ipynb_checkpoints" >> .gitignore
echo "*.so" >> .gitignore


#uncomment if you want to use lfs git
# find ./* -size +100M | cat > .gitattributes

echo "openpose_pytorch/model" >> .gitignore
echo "openpose/output" >> .gitignore


echo "pix2pixHD/datasets" >> .gitignore
echo "pix2pixHD/checkpoints" >> .gitignore
echo "pix2pixHD/results" >> .gitignore

#push using the git command
git add -A
git commit -m "$1"
git push
