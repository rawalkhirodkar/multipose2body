# Multi Human Pose to Body Translation
#!/bin/sh

if [ $# -eq 0 ]
  then
    echo "Please give the commit message. Aborting!"
    exit
fi

#ignore large files. note it appends to gitignore. It might be huge.
find ./* -size +100M | sed 's|^./||'| cat >> .gitignore


echo "hrnet/Outputs" >> .gitignore
echo "hrnet/models" >> .gitignore
echo "hrnet/data" >> .gitignore
echo "hrnet/cache" >> .gitignore
echo "syncrowd/Outputs" >> .gitignore
echo "syncrowd/data" >> .gitignore
echo "detectron2/models" >> .gitignore

echo "__pycache__/" >> .gitignore
echo ".DS_Store" >> .gitignore
echo "._*" >> .gitignore
echo "*.pyc" >> .gitignore
echo "*.ipynb_checkpoints" >> .gitignore
echo "*.so" >> .gitignore


echo "posefix/data" >> .gitignore
echo "posefix/output" >> .gitignore

#uncomment if you want to use lfs git
# find ./* -size +100M | cat > .gitattributes


#push using the git command
git add -A
git commit -m "$1"
git push